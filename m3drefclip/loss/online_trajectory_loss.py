import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlineTrajectoryLoss(nn.Module):
    """Supervise target survival and distractor removal across query updates."""

    def __init__(
        self,
        survival_weight=1.0,
        prune_weight=0.5,
        ranking_weight=1.0,
        ranking_margin=0.2,
        coverage_weight=0.0,
        candidate_set_weight=0.0,
        raw_candidate_set_weight=0.0,
        initial_survival_weight=0.0,
        transition_weight=0.0,
        rank_improvement_weight=0.0,
        dpo_weight=0.0,
        kl_weight=0.0,
        coverage_positive_weight=1.0,
        coverage_negative_weight=1.0,
        candidate_set_positive_weight=None,
        candidate_set_negative_weight=None,
        kl_detach_raw=True,
        kl_initial_only=True,
        dpo_beta=0.1,
        decision_threshold=0.5,
        independent_rounds=False,
        detach_temporal_baseline=False,
        eps=1e-6,
        **kwargs,
    ):
        super().__init__()
        self.survival_weight = float(survival_weight)
        self.prune_weight = float(prune_weight)
        self.ranking_weight = float(ranking_weight)
        self.ranking_margin = float(ranking_margin)
        self.coverage_weight = float(coverage_weight)
        self.candidate_set_weight = float(candidate_set_weight)
        self.raw_candidate_set_weight = float(raw_candidate_set_weight)
        self.initial_survival_weight = float(initial_survival_weight)
        self.transition_weight = float(transition_weight)
        self.rank_improvement_weight = float(rank_improvement_weight)
        self.dpo_weight = float(dpo_weight)
        self.kl_weight = float(kl_weight)
        self.coverage_positive_weight = float(coverage_positive_weight)
        self.coverage_negative_weight = float(coverage_negative_weight)
        self.candidate_set_positive_weight = float(
            coverage_positive_weight
            if candidate_set_positive_weight is None
            else candidate_set_positive_weight
        )
        self.candidate_set_negative_weight = float(
            coverage_negative_weight
            if candidate_set_negative_weight is None
            else candidate_set_negative_weight
        )
        self.kl_detach_raw = bool(kl_detach_raw)
        self.kl_initial_only = bool(kl_initial_only)
        self.dpo_beta = float(dpo_beta)
        self.decision_threshold = float(decision_threshold)
        self.independent_rounds = bool(independent_rounds)
        self.detach_temporal_baseline = bool(detach_temporal_baseline)
        if not 0.0 < self.decision_threshold < 1.0:
            raise ValueError("decision_threshold must be in (0, 1).")
        self.eps = float(eps)

    @staticmethod
    def _last_round(values, round_mask):
        last_round = round_mask.long().sum(dim=-1).sub(1).clamp_min(0)
        return values.gather(
            2,
            last_round.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, values.shape[-1]),
        ).squeeze(2), last_round

    @staticmethod
    def _target_logprob(scores, labels, valid_mask):
        min_value = torch.finfo(scores.dtype).min
        denominator = torch.logsumexp(scores.masked_fill(~valid_mask, min_value), dim=-1)
        numerator = torch.logsumexp(scores.masked_fill(~labels, min_value), dim=-1)
        return numerator - denominator

    def forward(
        self,
        alive_history,
        score_history,
        target_labels,
        proposal_mask,
        round_mask,
        raw_score_history=None,
        compatible_labels=None,
        candidate_labels=None,
        feedback=None,
        negative_score_history=None,
        negative_round_mask=None,
    ):
        if alive_history.dim() != 4:
            raise ValueError("alive_history must have shape [B, L, T, C].")
        labels = target_labels.bool()
        valid_candidates = proposal_mask.bool().unsqueeze(1).expand_as(labels)
        expanded_valid = valid_candidates.unsqueeze(2).expand_as(score_history)
        # Padded external-backbone logits may use torch.finfo.min. Summing those
        # values can overflow to -Inf, and the common ``sum() * 0`` zero-loss
        # idiom then becomes NaN. Keep the graph-connected anchor restricted to
        # valid candidates.
        zero = score_history.masked_fill(~expanded_valid, 0.0).sum() * 0.0
        valid_rounds = round_mask.bool() & labels.any(dim=-1).unsqueeze(-1)
        if not valid_rounds.any():
            return {
                "online_trajectory_survival_loss": zero,
                "online_trajectory_prune_loss": zero,
                "online_trajectory_ranking_loss": zero,
                "online_trajectory_coverage_loss": zero,
                "online_trajectory_candidate_set_loss": zero,
                "online_trajectory_raw_candidate_set_loss": zero,
                "online_trajectory_initial_survival_loss": zero,
                "online_trajectory_transition_loss": zero,
                "online_trajectory_rank_improvement_loss": zero,
                "online_trajectory_dpo_loss": zero,
                "online_trajectory_kl_loss": zero,
                "online_trajectory_loss": zero,
            }

        expanded_labels = labels.unsqueeze(2).expand_as(alive_history)
        target_alive = 1.0 - torch.prod(
            1.0 - alive_history.clamp(0.0, 1.0) * expanded_labels.float(), dim=-1
        )
        survival_loss = -torch.log(target_alive.clamp_min(self.eps))[valid_rounds].mean()
        initial_rows = valid_rounds[:, :, 0]
        if initial_rows.any():
            initial_survival_loss = -torch.log(
                target_alive[:, :, 0].clamp_min(self.eps)
            )[initial_rows].mean()
        else:
            initial_survival_loss = zero

        final_alive, last_round = self._last_round(alive_history, round_mask)
        trajectory_rows = round_mask.any(dim=-1) & labels.any(dim=-1)
        distractor_mask = valid_candidates & ~labels
        distractor_count = distractor_mask.sum(dim=-1).clamp_min(1)
        if self.independent_rounds:
            per_round_prune = (
                alive_history * distractor_mask.unsqueeze(2).float()
            ).sum(dim=-1) / distractor_count.unsqueeze(-1)
            prune_loss = per_round_prune[valid_rounds].mean()
        else:
            per_row_prune = (
                final_alive * distractor_mask.float()
            ).sum(dim=-1) / distractor_count
            prune_loss = per_row_prune[trajectory_rows].mean()

        min_value = torch.finfo(score_history.dtype).min
        positive_scores = score_history.masked_fill(~expanded_labels, min_value)
        negative_scores = score_history.masked_fill(~(expanded_valid & ~expanded_labels), min_value)
        positive_lse = torch.logsumexp(positive_scores, dim=-1)
        negative_lse = torch.logsumexp(negative_scores, dim=-1)
        has_negative = (expanded_valid & ~expanded_labels).any(dim=-1)
        final_round_mask = torch.zeros_like(round_mask, dtype=torch.bool)
        final_round_mask.scatter_(
            2,
            last_round.unsqueeze(-1),
            round_mask.any(dim=-1, keepdim=True),
        )
        # An ambiguous early query should preserve all plausible candidates,
        # not be forced to identify the target before clarification arrives.
        ranking_rows = valid_rounds & has_negative
        if not self.independent_rounds:
            ranking_rows &= final_round_mask
        if ranking_rows.any():
            ranking_loss = F.softplus(
                self.ranking_margin - positive_lse[ranking_rows] + negative_lse[ranking_rows]
            ).mean()
        else:
            ranking_loss = zero

        decision_logit = torch.logit(
            score_history.new_tensor(self.decision_threshold)
        )
        compatible = labels if compatible_labels is None else compatible_labels.bool()
        initial_scores = score_history[:, :, 0]
        initial_decision_scores = initial_scores - decision_logit
        compatible_rows = round_mask[:, :, 0] & compatible.any(dim=-1)
        if compatible_rows.any():
            positive_mask = compatible & valid_candidates
            negative_mask = (~compatible) & valid_candidates
            positive_loss = (
                F.softplus(-initial_decision_scores) * positive_mask.float()
            ).sum(dim=-1) / positive_mask.sum(dim=-1).clamp_min(1)
            negative_loss = (
                F.softplus(initial_decision_scores) * negative_mask.float()
            ).sum(dim=-1) / negative_mask.sum(dim=-1).clamp_min(1)
            coverage_loss = (
                self.coverage_positive_weight * positive_loss
                + self.coverage_negative_weight * negative_loss
            )[compatible_rows].mean()
        else:
            coverage_loss = zero

        candidate_set_loss = zero
        raw_candidate_set_loss = zero
        if candidate_labels is not None:
            candidate_targets = candidate_labels.bool()
            if candidate_targets.shape != score_history.shape:
                raise ValueError(
                    "candidate_labels must have the same [B,L,T,C] shape as score_history."
                )
            candidate_targets = candidate_targets & expanded_valid
            candidate_rows = valid_rounds & candidate_targets.any(dim=-1)
            if candidate_rows.any():
                candidate_negatives = expanded_valid & ~candidate_targets
                decision_scores = score_history - decision_logit
                set_positive = (
                    F.softplus(-decision_scores) * candidate_targets.float()
                ).sum(dim=-1) / candidate_targets.sum(dim=-1).clamp_min(1)
                set_negative = (
                    F.softplus(decision_scores) * candidate_negatives.float()
                ).sum(dim=-1) / candidate_negatives.sum(dim=-1).clamp_min(1)
                candidate_set_loss = (
                    self.candidate_set_positive_weight * set_positive
                    + self.candidate_set_negative_weight * set_negative
                )[candidate_rows].mean()
                if raw_score_history is not None:
                    raw_decision_scores = raw_score_history - decision_logit
                    raw_set_positive = (
                        F.softplus(-raw_decision_scores) * candidate_targets.float()
                    ).sum(dim=-1) / candidate_targets.sum(dim=-1).clamp_min(1)
                    raw_set_negative = (
                        F.softplus(raw_decision_scores) * candidate_negatives.float()
                    ).sum(dim=-1) / candidate_negatives.sum(dim=-1).clamp_min(1)
                    raw_candidate_set_loss = (
                        self.candidate_set_positive_weight * raw_set_positive
                        + self.candidate_set_negative_weight * raw_set_negative
                    )[candidate_rows].mean()

        final_scores, _ = self._last_round(score_history, round_mask)
        initial_valid = valid_candidates
        final_positive = self._target_logprob(final_scores, labels, initial_valid)
        initial_positive = self._target_logprob(initial_scores, labels, initial_valid)
        # Prevent apparent improvement obtained by degrading the earlier score.
        # The final state may still backpropagate through recurrent early states.
        if self.detach_temporal_baseline:
            initial_positive = initial_positive.detach()
        improvement_rows = trajectory_rows & (round_mask.sum(dim=-1) > 1)
        if self.independent_rounds:
            improvement_rows = torch.zeros_like(improvement_rows)
        if improvement_rows.any():
            rank_improvement_loss = F.softplus(
                self.ranking_margin + initial_positive[improvement_rows]
                - final_positive[improvement_rows]
            ).mean()
        else:
            rank_improvement_loss = zero

        transition_loss = zero
        if (
            not self.independent_rounds
            and feedback is not None
            and score_history.shape[2] > 1
        ):
            positive_evidence = self._target_logprob(
                score_history, expanded_labels, expanded_valid
            )
            previous_evidence = positive_evidence[:, :, :-1]
            if self.detach_temporal_baseline:
                previous_evidence = previous_evidence.detach()
            evidence_delta = positive_evidence[:, :, 1:] - previous_evidence
            transition_mask = round_mask[:, :, 1:] & round_mask[:, :, :-1]
            signed_feedback = feedback[:, :, 1:].float().clamp(-1.0, 1.0)
            supervised = transition_mask & (signed_feedback != 0)
            if supervised.any():
                transition_loss = F.softplus(
                    self.ranking_margin - signed_feedback[supervised] * evidence_delta[supervised]
                ).mean()

        dpo_loss = zero
        if negative_score_history is not None and not self.independent_rounds:
            positive_step_logp = self._target_logprob(
                score_history, expanded_labels, expanded_valid
            )
            negative_step_logp = self._target_logprob(
                negative_score_history, expanded_labels, expanded_valid
            )
            positive_logp = (positive_step_logp * round_mask.float()).sum(dim=-1)
            negative_logp = (negative_step_logp * round_mask.float()).sum(dim=-1)
            dpo_rows = trajectory_rows & (round_mask.sum(dim=-1) > 1)
            if negative_round_mask is not None:
                dpo_rows &= (
                    (negative_round_mask.bool() & round_mask).sum(dim=-1)
                    == round_mask.sum(dim=-1)
                )
            if dpo_rows.any():
                dpo_loss = -F.logsigmoid(
                    self.dpo_beta * (positive_logp[dpo_rows] - negative_logp[dpo_rows])
                ).mean()

        kl_loss = zero
        if raw_score_history is not None:
            raw_reference = (
                raw_score_history.detach() if self.kl_detach_raw else raw_score_history
            )
            raw_prob = torch.sigmoid(raw_reference).clamp(self.eps, 1.0 - self.eps)
            refined_prob = torch.sigmoid(score_history).clamp(self.eps, 1.0 - self.eps)
            bernoulli_kl = (
                refined_prob * torch.log(refined_prob / raw_prob)
                + (1.0 - refined_prob)
                * torch.log((1.0 - refined_prob) / (1.0 - raw_prob))
            )
            kl_mask = expanded_valid & round_mask.unsqueeze(-1)
            if self.kl_initial_only:
                initial_round_mask = torch.zeros_like(round_mask, dtype=torch.bool)
                initial_round_mask[:, :, 0] = round_mask[:, :, 0]
                kl_mask &= initial_round_mask.unsqueeze(-1)
            if kl_mask.any():
                kl_loss = bernoulli_kl[kl_mask].mean()

        total = (
            self.survival_weight * survival_loss
            + self.prune_weight * prune_loss
            + self.ranking_weight * ranking_loss
            + self.coverage_weight * coverage_loss
            + self.candidate_set_weight * candidate_set_loss
            + self.raw_candidate_set_weight * raw_candidate_set_loss
            + self.initial_survival_weight * initial_survival_loss
            + self.transition_weight * transition_loss
            + self.rank_improvement_weight * rank_improvement_loss
            + self.dpo_weight * dpo_loss
            + self.kl_weight * kl_loss
        )
        return {
            "online_trajectory_survival_loss": survival_loss,
            "online_trajectory_prune_loss": prune_loss,
            "online_trajectory_ranking_loss": ranking_loss,
            "online_trajectory_coverage_loss": coverage_loss,
            "online_trajectory_candidate_set_loss": candidate_set_loss,
            "online_trajectory_raw_candidate_set_loss": raw_candidate_set_loss,
            "online_trajectory_initial_survival_loss": initial_survival_loss,
            "online_trajectory_transition_loss": transition_loss,
            "online_trajectory_rank_improvement_loss": rank_improvement_loss,
            "online_trajectory_dpo_loss": dpo_loss,
            "online_trajectory_kl_loss": kl_loss,
            "online_trajectory_loss": total,
        }
