from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn


class TrajectoryConditionedGroundingAdapter(nn.Module):
    """Refine fixed Stage1 proposals using the current interaction history.

    The scene proposals stay fixed. Only query-dependent proposal residuals,
    grounding logits, belief, and the recurrent trajectory state are updated.
    """

    def __init__(
        self,
        proposal_dim: int,
        query_dim: int,
        hidden_dim: int = 128,
        relation_dim: int = 64,
        dropout: float = 0.1,
        initial_residual_scale: float = 0.1,
        max_residual_scale: float = 1.0,
        max_logit_residual: float = 4.0,
        max_feature_residual: float = 2.0,
        initial_prune_gate: float = 0.05,
        risk_gate_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.proposal_dim = int(proposal_dim)
        self.query_dim = int(query_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_residual_scale = float(max_residual_scale)
        self.max_logit_residual = float(max_logit_residual)
        self.max_feature_residual = float(max_feature_residual)
        self.initial_prune_gate = float(initial_prune_gate)
        self.risk_gate_enabled = bool(risk_gate_enabled)
        if not 0.0 < self.max_residual_scale <= 1.0:
            raise ValueError("max_residual_scale must be in (0, 1].")
        if not 0.0 <= self.initial_prune_gate <= 1.0:
            raise ValueError("initial_prune_gate must be in [0, 1].")

        self.proposal_proj = nn.Sequential(
            nn.LayerNorm(self.proposal_dim),
            nn.Linear(self.proposal_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.query_proj = nn.Sequential(
            nn.LayerNorm(self.query_dim),
            nn.Linear(self.query_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.relation_mlp = nn.Sequential(
            nn.Linear(10, relation_dim),
            nn.GELU(),
            nn.Linear(relation_dim, self.hidden_dim),
        )
        self.feedback_proj = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.GELU(),
        )
        self.prune_gate = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.prune_gate.requires_grad_(self.risk_gate_enabled)
        self.memory = nn.GRUCell(4 * self.hidden_dim, self.hidden_dim)

        residual_input_dim = 3 * self.hidden_dim
        self.feature_residual = nn.Sequential(
            nn.Linear(residual_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.proposal_dim),
        )
        self.logit_residual = nn.Sequential(
            nn.Linear(residual_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.summary_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        initial_scale = min(
            max(float(initial_residual_scale), 1e-6),
            self.max_residual_scale - 1e-6,
        )
        scale = initial_scale / self.max_residual_scale
        self.residual_scale_logit = nn.Parameter(
            torch.tensor(math.log(scale / (1.0 - scale)), dtype=torch.float32)
        )
        self._init_residual_heads()

    def _init_residual_heads(self) -> None:
        for head in (self.feature_residual, self.logit_residual):
            output = head[-1]
            nn.init.xavier_uniform_(output.weight, gain=0.01)
            nn.init.zeros_(output.bias)

    @staticmethod
    def _relation_features(bboxes: torch.Tensor) -> torch.Tensor:
        centers = bboxes.mean(dim=2)
        sizes = (bboxes[:, :, 1] - bboxes[:, :, 0]).abs().clamp(1e-4, 1e4)
        delta = centers.unsqueeze(1) - centers.unsqueeze(2)
        distance = delta.norm(dim=-1, keepdim=True).clamp_min(1e-4)
        direction = delta / distance
        log_sizes = torch.log(sizes)
        size_ratio = log_sizes.unsqueeze(1) - log_sizes.unsqueeze(2)
        return torch.cat((delta, distance, direction, size_ratio), dim=-1)

    @staticmethod
    def _masked_belief(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(logits) * mask.float()
        return probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def forward(
        self,
        proposal_features: torch.Tensor,
        bboxes: torch.Tensor,
        raw_logits: torch.Tensor,
        query_features: torch.Tensor,
        proposal_mask: torch.Tensor,
        alive_probs: Optional[torch.Tensor] = None,
        previous_state: Optional[torch.Tensor] = None,
        previous_belief: Optional[torch.Tensor] = None,
        feedback: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run one trajectory step.

        Shapes are ``proposal_features [N,C,Dp]``, ``bboxes [N,C,2,3]``,
        ``raw_logits/alive_probs [N,C]``, ``query_features [N,Dq]``, and
        ``previous_state [N,H]``.
        """
        if proposal_features.dim() != 3:
            raise ValueError("proposal_features must have shape [N,C,Dp].")
        if bboxes.shape[:2] != proposal_features.shape[:2] or bboxes.shape[-2:] != (2, 3):
            raise ValueError("bboxes must have shape [N,C,2,3].")
        if raw_logits.shape != proposal_features.shape[:2]:
            raise ValueError("raw_logits must have shape [N,C].")
        if query_features.shape != (proposal_features.shape[0], self.query_dim):
            raise ValueError(
                f"query_features must have shape [N,{self.query_dim}], got {tuple(query_features.shape)}."
            )

        mask = proposal_mask.bool()
        alive = mask.float() if alive_probs is None else alive_probs.float() * mask.float()
        current_belief = self._masked_belief(raw_logits, alive)
        if previous_belief is None:
            previous_belief = current_belief
        if previous_state is None:
            previous_state = proposal_features.new_zeros(
                (proposal_features.shape[0], self.hidden_dim)
            )
        if feedback is None:
            feedback = raw_logits.new_zeros(raw_logits.shape[0])

        candidate_hidden = self.proposal_proj(proposal_features)
        query_hidden = self.query_proj(query_features)
        candidate_summary = torch.sum(
            candidate_hidden * current_belief.unsqueeze(-1), dim=1
        )

        relation_hidden = self.relation_mlp(self._relation_features(bboxes))
        pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        neighbor_weight = current_belief.unsqueeze(1).expand_as(pair_mask.float())
        neighbor_weight = neighbor_weight * pair_mask.float()
        neighbor_weight = neighbor_weight / neighbor_weight.sum(
            dim=2, keepdim=True
        ).clamp_min(1e-6)
        relation_context = torch.sum(
            relation_hidden * neighbor_weight.unsqueeze(-1), dim=2
        )

        entropy = -torch.sum(
            current_belief * torch.log(current_belief.clamp_min(1e-6)), dim=-1
        )
        belief_shift = torch.abs(current_belief - previous_belief).sum(dim=-1)
        alive_fraction = alive.sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
        feedback_stats = torch.stack(
            (feedback.float(), entropy, belief_shift, alive_fraction), dim=-1
        )
        feedback_hidden = self.feedback_proj(feedback_stats)
        memory_input = torch.cat(
            (query_hidden, candidate_summary, feedback_hidden, previous_state), dim=-1
        )
        state = self.memory(memory_input, previous_state)

        state_per_candidate = state.unsqueeze(1).expand(-1, proposal_features.shape[1], -1)
        residual_input = torch.cat(
            (candidate_hidden, relation_context, state_per_candidate), dim=-1
        )
        scale = self.max_residual_scale * torch.sigmoid(self.residual_scale_logit)
        feature_delta = (
            self.max_feature_residual
            * torch.tanh(self.feature_residual(residual_input))
            * mask.unsqueeze(-1)
        )
        raw_logit_delta = (
            self.max_logit_residual
            * torch.tanh(self.logit_residual(residual_input).squeeze(-1))
            * mask.float()
        )
        if self.risk_gate_enabled:
            clarification_evidence = feedback.float().clamp(0.0, 1.0).unsqueeze(-1)
            learned_prune_gate = torch.sigmoid(self.prune_gate(feedback_stats))
            prune_gate = self.initial_prune_gate + (
                1.0 - self.initial_prune_gate
            ) * clarification_evidence * learned_prune_gate
            logit_delta = torch.relu(raw_logit_delta) + prune_gate * torch.minimum(
                raw_logit_delta, torch.zeros_like(raw_logit_delta)
            )
        else:
            prune_gate = raw_logits.new_ones((raw_logits.shape[0], 1))
            logit_delta = raw_logit_delta
        refined_features = proposal_features + scale * feature_delta
        refined_logits = raw_logits + scale * logit_delta
        refined_belief = self._masked_belief(refined_logits, alive)
        target_summary = self.summary_proj(
            torch.sum(candidate_hidden * refined_belief.unsqueeze(-1), dim=1)
        )

        return {
            "proposal_features": refined_features,
            "feature_residual": feature_delta,
            "logits": refined_logits,
            "logit_residual": logit_delta,
            "prune_gate": prune_gate,
            "belief": refined_belief,
            "state": state,
            "target_summary": target_summary,
            "residual_scale": scale,
        }
