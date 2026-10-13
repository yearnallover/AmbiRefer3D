from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def set_constraint_ids_to_batch(batch, cid, constraint_table, vocab, max_len, device):
    B = cid.shape[0]
    c_ctype = torch.zeros((B,), dtype=torch.long, device=device)
    c_rel = torch.zeros((B,), dtype=torch.long, device=device)
    c_attr = torch.zeros((B,), dtype=torch.long, device=device)
    field_ids = torch.zeros((B, max_len), dtype=torch.long, device=device)

    for i in range(B):
        row = constraint_table.row(int(cid[i].item()))
        c_ctype[i] = row.ctype_id
        c_rel[i] = row.rel_id
        c_attr[i] = row.attr_id
        field_ids[i] = torch.tensor(vocab.encode(row.field_text, max_len), dtype=torch.long, device=device)

    batch["c_ctype_id"] = c_ctype
    batch["c_rel_id"] = c_rel
    batch["c_attr_id"] = c_attr
    batch["field_ids"] = field_ids


def _clone_batch(batch: Dict):
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def build_pseudo_target_from_prev_mask(
    cand_feat: torch.Tensor,
    alive_mask: torch.Tensor,
    strategy: str = "top1",
    prev_mask_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    B, C, D = cand_feat.shape
    device = cand_feat.device

    if strategy == "mean":
        w = alive_mask.unsqueeze(-1).float()
        denom = w.sum(dim=1).clamp_min(1.0)
        return (cand_feat * w).sum(dim=1) / denom

    if strategy == "random":
        idxs = []
        for i in range(B):
            valid = torch.nonzero(alive_mask[i] > 0, as_tuple=False).squeeze(1)
            if valid.numel() == 0:
                valid = torch.tensor([0], device=device)
            idxs.append(valid[torch.randint(0, valid.numel(), (1,), device=device)].item())
        idxs = torch.tensor(idxs, device=device, dtype=torch.long)
        return cand_feat.gather(1, idxs.view(-1, 1, 1).expand(-1, 1, D)).squeeze(1)

    if strategy == "top1":
        if prev_mask_logits is None:
            raise ValueError("top1 target strategy requires prev_mask_logits.")
        scores = prev_mask_logits.masked_fill(alive_mask <= 0, -1e9)
        pseudo_idx = scores.argmax(dim=1)
        return cand_feat.gather(1, pseudo_idx.view(-1, 1, 1).expand(-1, 1, D)).squeeze(1)

    raise ValueError(f"Unknown target summary strategy: {strategy}")


def choose_constraint_random(used: torch.Tensor, active: torch.Tensor | None = None):
    device = used.device
    B, _ = used.shape
    cid = torch.zeros((B,), dtype=torch.long, device=device)

    for i in range(B):
        if active is not None and not bool(active[i].item()):
            continue
        valid_cids = torch.nonzero(used[i] == 0, as_tuple=False).squeeze(1)
        if valid_cids.numel() > 0:
            cid[i] = valid_cids[torch.randint(0, valid_cids.numel(), (1,), device=device)]
    return cid


def choose_constraint_heuristic(model, batch, constraint_table, vocab, max_len, used, alive, thr, active=None):
    device = alive.device
    B, _ = alive.shape
    Nc = constraint_table.num_constraints()

    best_cid = torch.zeros((B,), dtype=torch.long, device=device)
    best_keep = torch.full((B,), 10**9, dtype=torch.long, device=device)

    for cid_try in range(Nc):
        cid_tensor = torch.full((B,), cid_try, dtype=torch.long, device=device)
        tmp = _clone_batch(batch)
        set_constraint_ids_to_batch(tmp, cid_tensor, constraint_table, vocab, max_len, device)
        tmp["alive_mask"] = alive
        tmp["used_constraint_mask"] = used

        out = model(tmp)
        keep_cnt = (alive * (torch.sigmoid(out["mask_logits"]) >= thr).float()).sum(dim=1).long()
        valid = (used[:, cid_try] == 0) & (keep_cnt > 0)
        if active is not None:
            valid = valid & active
        better = valid & (keep_cnt < best_keep)
        best_keep = torch.where(better, keep_cnt, best_keep)
        best_cid = torch.where(better, torch.full_like(best_cid, cid_try), best_cid)

    return best_cid


def apply_constraints_all_at_once(model, batch, constraint_ids_topk, constraint_table, vocab, max_len, alive):
    device = alive.device
    B, C = alive.shape
    joint_logits = torch.zeros((B, C), device=device)

    for k in range(constraint_ids_topk.shape[1]):
        tmp = _clone_batch(batch)
        set_constraint_ids_to_batch(tmp, constraint_ids_topk[:, k], constraint_table, vocab, max_len, device)
        tmp["alive_mask"] = alive
        out = model(tmp)
        joint_logits += out["mask_logits"]

    return joint_logits


def _active_top1(mask_logits: torch.Tensor, alive: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    masked_logits = mask_logits.masked_fill(alive <= 0, -1e9)
    return masked_logits.max(dim=-1)


def _history_at(hist: List[torch.Tensor], k: int, default: torch.Tensor) -> torch.Tensor:
    if not hist:
        return default
    return hist[min(k, len(hist)) - 1]


def risk_sensitive_candidate_update(
    probability: torch.Tensor,
    alive: torch.Tensor,
    threshold: float,
    selector_confidence: torch.Tensor | None = None,
    min_keep_ratio: float = 0.5,
    min_keep_count: int = 1,
    terminal_ready: torch.Tensor | None = None,
) -> torch.Tensor:
    """Prune conservatively when the selected clarification is uncertain."""
    batch_size, num_candidates = alive.shape
    if selector_confidence is None:
        selector_confidence = probability.new_ones(batch_size)
    confidence = selector_confidence.float().clamp(0.0, 1.0)
    adaptive_threshold = float(threshold) * (0.5 + 0.5 * confidence)
    keep = (probability >= adaptive_threshold.unsqueeze(1)) & (alive > 0)

    alive_count = (alive > 0).sum(dim=1)
    required = torch.ceil(alive_count.float() * float(min_keep_ratio)).long()
    required = required.clamp_min(int(min_keep_count))
    required = torch.minimum(required, alive_count)
    if terminal_ready is not None:
        terminal_minimum = torch.full_like(required, int(min_keep_count))
        terminal_minimum = torch.minimum(terminal_minimum, alive_count)
        required = torch.where(
            terminal_ready.bool(), terminal_minimum, required
        )
    masked_probability = probability.masked_fill(alive <= 0, -1.0)
    for row in range(batch_size):
        count = int(required[row].item())
        if count <= 0 or int(keep[row].sum().item()) >= count:
            continue
        top_indices = torch.topk(
            masked_probability[row], k=min(count, num_candidates)
        ).indices
        keep[row, top_indices] = True
    return alive * keep.float()


@torch.no_grad()
def eval_rollout_one(
    model,
    batch: Dict[str, torch.Tensor],
    constraint_table,
    vocab,
    max_len: int,
    max_round: int = 4,
    thr: float = 0.4,
    stop_thr: float = 0.5,
):
    pred_idx, pred_logit, rounds, alive_hist, _ = eval_rollout_batch(
        model=model,
        batch=batch,
        constraint_table=constraint_table,
        vocab=vocab,
        max_len=max_len,
        max_round=max_round,
        thr=thr,
        stop_thr=stop_thr,
        eval_mode="agent",
        v_target_strategy="top1",
        selector_strategy="ours",
    )
    return int(pred_idx[0]), float(pred_logit[0]), int(rounds[0]), [bool(x[0]) for x in alive_hist]


@torch.no_grad()
def eval_rollout_batch(
    model,
    batch: dict,
    constraint_table,
    vocab,
    max_len: int,
    max_round: int = 4,
    thr: float = 0.4,
    stop_thr: float = 0.5,
    eval_mode: str = "oracle",
    v_target_strategy: str = "top1",
    selector_strategy: str = "ours",
    risk_sensitive_pruning: bool = True,
    min_keep_ratio: float = 0.5,
    min_keep_count: int = 1,
    return_trace: bool = False,
):
    device = next(model.parameters()).device
    model.eval()

    for k in batch:
        batch[k] = batch[k].to(device)

    B, C = batch["cand_mask"].shape
    Nc = constraint_table.num_constraints()
    alive = batch.get("alive_mask", batch["cand_mask"]).clone()
    alive = alive * batch["cand_mask"]
    available = batch.get("available_constraint_mask")
    used = (
        (~available.bool()).float()
        if available is not None
        else torch.zeros((B, Nc), device=device)
    )
    state_constraint_ids = batch.get("state_constraint_ids")
    if state_constraint_ids is not None and state_constraint_ids.ndim == 2:
        scheduled_rows = (state_constraint_ids >= 0).any(dim=1)
        scheduled_lengths = (state_constraint_ids >= 0).sum(dim=1).long()
    else:
        state_constraint_ids = None
        scheduled_rows = torch.zeros((B,), dtype=torch.bool, device=device)
        scheduled_lengths = torch.zeros((B,), dtype=torch.long, device=device)
    minimum_rounds = batch.get(
        "minimum_rollout_rounds",
        torch.zeros((B,), dtype=torch.long, device=device),
    ).long().clamp_min(0)
    gt = batch["target_index"].long()
    valid_gt = (gt >= 0) & (gt < C)
    safe_gt = gt.clamp(0, max(C - 1, 0))
    target_candidate_mask = batch.get("target_candidate_mask")
    if target_candidate_mask is not None:
        target_candidate_mask = target_candidate_mask.bool() & (alive > 0)
        valid_gt = target_candidate_mask.any(dim=1)

    def target_is_alive(current_alive: torch.Tensor) -> torch.Tensor:
        if target_candidate_mask is not None:
            return valid_gt & (target_candidate_mask & (current_alive > 0)).any(dim=1)
        return valid_gt & (
            current_alive.gather(1, safe_gt.view(B, 1)).squeeze(1) > 0.5
        )

    finished = torch.zeros((B,), dtype=torch.bool, device=device)
    rounds = torch.full((B,), max_round, dtype=torch.long, device=device)
    final_logit = torch.full((B,), -1e9, device=device)
    final_idx = torch.zeros((B,), dtype=torch.long, device=device)
    initial_belief = batch.get("cand_belief")
    if initial_belief is not None:
        initial_belief = initial_belief.float() * alive.float()
        initial_belief = initial_belief / initial_belief.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        cumulative_log_belief = torch.log(initial_belief.clamp_min(1e-12))
        cumulative_log_belief = cumulative_log_belief.masked_fill(
            alive <= 0, -1e9
        )
    else:
        prior_logits = batch.get("cand_stage1_scores")
        if prior_logits is None:
            prior_logits = torch.zeros_like(alive)
        prior_logits = prior_logits.float().masked_fill(alive <= 0, -1e9)
        cumulative_log_belief = torch.log_softmax(prior_logits, dim=1)
    dynamic_alive_probability = batch.get("cand_alive_probs")
    if dynamic_alive_probability is None:
        dynamic_alive_probability = alive.float()
    else:
        dynamic_alive_probability = (
            dynamic_alive_probability.float().clamp(0.0, 1.0) * alive.float()
        )
    alive_hist: List[torch.Tensor] = []
    alive_cnt_hist: List[torch.Tensor] = []
    pred_idx_hist: List[torch.Tensor] = []
    finished_hist: List[torch.Tensor] = []
    alive_mask_hist: List[torch.Tensor] = []
    belief_hist: List[torch.Tensor] = []
    selected_constraint_hist: List[torch.Tensor] = []
    selector_confidence_hist: List[torch.Tensor] = []
    stop_probability_hist: List[torch.Tensor] = []
    mask_probability_hist: List[torch.Tensor] = []
    initial_alive = alive.detach().cpu().clone()

    if selector_strategy == "all_at_once":
        batch["alive_mask"] = alive
        batch["used_constraint_mask"] = used
        masked_log_belief = cumulative_log_belief.masked_fill(
            alive <= 0, -1e9
        )
        current_belief = torch.softmax(masked_log_belief, dim=1) * alive.float()
        current_belief = current_belief / current_belief.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        batch["cand_belief"] = current_belief
        batch["cand_alive_probs"] = dynamic_alive_probability * alive.float()
        batch["target_summary"] = None
        out_sel = model(batch)
        topk = torch.topk(out_sel["selector_logits"], k=min(3, Nc), dim=-1).indices
        mask_logits = apply_constraints_all_at_once(model, batch, topk, constraint_table, vocab, max_len, alive)
        topv, topi = _active_top1(mask_logits, alive)
        prob = torch.sigmoid(mask_logits)
        alive = alive * (prob >= thr).float()
        target_alive = target_is_alive(alive)
        result = (
            topi.detach().cpu(),
            topv.detach().cpu(),
            torch.ones_like(rounds).detach().cpu(),
            [target_alive.detach().cpu()],
            [alive.sum(dim=1).long().detach().cpu()],
        )
        if return_trace:
            return result + ({
                "pred_idx_history": [topi.detach().cpu()],
                "finished_history": [
                    torch.ones_like(rounds, dtype=torch.bool).detach().cpu()
                ],
                "initial_alive_mask": initial_alive,
                "alive_mask_history": [alive.detach().cpu().clone()],
                "belief_history": [
                    torch.softmax(mask_logits.masked_fill(alive <= 0, -1e9), dim=1)
                    .detach().cpu()
                ],
                "selected_constraint_history": [topk[:, 0].detach().cpu()],
                "selector_confidence_history": [
                    torch.ones_like(rounds, dtype=torch.float32).detach().cpu()
                ],
                "stop_probability_history": [
                    torch.ones_like(rounds, dtype=torch.float32).detach().cpu()
                ],
                "mask_probability_history": [prob.detach().cpu()],
            },)
        return result

    oracle_cid = batch.get("constraint_id", None)
    for r in range(max_round):
        active = ~finished
        if not bool(active.any().item()):
            break

        schedule_active = scheduled_rows.clone()
        if state_constraint_ids is None or r >= state_constraint_ids.shape[1]:
            schedule_active.zero_()
            scheduled_cid = torch.zeros((B,), dtype=torch.long, device=device)
        else:
            scheduled_cid = state_constraint_ids[:, r].long().clamp_min(0)
            schedule_active &= state_constraint_ids[:, r] >= 0
        step_used = used
        if schedule_active.any():
            step_used = used.clone()
            step_used[schedule_active] = 1.0
            step_used[schedule_active] = step_used[schedule_active].scatter(
                1, scheduled_cid[schedule_active].view(-1, 1), 0.0
            )

        batch["alive_mask"] = alive
        batch["used_constraint_mask"] = step_used
        masked_log_belief = cumulative_log_belief.masked_fill(
            alive <= 0, -1e9
        )
        current_belief = torch.softmax(masked_log_belief, dim=1) * alive.float()
        current_belief = current_belief / current_belief.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        batch["cand_belief"] = current_belief
        batch["cand_alive_probs"] = dynamic_alive_probability * alive.float()
        if r == 0 or v_target_strategy == "candidate_belief":
            # Stage2Net reconstructs this summary from cand_belief after masking
            # it by the current alive set. This is the same state used by the
            # teacher-forced trajectory rows during training.
            batch["target_summary"] = None
        else:
            tmp_out = model(batch)
            batch["target_summary"] = build_pseudo_target_from_prev_mask(
                cand_feat=tmp_out.get("selector_cand_feat", tmp_out["cand_feat"]),
                prev_mask_logits=cumulative_log_belief,
                alive_mask=alive,
                strategy=v_target_strategy,
            )

        selector_confidence = torch.ones(B, device=device)
        if eval_mode == "oracle" and r == 0 and oracle_cid is not None:
            cid = oracle_cid.long()
        elif selector_strategy == "ours":
            selector_logits = model(batch)["selector_logits"]
            selector_probability = torch.softmax(selector_logits, dim=-1)
            selector_confidence, cid = selector_probability.max(dim=-1)
        elif selector_strategy == "random":
            cid = choose_constraint_random(used, active=active)
        elif selector_strategy == "heuristic":
            cid = choose_constraint_heuristic(model, batch, constraint_table, vocab, max_len, used, alive, thr, active)
        else:
            raise ValueError(f"Unknown selector_strategy: {selector_strategy}")

        if schedule_active.any():
            cid = torch.where(schedule_active, scheduled_cid, cid)
            if selector_strategy == "ours":
                selector_confidence = torch.where(
                    schedule_active,
                    selector_probability.gather(1, cid.view(-1, 1)).squeeze(1),
                    selector_confidence,
                )

        used[active] = used[active].scatter(1, cid[active].view(-1, 1), 1.0)
        if not (eval_mode == "oracle" and r == 0 and oracle_cid is not None):
            set_constraint_ids_to_batch(batch, cid, constraint_table, vocab, max_len, device)

        out = model(batch)
        mask_logits = out["mask_logits"]
        stop_logit = out["stop_logit"]
        stop_probability = torch.sigmoid(stop_logit)
        stop_now = stop_probability >= stop_thr
        step_log_evidence = torch.nn.functional.logsigmoid(mask_logits)
        updated_log_belief = cumulative_log_belief + step_log_evidence
        cumulative_log_belief = torch.where(
            active.unsqueeze(1), updated_log_belief, cumulative_log_belief
        )

        probability = torch.sigmoid(mask_logits)
        proposed_alive_probability = dynamic_alive_probability * probability
        dynamic_alive_probability = torch.where(
            active.unsqueeze(1),
            proposed_alive_probability,
            dynamic_alive_probability,
        )
        if risk_sensitive_pruning:
            next_alive = risk_sensitive_candidate_update(
                probability,
                alive,
                threshold=thr,
                selector_confidence=selector_confidence,
                min_keep_ratio=min_keep_ratio,
                min_keep_count=min_keep_count,
                terminal_ready=stop_now & ((r + 1) >= minimum_rounds),
            )
        else:
            next_alive = alive * (probability >= thr).float()
        alive = torch.where(active.view(B, 1), next_alive, alive)
        dynamic_alive_probability = dynamic_alive_probability * alive.float()

        # Compose evidence from every accepted clarification. This matches the
        # trajectory log-probability objective and avoids forgetting earlier
        # constraints whenever more than one candidate remains alive.
        topv, topi = _active_top1(cumulative_log_belief, alive)
        final_logit = torch.where(active, topv, final_logit)
        final_idx = torch.where(active, topi, final_idx)

        target_alive = target_is_alive(alive)
        alive_cnt = alive.sum(dim=1).long()
        alive_hist.append(target_alive.detach().cpu())
        alive_cnt_hist.append(alive_cnt.detach().cpu())
        if return_trace:
            active_constraint = torch.where(
                active, cid, torch.full_like(cid, -1)
            )
            active_confidence = torch.where(
                active,
                selector_confidence,
                torch.full_like(selector_confidence, float("nan")),
            )
            active_stop_probability = torch.where(
                active,
                stop_probability,
                torch.full_like(stop_probability, float("nan")),
            )
            round_log_belief = cumulative_log_belief.masked_fill(
                alive <= 0, -1e9
            )
            round_belief = torch.softmax(round_log_belief, dim=1) * alive.float()
            round_belief = round_belief / round_belief.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-6)
            alive_mask_hist.append(alive.detach().cpu().clone())
            belief_hist.append(round_belief.detach().cpu())
            selected_constraint_hist.append(active_constraint.detach().cpu())
            selector_confidence_hist.append(active_confidence.detach().cpu())
            stop_probability_hist.append(active_stop_probability.detach().cpu())
            mask_probability_hist.append(probability.detach().cpu())

        one_left = alive_cnt <= 1
        legacy_no_actions_left = (used >= 0.5).all(dim=1)
        scheduled_no_actions_left = (r + 1) >= scheduled_lengths
        no_actions_left = torch.where(
            scheduled_rows, scheduled_no_actions_left, legacy_no_actions_left
        )
        minimum_reached = (r + 1) >= minimum_rounds
        finish_now = active & minimum_reached & (stop_now | one_left | no_actions_left)
        rounds = torch.where(finish_now, torch.full_like(rounds, r + 1), rounds)
        finished = finished | finish_now
        pred_idx_hist.append(final_idx.detach().cpu().clone())
        finished_hist.append(finished.detach().cpu().clone())

    result = (
        final_idx.detach().cpu(),
        final_logit.detach().cpu(),
        rounds.detach().cpu(),
        alive_hist,
        alive_cnt_hist,
    )
    if return_trace:
        return result + ({
            "pred_idx_history": pred_idx_hist,
            "finished_history": finished_hist,
            "initial_alive_mask": initial_alive,
            "alive_mask_history": alive_mask_hist,
            "belief_history": belief_hist,
            "selected_constraint_history": selected_constraint_hist,
            "selector_confidence_history": selector_confidence_hist,
            "stop_probability_history": stop_probability_hist,
            "mask_probability_history": mask_probability_hist,
        },)
    return result
