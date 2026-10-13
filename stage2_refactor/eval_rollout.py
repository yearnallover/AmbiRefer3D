"""Multi-round constraint selection, candidate filtering and learned stopping."""
import torch


def set_constraint_ids_to_batch(batch, cid, constraint_table, vocab, max_len, device):
    constraint_table.apply_id_to_batch(batch, cid, vocab, max_len)


def build_pseudo_target_from_prev_mask(cand_feat, alive_mask, strategy='top1', prev_mask_logits=None):
    if strategy == 'mean':
        weights = alive_mask.unsqueeze(-1)
        return (cand_feat * weights).sum(1) / weights.sum(1).clamp_min(1)
    if strategy == 'random':
        selected = torch.multinomial(alive_mask + 1e-8, 1).squeeze(1)
    elif strategy == 'top1':
        selected = prev_mask_logits.masked_fill(alive_mask <= 0, -1e9).argmax(1)
    else:
        raise ValueError(f'Unknown target summary strategy: {strategy}')
    return cand_feat[torch.arange(cand_feat.shape[0], device=cand_feat.device), selected]


@torch.no_grad()
def eval_rollout_batch(model, batch, constraint_table, vocab, max_len, max_round=4,
                       thr=0.4, stop_thr=0.5, eval_mode='agent', v_target_strategy='top1',
                       selector_strategy='ours'):
    device = next(model.parameters()).device
    batch = {key: value.to(device) for key, value in batch.items()}
    model.eval()
    alive = batch['cand_mask'].clone()
    batch_size, count = alive.shape
    used = torch.zeros(batch_size, constraint_table.num_constraints(), device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    rounds = torch.full((batch_size,), max_round, dtype=torch.long, device=device)
    indices = torch.full((batch_size,), -1, dtype=torch.long, device=device)
    logits = torch.full((batch_size,), -1e9, device=device)
    target = batch['target_index'].long()
    valid_target = (target >= 0) & (target < count)
    safe_target = target.clamp(0, count - 1)
    alive_history, counts_history = [], []
    previous_scores = None
    for round_index in range(max_round):
        active = ~finished
        if active.any():
            batch['alive_mask'] = alive
            batch['used_constraint_mask'] = used
            batch['target_summary'] = None
            if previous_scores is not None:
                encoded = model(batch)
                batch['target_summary'] = build_pseudo_target_from_prev_mask(
                    encoded['cand_feat'], alive, v_target_strategy, previous_scores)
            selector = model(batch)['selector_logits']
            if eval_mode == 'oracle' and round_index == 0:
                selected = batch['constraint_id']
            elif selector_strategy == 'random':
                selected = torch.multinomial((used == 0).float() + 1e-8, 1).squeeze(1)
            elif selector_strategy in ('ours', 'all_at_once'):
                selected = selector.argmax(1)
            elif selector_strategy == 'heuristic':
                retained = []
                for cid in range(used.shape[1]):
                    probe = dict(batch)
                    set_constraint_ids_to_batch(probe, torch.full_like(target, cid), constraint_table, vocab, max_len, device)
                    score = model(probe)['mask_logits']
                    keep = (alive * (torch.sigmoid(score) >= thr)).sum(1)
                    retained.append(keep.masked_fill((used[:, cid] > 0) | (keep == 0), float('inf')))
                selected = torch.stack(retained, dim=1).argmin(1)
            else:
                raise ValueError(f'Unknown selector strategy: {selector_strategy}')
            set_constraint_ids_to_batch(batch, selected, constraint_table, vocab, max_len, device)
            out = model(batch)
            scores = out['mask_logits']
            if selector_strategy == 'all_at_once':
                top_ids = selector.topk(min(3, used.shape[1]), dim=1).indices
                score_list = []
                for position in range(top_ids.shape[1]):
                    set_constraint_ids_to_batch(batch, top_ids[:, position], constraint_table, vocab, max_len, device)
                    score_list.append(model(batch)['mask_logits'])
                scores = torch.stack(score_list).mean(0)
            keep = alive * (torch.sigmoid(scores) >= thr).float()
            alive = torch.where(active.unsqueeze(1), keep, alive)
            used[active, selected[active]] = 1
            masked = scores.masked_fill(alive <= 0, -1e9)
            top_value, top_index = masked.max(1)
            has_candidate = alive.any(1)
            predicted = torch.where(has_candidate, top_index, -torch.ones_like(top_index))
            indices = torch.where(active, predicted, indices)
            logits = torch.where(active, top_value, logits)
            stopped = active & ((torch.sigmoid(out['stop_logit']) >= stop_thr)
                                | (alive.sum(1) <= 1) | (used > 0).all(1))
            if selector_strategy == 'all_at_once':
                stopped = active
            rounds = torch.where(stopped, torch.full_like(rounds, round_index + 1), rounds)
            finished |= stopped
            previous_scores = scores
        survived = valid_target & (alive.gather(1, safe_target.unsqueeze(1)).squeeze(1) > 0)
        alive_history.append(survived.cpu())
        counts_history.append(alive.sum(1).long().cpu())
    return indices.cpu(), logits.cpu(), rounds.cpu(), alive_history, counts_history


@torch.no_grad()
def eval_rollout_one(model, batch, constraint_table, vocab, max_len, max_round=4,
                     thr=0.4, stop_thr=0.5, **kwargs):
    indices, logits, rounds, survived, _ = eval_rollout_batch(
        model, batch, constraint_table, vocab, max_len, max_round, thr, stop_thr, **kwargs)
    return int(indices[0]), float(logits[0]), int(rounds[0]), [bool(row[0]) for row in survived]
