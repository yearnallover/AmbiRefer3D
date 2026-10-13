from __future__ import annotations
from dataclasses import asdict
import argparse
import csv
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from stage2_refactor.utils.io import load_json, save_json
from stage2_refactor.utils.misc import set_seed, get_device
from stage2_refactor.utils.split import split_samples_7_3, split_samples_0_10

from stage2_refactor.data.vocab import Vocab
from stage2_refactor.data.bbox_provider import PassthroughBBoxProvider
from stage2_refactor.data.dataset_semantic import RoundDatasetSemantic
from stage2_refactor.data.collate_semantic import collate_semantic

from stage2_refactor.model.stage2_net_semantic import Stage2NetSemantic
from stage2_refactor.learning.constraint_table import ConstraintTable
from stage2_refactor.learning.losses_semantic import candidate_ce_loss, selector_ce_loss, stop_bce_loss

from stage2_refactor.eval_rollout import eval_rollout_batch
from stage2_refactor.config import TrainConfig

def load_pretrained_flexible(model: torch.nn.Module, state: dict) -> None:
    """
    Load as much as possible when shapes mismatch:
      - for token embeddings: copy overlap rows
      - for selector.out: copy overlap rows
      - others: strict if same shape, else skip
    """
    msd = model.state_dict()
    loaded, skipped = [], []

    def copy_overlap(dst, src, name):
        # dst, src are tensors
        if dst.ndim == 2 and src.ndim == 2:  # [N, D]
            n = min(dst.shape[0], src.shape[0])
            d = min(dst.shape[1], src.shape[1])
            dst[:n, :d].copy_(src[:n, :d])
            loaded.append(f"{name} overlap [{n},{d}]")
        elif dst.ndim == 1 and src.ndim == 1:  # [N]
            n = min(dst.shape[0], src.shape[0])
            dst[:n].copy_(src[:n])
            loaded.append(f"{name} overlap [{n}]")
        else:
            skipped.append(f"{name} shape {tuple(src.shape)} -> {tuple(dst.shape)}")

    with torch.no_grad():
        for k, v in state.items():
            if k not in msd:
                skipped.append(f"{k} (unexpected)")
                continue

            dst = msd[k]
            src = v

            if dst.shape == src.shape:
                dst.copy_(src)
                loaded.append(k)
            else:
                # handle common mismatch keys
                if k in ["qenc.wemb.weight", "cenc.wemb.weight"]:
                    copy_overlap(dst, src, k)
                elif k in ["selector.out.weight", "selector.out.bias"]:
                    copy_overlap(dst, src, k)
                else:
                    skipped.append(f"{k} shape {tuple(src.shape)} -> {tuple(dst.shape)}")

    model.load_state_dict(msd, strict=False)
    print(f"[pretrained-flex] loaded={len(loaded)} skipped={len(skipped)}")
    if skipped:
        print("[pretrained-flex] skipped examples:", skipped[:20])

def build_vocab_from_train(train_samples, max_size=20000, min_freq=1):
    texts = []
    for s in train_samples:
        texts.append(s.get("query", "") or "")
        c = s.get("constraint_gt") or {}
        texts.append(c.get("landmark_text", "") or "")
        texts.append(c.get("attr_val_text", "") or "")
    return Vocab.build(texts, min_freq=min_freq, max_size=max_size)


def make_dataloader(samples, vocab, ctable, args, shuffle: bool):
    bbox_provider = PassthroughBBoxProvider()
    ds = RoundDatasetSemantic(
        samples=samples,
        vocab=vocab,
        max_len=args.max_len,
        bbox_provider=bbox_provider,
        img_cache_dir=args.img_cache_dir,
        imgfeat_dim=768,
        use_imgfeat=True,
        use_bbox_geom=bool(args.use_bbox_geom),

        constraint_table=ctable,
    )
    bs = args.batch_size if shuffle else args.eval_batch_size
    dl = DataLoader(
        ds,
        batch_size=bs, shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_semantic,
        drop_last=False,
    )
    return dl
@torch.no_grad()
def run_eval(model, dl_test, ctable, vocab, args, v_target_strategy, selector_strategy="ours", eval_mode="oracle"):
    model.eval()
    total_infer_time = 0.0
    total = 0
    correct = 0
    conditional_total = 0
    conditional_correct = 0
    sum_rounds = 0
    initial_target_survival = 0
    candidate_count_sum = 0
    success_at = {k: 0 for k in range(1, args.max_round + 1)}
    target_survival_at = {k: 0 for k in range(1, args.max_round + 1)}
    sum_alive_cnt = {k: 0 for k in range(1, args.max_round + 1)}

    results = []
    sample_offset = 0

    for batch in tqdm(dl_test, total=len(dl_test), desc="[EVAL]"):
        t0 = time.perf_counter()
        pred_idx, pred_logit, rounds, alive_hist, alive_cnt_hist = eval_rollout_batch(
            model=model,
            batch=batch,
            constraint_table=ctable,
            vocab=vocab,
            max_len=args.max_len,
            max_round=args.max_round,
            thr=args.thr,
            stop_thr=args.stop_thr,
            eval_mode=eval_mode,
            v_target_strategy=v_target_strategy,
            selector_strategy=selector_strategy,
        )

        t1 = time.perf_counter()
        total_infer_time += (t1 - t0)

        gt = batch["target_index"].cpu().long()
        cand_mask = batch["cand_mask"].cpu()
        B = gt.shape[0]
        valid_target = (gt >= 0) & (gt < cand_mask.shape[1])
        safe_gt = gt.clamp(0, max(cand_mask.shape[1] - 1, 0))
        initial_alive = valid_target & (cand_mask.gather(1, safe_gt.view(B, 1)).squeeze(1) > 0.5)

        ok = valid_target & (pred_idx.long() == gt)
        for r, cnt in enumerate(alive_cnt_hist):
            k = r + 1
            sum_alive_cnt[k] += int(cnt.sum().item())

        total += B
        correct += int(ok.sum().item())
        conditional_total += int(initial_alive.sum().item())
        conditional_correct += int((ok & initial_alive).sum().item())
        sum_rounds += int(rounds.sum().item())
        initial_target_survival += int(initial_alive.sum().item())
        candidate_count_sum += int(cand_mask.sum(dim=1).sum().item())

        def alive_at_k_batch(alive_hist, K):
            if len(alive_hist) == 0:
                return torch.zeros((B,), dtype=torch.bool)
            idx = min(K, len(alive_hist)) - 1
            return alive_hist[idx].bool()

        for k in range(1, args.max_round + 1):
            success_at[k] += int((ok & (rounds <= k)).sum().item())
            target_survival_at[k] += int(alive_at_k_batch(alive_hist, k).sum().item())

        final_alive = alive_at_k_batch(alive_hist, args.max_round)
        for j in range(B):
            results.append({
                "i": int(sample_offset + j),
                "pred_index": int(pred_idx[j].item()),
                "gt_index": int(gt[j].item()),
                "correct": bool(ok[j].item()),
                "pred_logit": float(pred_logit[j].item()),
                "rounds": int(rounds[j].item()),
                "initial_target_survival": bool(initial_alive[j].item()),
                "final_target_survival": bool(final_alive[j].item()),
                "candidate_count": int(cand_mask[j].sum().item()),
            })
        sample_offset += B

    avg_time_per_query = total_infer_time / max(total, 1)
    avg_time_per_batch = total_infer_time / max(len(dl_test), 1)
    acc = correct / max(total, 1)
    conditional_acc = conditional_correct / max(conditional_total, 1)
    avg_rounds = sum_rounds / max(total, 1)
    avg_alive_cnt = {
        k: sum_alive_cnt[k] / max(total, 1)
        for k in sum_alive_cnt
    }

    out = {
        "acc": float(acc),
        "num": int(total),
        "conditional_acc": float(conditional_acc),
        "conditional_num": int(conditional_total),
        "avg_rounds": float(avg_rounds),
        "initial_target_survival": float(initial_target_survival / max(total, 1)),
        "final_target_survival": float(target_survival_at[args.max_round] / max(total, 1)),
        "avg_candidate_count": float(candidate_count_sum / max(total, 1)),
        "max_round": int(args.max_round),
        "thr": float(args.thr),
        "avg_alive_cnt": avg_alive_cnt,
        "success_at": {str(k): float(v / max(total, 1)) for k, v in success_at.items()},
        "target_survival_at": {str(k): float(v / max(total, 1)) for k, v in target_survival_at.items()},
        "stop_thr": float(args.stop_thr),
        "eval_mode": eval_mode,
        "selector_strategy": selector_strategy,
        "v_target_strategy": v_target_strategy,
        "results": results,
        "avg_time_per_query": float(avg_time_per_query),
        "avg_time_per_batch": float(avg_time_per_batch),
        "total_infer_time": float(total_infer_time),
    }
    for k, value in out["success_at"].items():
        out[f"success@{k}"] = value
    for k, value in out["target_survival_at"].items():
        out[f"target_survival@{k}"] = value
        out[f"target_alive@{k}"] = value
    return out

def get_loss_switches(setting: str):
    setting_map = {
        "sel":       {"use_lsel": 1, "use_lcand": 0, "use_lstop": 0},
        "cand":      {"use_lsel": 0, "use_lcand": 1, "use_lstop": 0},
        "stop":      {"use_lsel": 0, "use_lcand": 0, "use_lstop": 1},
        "sel_cand":  {"use_lsel": 1, "use_lcand": 1, "use_lstop": 0},
        "sel_stop":  {"use_lsel": 1, "use_lcand": 0, "use_lstop": 1},
        "cand_stop": {"use_lsel": 0, "use_lcand": 1, "use_lstop": 1},
        "full":      {"use_lsel": 1, "use_lcand": 1, "use_lstop": 1},
    }
    return setting_map[setting]


def flatten_eval_summary(eval_out):
    row = {k: v for k, v in eval_out.items() if k != "results" and not isinstance(v, dict)}
    for prefix in ("success_at", "target_survival_at", "avg_alive_cnt"):
        for k, v in eval_out.get(prefix, {}).items():
            row[f"{prefix}_{k}"] = v
    return row


def save_eval_artifacts(out_eval_json, best, split, eval_history):
    payload = {
        "best": best,
        "split": split,
        "eval_history": eval_history,
    }
    save_json(out_eval_json, payload)
    csv_path = out_eval_json.rsplit(".", 1)[0] + ".csv"
    rows = [flatten_eval_summary(x) for x in eval_history]
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"saved eval log {out_eval_json}")
    if rows:
        print(f"saved eval csv {csv_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round_json_path", required=True)
    ap.add_argument("--eval_round_json_path", default="", help="Separate official evaluation split; train on all training rows")
    ap.add_argument("--img_cache_dir", required=True)
    ap.add_argument("--out_ckpt", required=True)
    ap.add_argument(
    "--selector_strategy",
    type=str,
    default="ours",
    choices=["ours", "random", "heuristic", "all_at_once"],
    help="constraint selection strategy for evaluation"
)
    ap.add_argument("--out_eval_json", required=True)
    ap.add_argument("--load_from_pretrained", type=str, default="", help="path to ckpt (.pt/.pth) to warm start")
    ap.add_argument("--load_strict", type=int, default=0, help="1=strict load, 0=non-strict (default)")
    ap.add_argument("--spec_path", type=str, required=True, help="freeze vocab/constraint_keys (+ optional split keys)")
    ap.add_argument("--build_spec", type=int, default=0, help="1=build spec from current json (TRAIN baseline only), then exit or continue")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--test_only", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--eval_mode", type=str, default="oracle",
                choices=["oracle", "agent"],
                help="oracle: round0 use constraint_gt then selector; agent: selector from round0")
    ap.add_argument(
        "--ablation_setting",
        type=str,
        default="full",
        choices=[
            "sel",
            "cand",
            "stop",
            "sel_cand",
            "sel_stop",
            "cand_stop",
            "full",
        ],
        help="stage2 loss ablation setting"
    )
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--max_len", type=int, default=16)
    ap.add_argument("--use_bbox_geom", type=int, default=0)

    ap.add_argument("--seed", type=int, default=0)

    # rollout eval params
    ap.add_argument("--max_round", type=int, default=4)
    ap.add_argument("--thr", type=float, default=0.4)
    ap.add_argument("--stop_thr", type=float, default=0.5)
    ap.add_argument("--v_target_strategy", type=str, default="top1",
                choices=["top1", "random", "mean"],
                help="v_target")

    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    switches = get_loss_switches(args.ablation_setting)
    args.use_lsel = switches["use_lsel"]
    args.use_lcand = switches["use_lcand"]
    args.use_lstop = switches["use_lstop"]
    v_target_strategy = args.v_target_strategy
        # ---------- split once, reuse forever ----------
    from stage2_refactor.utils.spec import ExperimentSpec, save_spec, load_spec

    samples_all = load_json(args.round_json_path)

    # 固定 id maps（必须固定，否则 constraint_table id 会漂）
    rel2id = {"near": 0, "behind": 1, "in_front_of": 2, "left_of": 3, "right_of": 4, "next_to": 5,
              "above": 6, "below": 7, "between": 8, "closest": 9, "farthest": 10,
              "opposite": 11, "along": 12, "inside": 13, "contains": 14}
    attr2id = {"color": 0, "material": 1, "shape": 2, "size": 3, "height": 4, "other": 5}

    def sample_key(s):
        cand_ids = [int(x) for x in s.get("cand_ids", [])]
        target_index = int(s.get("target_index", 0))
        target_id = cand_ids[target_index] if 0 <= target_index < len(cand_ids) else -1
        return (
            s.get("scene_id",""),
            str(s.get("ann_id","")),
            int(s.get("round", 0)),
            s.get("query",""),
            int(target_id),
            tuple(sorted(cand_ids)),
        )

    def normalize_key(k):
        # New format: scene, ann, round, query, target_id, sorted_cand_ids.
        # Old experimental format: scene, ann, round, query, ordered_cand_ids, target_index.
        if len(k) >= 6 and isinstance(k[4], (list, tuple)):
            cand_ids = [int(x) for x in k[4]]
            target_index = int(k[5])
            target_id = cand_ids[target_index] if 0 <= target_index < len(cand_ids) else -1
            return (k[0], str(k[1]), int(k[2]), k[3], int(target_id), tuple(sorted(cand_ids)))
        return (
            k[0],
            str(k[1]),
            int(k[2]),
            k[3],
            int(k[4]),
            tuple(int(x) for x in k[5]),
        )

    if args.build_spec:
        # 只在“训练基准集 json”上执行一次
        if args.eval_round_json_path and not args.test_only:
            train_samples = samples_all
            test_samples = load_json(args.eval_round_json_path)
            if {s["scene_id"] for s in train_samples} & {s["scene_id"] for s in test_samples}:
                raise ValueError("Training and evaluation scenes overlap")
        elif not args.test_only:
            train_samples, test_samples = split_samples_7_3(samples_all, seed=args.seed)
        else:
            train_samples, test_samples = split_samples_0_10(samples_all, seed=args.seed)

        vocab = build_vocab_from_train(train_samples)
        ctable = ConstraintTable.build_from_samples(train_samples, rel2id=rel2id, attr2id=attr2id)

        # 存 split keys（而不是 indices，避免 json 顺序变化）
        train_keys = [sample_key(s) for s in train_samples]
        test_keys  = [sample_key(s) for s in test_samples]

        spec = ExperimentSpec(
            seed=args.seed,
            train_indices=[],   # 不用 indices 了，留空
            test_indices=[],
            vocab_itos=vocab.itos,
            constraint_keys=ctable.keys,
            rel2id=rel2id,
            attr2id=attr2id,
            train_keys=train_keys,
            test_keys=test_keys,
        )
        save_spec(args.spec_path, spec)
        print(f"[spec] built & saved to {args.spec_path}")

    else:
        # 永远从 spec 固定 vocab / constraints
        spec = load_spec(args.spec_path)
        # 兼容读取 keys
        train_keys = [normalize_key(k) for k in (spec.train_keys or [])]
        test_keys = [normalize_key(k) for k in (spec.test_keys or [])]

        vocab = Vocab.from_itos(spec.vocab_itos)
        ctable = ConstraintTable(keys=spec.constraint_keys, rel2id=spec.rel2id, attr2id=spec.attr2id)

        # split 行为：
        if args.test_only:
            # test_only：0:10 —— 当前 json 全部当 test
            train_samples = []
            test_samples = samples_all
        else:
            # train/eval：必须严格复用 spec 的 split
            if not train_keys or not test_keys:
                raise RuntimeError("spec missing train_keys/test_keys. Rebuild spec with --build_spec 1 on a training baseline json.")
            evaluation_samples = load_json(args.eval_round_json_path) if args.eval_round_json_path else samples_all
            key_to_sample = {sample_key(s): s for s in samples_all + evaluation_samples}
            train_samples = [key_to_sample[k] for k in train_keys if k in key_to_sample]
            test_samples  = [key_to_sample[k] for k in test_keys  if k in key_to_sample]

            # 如果某些 key 匹配不到，说明你这次 json 和训练基准 json 的样本集合不一致
            miss_tr = len(train_keys) - len(train_samples)
            miss_te = len(test_keys) - len(test_samples)
            if miss_tr or miss_te:
                raise ValueError(f"Round JSON differs from spec: missing train={miss_tr}, eval={miss_te}")

    rel2id, attr2id = ctable.rel2id, ctable.attr2id
    if args.test_only and not args.load_from_pretrained:
        raise ValueError("--test_only requires --load_from_pretrained")
    if not test_samples or (not args.test_only and not train_samples):
        raise ValueError("Both train and evaluation splits must be nonempty")
    from pathlib import Path
    Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_eval_json).parent.mkdir(parents=True, exist_ok=True)

    print(f"[split] total={len(samples_all)} train={len(train_samples)} test={len(test_samples)} test_only={bool(args.test_only)}")
    num_constraints = ctable.num_constraints()
    print(f"[inventory] num_constraints={num_constraints} (fixed by spec)")


    # ---------- dataloaders ----------
    if not args.test_only:
        dl_train = make_dataloader(train_samples, vocab, ctable, args, shuffle=True)
    cfg = TrainConfig(
        max_len=args.max_len,
        img_cache_dir=args.img_cache_dir,
        imgfeat_dim=768,
        use_imgfeat=True,
        use_bbox_geom=bool(args.use_bbox_geom),

        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        )
    dl_test = make_dataloader(test_samples, vocab, ctable, args, shuffle=False)

    # ---------- model ----------
    

    model = Stage2NetSemantic(vocab_size=len(vocab.itos), cfg=cfg, num_constraints=num_constraints, num_rel=max(rel2id.values()) + 1, num_attr=max(attr2id.values()) + 1).to(device)

    # -------- load pretrained (optional) --------
    if args.load_from_pretrained:
        ckpt = torch.load(args.load_from_pretrained, map_location="cpu", weights_only=False)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        if args.load_strict:
            model.load_state_dict(state, strict=True)
            print(f"[pretrained] strict loaded {args.load_from_pretrained}")
        else:
            load_pretrained_flexible(model, state)
            print(f"[pretrained] flexible warm-start from {args.load_from_pretrained}")


    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    w_sel, w_mask, w_stop = 1.0, 1.0, 0.2

    best_acc = -1.0
    best_state = None
    best_epoch = -1
    eval_history = []

    def metric_line(eval_out):
        max_round = int(eval_out["max_round"])
        success_text = " ".join(
            f"S@{k}={eval_out[f'success@{k}']:.4f}" for k in range(1, max_round + 1)
        )
        survival_text = " ".join(
            f"TS@{k}={eval_out[f'target_survival@{k}']:.4f}" for k in range(1, max_round + 1)
        )
        return (
            f"acc={eval_out['acc']:.4f} cond_acc={eval_out['conditional_acc']:.4f} "
            f"cond_num={eval_out['conditional_num']} {success_text} AvgR={eval_out['avg_rounds']:.2f} "
            f"InitTS={eval_out['initial_target_survival']:.4f} "
            f"FinalTS={eval_out['final_target_survival']:.4f} {survival_text} "
            f"AvgCand={eval_out['avg_candidate_count']:.2f} "
            f"avg_alive_cnt={eval_out['avg_alive_cnt']}"
        )

    def write_eval_text(path, ep, eval_out):
        with open(path, "w") as f:
            f.write(f"Eval results at epoch {ep}:\n")
            f.write(f"Accuracy: {eval_out['acc']:.4f}\n")
            f.write(f"Conditional Accuracy: {eval_out['conditional_acc']:.4f}\n")
            f.write(f"Conditional Num: {eval_out['conditional_num']}\n")
            for k in range(1, int(eval_out["max_round"]) + 1):
                f.write(f"Success@{k}: {eval_out[f'success@{k}']:.4f}\n")
            f.write(f"Average Rounds: {eval_out['avg_rounds']:.2f}\n")
            f.write(f"Initial Target Survival: {eval_out['initial_target_survival']:.4f}\n")
            f.write(f"Final Target Survival: {eval_out['final_target_survival']:.4f}\n")
            for k in range(1, int(eval_out["max_round"]) + 1):
                f.write(f"Target Survival@{k}: {eval_out[f'target_survival@{k}']:.4f}\n")
            f.write(f"Average Candidate Count: {eval_out['avg_candidate_count']:.2f}\n")
            f.write(f"Average Alive Count: {eval_out['avg_alive_cnt']}\n")
            f.write(f"Inference Time per Query: {eval_out['avg_time_per_query']*1000:.2f} ms\n")
            f.write(f"Total Inference Time: {eval_out['total_infer_time']:.2f} s\n")

    # ---------- train loop ----------
    for ep in range(args.epochs):
        
        tot = 0.0
        n = 0
        # from ipdb import set_trace
        # set_trace()
        if not args.test_only:
            model.train()
            for batch in tqdm(dl_train, desc=f"[TRAIN] Epoch {ep}"):
                loss = torch.tensor(0.0, device=device)
                for k in batch:
                    batch[k] = batch[k].to(device)

                B = batch["cand_mask"].shape[0]
                batch["alive_mask"] = batch["cand_mask"]
                batch["used_constraint_mask"] = torch.zeros((B, num_constraints), device=device)

                out = model(batch)
                continue_rows = batch["stop_y"] < 0.5
                if continue_rows.any():
                    loss_sel = selector_ce_loss(
                        out["selector_logits"][continue_rows], batch["constraint_id"][continue_rows]
                    )
                    loss_mask = candidate_ce_loss(
                        out["mask_logits"][continue_rows],
                        batch["target_index"][continue_rows],
                        batch["cand_mask"][continue_rows],
                    )
                else:
                    loss_sel = out["selector_logits"].sum() * 0.0
                    loss_mask = out["mask_logits"].sum() * 0.0
                loss_stop = stop_bce_loss(out["stop_logit"], batch["stop_y"])
                if args.use_lsel:
                    loss += w_sel * loss_sel
                if args.use_lcand:
                    loss += w_mask * loss_mask
                if args.use_lstop:
                    loss += w_stop * loss_stop
                # loss = w_sel * loss_sel + w_mask * loss_mask + w_stop * loss_stop

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                tot += float(loss.detach().cpu())
                n += 1

            print(f"[epoch {ep + 1}] train_loss={tot/max(n,1):.4f}")

        if (ep + 1) % args.eval_every == 0 or (ep == args.epochs - 1) or args.test_only:
            eval_out = run_eval(model, dl_test, ctable, vocab, args,v_target_strategy=v_target_strategy,selector_strategy=args.selector_strategy,eval_mode=args.eval_mode)

            eval_out["epoch"] = ep + 1
            eval_history.append(eval_out)
            acc = float(eval_out["acc"])
            print(f"[epoch {ep + 1}] {metric_line(eval_out)}")
            #保存到txt文件中，按照args.round_json_path的basename命名
            if args.test_only:
                #需要保存到这个out_eval_json相同路径的叫做eval的文件夹中，名字不变
                eval_txt_path = args.out_eval_json.replace('.json', f'_{args.eval_mode}_epoch{ep}_test.txt')
            else:
                eval_txt_path = args.out_eval_json.replace('.json', f'_{args.eval_mode}_{args.ablation_setting}_{args.v_target_strategy}_{args.selector_strategy}_best.txt')

            print(f"Saved eval results to {eval_txt_path}")
            if acc > best_acc:
                best_acc = acc
                best_epoch = ep + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f"[best] updated best_acc={best_acc:.4f} at epoch={best_epoch}")
                write_eval_text(eval_txt_path, ep, eval_out)
            if args.test_only:
                save_eval_artifacts(
                    args.out_eval_json,
                    best={"acc": acc, "epoch": ep},
                    split={"train": len(train_samples), "test": len(test_samples), "seed": args.seed},
                    eval_history=eval_history,
                )
                return

        # ---------- periodic eval ----------
        
    # ---------- save best ckpt ----------
    if best_state is None:
        best_state = model.state_dict()
        best_epoch = args.epochs - 1
        best_acc = float("nan")

    torch.save({
        "state_dict": best_state,
        "model_config": asdict(cfg),
        "vocab": vocab.itos,
        "constraint_keys": ctable.keys,
        "rel2id": rel2id,
        "attr2id": attr2id,
        "max_len": args.max_len,
        "use_bbox_geom": bool(args.use_bbox_geom),
        "split": {"train": len(train_samples), "test": len(test_samples), "seed": args.seed},
        "best": {"acc": best_acc, "epoch": best_epoch},
        "eval_history": eval_history,
    }, args.out_ckpt)
    print(f"saved ckpt {args.out_ckpt} (best_acc={best_acc:.4f} best_epoch={best_epoch})")

    # ---------- save eval history json ----------
    save_eval_artifacts(
        args.out_eval_json,
        best={"acc": best_acc, "epoch": best_epoch},
        split={"train": len(train_samples), "test": len(test_samples), "seed": args.seed},
        eval_history=eval_history,
    )


if __name__ == "__main__":
    main()
