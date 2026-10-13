from __future__ import annotations
import argparse
import json
import torch
from torch.utils.data import DataLoader

from stage2_refactor.utils.io import load_json, save_json
from stage2_refactor.utils.misc import set_seed, get_device
from stage2_refactor.data.vocab import Vocab
from stage2_refactor.data.bbox_provider import PassthroughBBoxProvider
from stage2_refactor.data.dataset_semantic import RoundDatasetSemantic
from stage2_refactor.data.collate_semantic import collate_semantic
from stage2_refactor.model.stage2_net_semantic import Stage2NetSemantic
from stage2_refactor.learning.constraint_table import ConstraintTable
from tqdm import tqdm
from stage2_refactor.eval_rollout import eval_rollout_one

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round_json_path", required=True)
    ap.add_argument("--img_cache_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_path", required=True)

    ap.add_argument("--max_round", type=int, default=4)
    ap.add_argument("--thr", type=float, default=0.4)
    ap.add_argument("--stop_thr", type=float, default=0.5)
    ap.add_argument("--max_len", type=int, default=16)
    ap.add_argument("--use_bbox_geom", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    vocab = Vocab.from_itos(ckpt["vocab"])
    rel2id = ckpt["rel2id"]
    attr2id = ckpt["attr2id"]
    ctable = ConstraintTable(keys=ckpt["constraint_keys"], rel2id=rel2id, attr2id=attr2id)
    num_constraints = ctable.num_constraints()

    from stage2_refactor.config import TrainConfig
    cfg = TrainConfig(
        max_len=args.max_len,
        img_cache_dir=args.img_cache_dir,
        imgfeat_dim=768,
        use_imgfeat=True,
        use_bbox_geom=bool(args.use_bbox_geom),
    )

    model = Stage2NetSemantic(vocab_size=len(vocab.itos), cfg=cfg, num_constraints=num_constraints, num_rel=max(rel2id.values()) + 1, num_attr=max(attr2id.values()) + 1).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    samples = load_json(args.round_json_path)

    # eval dataset: we just need the candidate set + query.
    # We'll reuse RoundDatasetSemantic; constraint fields will be overwritten by ConstraintTable during rollout.
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
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_semantic)

    results = []
    correct = 0
    total = 0

    for i, batch in tqdm(enumerate(dl),desc="[INFO]Eval:"):
        pred_idx, pred_logit, rounds = eval_rollout_one(
            model=model,
            batch=batch,
            constraint_table=ctable,
            vocab=vocab,
            max_len=args.max_len,
            max_round=args.max_round,
            thr=args.thr,
            stop_thr=args.stop_thr,
        )

        # ground truth target_index
        gt = int(batch["target_index"].item())
        ok = (pred_idx == gt)
        correct += int(ok)
        total += 1

        results.append({
            "i": i,
            "pred_index": pred_idx,
            "gt_index": gt,
            "correct": bool(ok),
            "pred_logit": float(pred_logit),
            "rounds": int(rounds),
        })

    acc = correct / max(total, 1)
    out = {"acc": acc, "num": total, "max_round": args.max_round, "thr": args.thr, "stop_thr": args.stop_thr, "results": results}
    save_json(args.out_path, out)
    print(f"acc={acc:.4f} saved {args.out_path}")

if __name__ == "__main__":
    main()
