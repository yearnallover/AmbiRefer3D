from __future__ import annotations
import argparse
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from stage2_refactor.utils.io import load_json
from stage2_refactor.utils.misc import set_seed, get_device
from stage2_refactor.data.vocab import Vocab
from stage2_refactor.data.bbox_provider import PassthroughBBoxProvider
from stage2_refactor.data.dataset_semantic import RoundDatasetSemantic
from stage2_refactor.data.collate_semantic import collate_semantic
from stage2_refactor.model.stage2_net_semantic import Stage2NetSemantic
from stage2_refactor.learning.constraint_table import ConstraintTable
from stage2_refactor.learning.losses_semantic import candidate_ce_loss, selector_ce_loss, stop_bce_loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round_json_path", required=True)
    ap.add_argument("--img_cache_dir", required=True)
    ap.add_argument("--out_ckpt", required=True)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--max_len", type=int, default=16)
    ap.add_argument("--use_bbox_geom", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()

    samples_all = load_json(args.round_json_path)

    from stage2_refactor.utils.split import split_samples_7_3
    train_samples, test_samples = split_samples_7_3(samples_all, seed=args.seed)

    print(f"[split] total={len(samples_all)} train={len(train_samples)} test={len(test_samples)}")
    samples = train_samples

    # from ipdb import set_trace
    # set_trace()
    # vocab over query + field text
    texts = []
    for s in samples:
        texts.append(s.get("query", "") or "")
        c = s.get("constraint_gt") or {}
        texts.append(c.get("landmark_text", "") or "")
        texts.append(c.get("attr_val_text", "") or "")
    vocab = Vocab.build(texts, min_freq=1, max_size=20000)

    # maps must match dataset/model
    rel2id = {"near": 0, "behind": 1, "in_front_of": 2, "left_of": 3, "right_of": 4, "next_to": 5}
    attr2id = {"color": 0, "material": 1, "shape": 2, "size": 3, "height": 4, "other": 5}

    ctable = ConstraintTable.build_from_samples(samples, rel2id=rel2id, attr2id=attr2id)
    num_constraints = ctable.num_constraints()

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
        constraint_table=ctable,   # <-- make sure your dataset __init__ supports this
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_semantic)

    # model config (minimal inline)
    from stage2_refactor.config import TrainConfig
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

    model = Stage2NetSemantic(vocab_size=len(vocab.itos), cfg=cfg, num_constraints=num_constraints).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    w_sel, w_mask, w_stop = 1.0, 1.0, 0.2

    model.train()
    for ep in range(args.epochs):
        tot = 0.0
        n = 0
        for batch in tqdm(dl, desc=f"Epoch {ep}"):
            for k in batch:
                batch[k] = batch[k].to(device)

            # training masks: all alive, none used
            B = batch["cand_mask"].shape[0]
            batch["alive_mask"] = batch["cand_mask"]
            batch["used_constraint_mask"] = torch.zeros((B, num_constraints), device=device)

            out = model(batch)
            loss_sel = selector_ce_loss(out["selector_logits"], batch["constraint_id"])
            loss_mask = candidate_ce_loss(out["mask_logits"], batch["target_index"], batch["cand_mask"])
            loss_stop = stop_bce_loss(out["stop_logit"], batch["stop_y"])
            loss = w_sel * loss_sel + w_mask * loss_mask + w_stop * loss_stop

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot += float(loss.detach().cpu())
            n += 1

        print(f"[epoch {ep}] loss={tot/max(n,1):.4f}")

    torch.save({
        "state_dict": model.state_dict(),
        "vocab": vocab.itos,
        "constraint_keys": ctable.keys,
        "rel2id": rel2id,
        "attr2id": attr2id,
        "max_len": args.max_len,
        "use_bbox_geom": bool(args.use_bbox_geom),
        "split": {"train": len(train_samples), "test": len(test_samples)},
    }, args.out_ckpt)
    print("saved", args.out_ckpt)

if __name__ == "__main__":
    main()
