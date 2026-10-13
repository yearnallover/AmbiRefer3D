from dataclasses import dataclass

@dataclass
class TrainConfig:
    # paths
    round_json_path: str = "data/round_samples.json"
    img_cache_dir: str = "outputs/imgfeat_cache"   # scene_id.npz with obj_ids, feats
    scannet_root: str = ""                         # if you still want bbox provider; can be empty

    # text
    max_len: int = 16

    # candidate inputs
    use_imgfeat: bool = True
    imgfeat_dim: int = 768  # ViT-B/32
    use_bbox_geom: bool = False  # Path A default: rely on image; you can turn on later

    # model dims
    d_text: int = 256
    d_constraint: int = 256
    d_cand: int = 256
    d_state: int = 256

    # training
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-2
    epochs: int = 5
    grad_clip: float = 1.0
    seed: int = 42

