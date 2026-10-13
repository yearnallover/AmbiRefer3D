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
    use_relative_geom: bool = False
    use_semantic_label: bool = False
    num_semantic_classes: int = 256
    use_stage1_features: bool = False
    stage1_feat_dim: int = 128
    use_stage1_quality: bool = False
    use_scene_context_attention: bool = False
    use_constraint_conditioned_context: bool = False
    # False preserves the behavior of checkpoints trained before the audit.
    selector_pre_action_state: bool = False
    stage1_cache_dir: str = ""
    stage1_feature_key: str = "features_3d"
    use_stage1_cache_bboxes: bool = False
    require_stage1_cache: bool = False

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

    # losses
    w_policy: float = 0.0   # Path A: optional, keep 0 for now
    w_mask: float = 1.0
    w_rank: float = 0.5
    w_stop: float = 0.2

    rank_margin: float = 0.3
    rank_pairs: int = 8
