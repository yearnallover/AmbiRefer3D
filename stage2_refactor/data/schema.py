from dataclasses import dataclass
from typing import Optional, Dict, Any, List

@dataclass
class SemanticConstraintGT:
    # ctype: "REL" or "ATTR"
    ctype: str
    rel: Optional[str] = None             # e.g. "near", "behind"
    landmark_text: Optional[str] = None   # e.g. "cabinet", "curtain"
    attr_key: Optional[str] = None        # e.g. "color", "material"
    attr_val_text: Optional[str] = None   # e.g. "brown", "marble", "low"

@dataclass
class RoundSample:
    scene_id: str
    ann_id: str
    query: str
    cand_ids: List[int]
    # real stage1 may provide these; simulation may load them
    cand_bboxes: Optional[List[List[float]]] = None  # [C,7]
    cand_imgfeat: Optional[List[List[float]]] = None # [C,D]
    constraint_gt: Optional[SemanticConstraintGT] = None

    # supervision
    target_index: int = 0   # target always 0 in your cand_ids layout
    stop_y: int = 0
    meta: Optional[Dict[str, Any]] = None
