import json
from pathlib import Path
from typing import Any, Dict, List

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def ensure_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else [x]

def as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and len(x.strip()) > 0
