import json
import os
import random
from collections import defaultdict

INPUT_SCENE_GRAPH = "scene_graph.json"
OUT_DIR = "generated_annotations"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 1. objects.json
# ============================================================

def build_objects(scene_graph):
    objs = []

    for obj in scene_graph["objects"]:
        objs.append({
            "object_id": obj["object_id"],
            "name": obj.get("instance_name", None),
            "category": obj.get("category", None),
            "super_category": obj.get("super_category", None),
            "attributes": obj.get("attributes", {}),
            "room_id": obj.get("room_id", None),
            "room_name": obj.get("room_name", None)
        })

    return objs


# ============================================================
# 2. bbox.json
# ============================================================

def build_bboxes(scene_graph):
    bboxes = {}

    for obj in scene_graph["objects"]:
        obj_id = obj["object_id"]
        bbox = obj.get("bbox_3d", {})

        bboxes[obj_id] = {
            "center": bbox.get("center"),
            "size": bbox.get("size"),
            "corners": bbox.get("corners"),
            "location": obj.get("location"),
            "orientation_euler": obj.get("orientation", {}).get("euler", None)
        }

    return bboxes


# ============================================================
# 3. relations.json（使用你提供的 subj/obj/predicate）
# ============================================================

def build_relations(scene_graph):
    relations = defaultdict(list)

    # 你没有贴出关系字段实际 key，但一般写成 scene_graph["relations"]
    if "relations" not in scene_graph:
        print("WARNING: no 'relations' field found in scene_graph.")
        return relations

    for rel in scene_graph["relations"]:
        subj = rel.get("subj")
        obj = rel.get("obj")
        predicate = rel.get("predicate")
        geom = rel.get("geom", {})

        relations[subj].append({
            "predicate": predicate,
            "object": obj,
            "distance": geom.get("dist", None)
        })

    return relations


# ============================================================
# 4. splits.json（随机划分 70/15/15）
# ============================================================

def build_splits(scene_graph, train_ratio=0.7, val_ratio=0.15):
    obj_ids = [obj["object_id"] for obj in scene_graph["objects"]]
    random.shuffle(obj_ids)

    n = len(obj_ids)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return {
        "train": obj_ids[:train_end],
        "val": obj_ids[train_end:val_end],
        "test": obj_ids[val_end:]
    }


# ============================================================
# Main
# ============================================================

def main():
    with open(INPUT_SCENE_GRAPH, "r") as f:
        scene_graph = json.load(f)

    # objects
    objects = build_objects(scene_graph)
    json.dump(objects, open(f"{OUT_DIR}/objects.json", "w"), indent=2)

    # bbox
    bboxes = build_bboxes(scene_graph)
    json.dump(bboxes, open(f"{OUT_DIR}/bbox.json", "w"), indent=2)

    # relations
    relations = build_relations(scene_graph)
    json.dump(relations, open(f"{OUT_DIR}/relations.json", "w"), indent=2)

    # splits
    splits = build_splits(scene_graph)
    json.dump(splits, open(f"{OUT_DIR}/splits.json", "w"), indent=2)

    print("Generated: objects.json, bbox.json, relations.json, splits.json")


if __name__ == "__main__":
    main()
