import os
import json
import numpy as np
import open3d as o3d
import random
import h5py
import torch
import MinkowskiEngine as ME
from collections import defaultdict


# ============================================================
# Part A: 3D Reconstruction & Feature Extraction
# ============================================================

def export_scene_ply(objects, output_path):
    all_points, all_colors, all_sem, all_inst = [], [], [], []

    for obj in objects:
        verts = np.array(obj.get_mesh_vertices())  
        colors = (np.array(obj.get_vertex_colors()) * 255)
        inst_id = int(obj.get_cp("object_id"))
        sem_id = int(obj.get_cp("category_id"))

        all_points.append(verts)
        all_colors.append(colors)
        all_sem.append(np.full(len(verts), sem_id))
        all_inst.append(np.full(len(verts), inst_id))

    xyz = np.concatenate(all_points, axis=0)
    rgb = np.concatenate(all_colors, axis=0)
    semantic = np.concatenate(all_sem, axis=0)
    instance = np.concatenate(all_inst, axis=0)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    pc.colors = o3d.utility.Vector3dVector(rgb / 255)
    o3d.io.write_point_cloud(output_path, pc)

    np.save(output_path.replace(".ply", "_semantic.npy"), semantic)
    np.save(output_path.replace(".ply", "_instance.npy"), instance)


def export_segs_json(objects, segs_json_path):
    seg_indices = []
    current_seg = 0
    for obj in objects:
        seg_indices.extend([current_seg] * len(obj.get_mesh_vertices()))
        current_seg += 1
    json.dump({"segIndices": seg_indices}, open(segs_json_path, "w"))


def export_aggregation(scene_graph_path, segs_json_path, output_path):
    sg = json.load(open(scene_graph_path))
    groups = []
    seg_id = 0

    for obj in sg["objects"]:
        groups.append({
            "id": obj["object_id"],
            "label": obj["category"],
            "segments": [seg_id]
        })
        seg_id += 1

    json.dump({"segGroups": groups}, open(output_path, "w"))


def export_voxel_pth(ply_path, output_path, voxel_size=0.02):
    pc = o3d.io.read_point_cloud(ply_path)
    xyz = np.asarray(pc.points)
    rgb = (np.asarray(pc.colors) * 255).astype(np.float32)

    sem = np.load(ply_path.replace(".ply", "_semantic.npy"))
    inst = np.load(ply_path.replace(".ply", "_instance.npy"))

    coords, feats, _ = ME.utils.sparse_quantize(
        coordinates=xyz,
        features=rgb,
        labels=sem,
        quantization_size=voxel_size
    )

    torch.save({
        "coords": torch.tensor(coords),
        "feats": torch.tensor(feats),
        "semantic_labels": torch.tensor(sem),
        "instance_labels": torch.tensor(inst)
    }, output_path)


def export_enet_features(scene_id, render_dir, output_path, enet):
    views = sorted([f for f in os.listdir(render_dir) if f.endswith(".png")])
    feats = []

    for v in views:
        img = load_image(os.path.join(render_dir, v))
        feat = enet(img).cpu().numpy()
        feats.append(feat)

    feats = np.stack(feats).max(axis=0)

    with h5py.File(output_path, "a") as f:
        f.create_dataset(scene_id, data=feats)


# ============================================================
# Part B: Multi3DRefer Annotation Generation
# ============================================================

def build_objects(scene_graph):
    objs = []
    for obj in scene_graph["objects"]:
        objs.append({
            "object_id": obj["object_id"],
            "name": obj.get("instance_name"),
            "category": obj.get("category"),
            "super_category": obj.get("super_category"),
            "attributes": obj.get("attributes", {}),
            "room_id": obj.get("room_id"),
            "room_name": obj.get("room_name")
        })
    return objs


def build_bboxes(scene_graph):
    bboxes = {}
    for obj in scene_graph["objects"]:
        obj_id = obj["object_id"]
        bbox = obj["bbox_3d"]
        bboxes[obj_id] = {
            "center": bbox.get("center"),
            "size": bbox.get("size"),
            "corners": bbox.get("corners"),
            "orientation_euler": obj.get("orientation", {}).get("euler"),
            "location": obj.get("location")
        }
    return bboxes


def build_relations(scene_graph):
    relations = defaultdict(list)
    if "relations" not in scene_graph:
        return relations

    for rel in scene_graph["relations"]:
        relations[rel["subj"]].append({
            "predicate": rel["predicate"],
            "object": rel["obj"],
            "distance": rel.get("geom", {}).get("dist")
        })
    return relations


def build_splits(scene_graph, train_ratio=0.7, val_ratio=0.15):
    obj_ids = [o["object_id"] for o in scene_graph["objects"]]
    random.shuffle(obj_ids)
    n = len(obj_ids)
    return {
        "train": obj_ids[:int(n * train_ratio)],
        "val": obj_ids[int(n * train_ratio):int(n * (train_ratio + val_ratio))],
        "test": obj_ids[int(n * (train_ratio + val_ratio)):]
    }


# ============================================================
# Unified Export Function
# ============================================================

def export_multi3drefer_scene(scene_id, out_dir, objects, scene_graph_path, render_dir):

    scene_graph = json.load(open(scene_graph_path))

    # =========================
    # A. 3D Data
    # =========================
    scannet_dir = out_dir / "multi3drefer"
    os.makedirs(scannet_dir, exist_ok=True)

    ply_path = scannet_dir / f"{scene_id}.ply"
    export_scene_ply(objects, str(ply_path))

    segs_path = scannet_dir / "segs.json"
    export_segs_json(objects, str(segs_path))

    agg_path = scannet_dir / "aggregation.json"
    export_aggregation(scene_graph_path, str(segs_path), str(agg_path))

    voxel_path = scannet_dir / f"{scene_id}.pth"
    export_voxel_pth(str(ply_path), str(voxel_path))

    # enet
    enet = load_enet()
    export_enet_features(scene_id, render_dir, str(scannet_dir / "enet_feats.hdf5"), enet)

    # =========================
    # B. Multi3DRefer metadata
    # =========================
    anno_dir = out_dir / "annotations"
    os.makedirs(anno_dir, exist_ok=True)

    json.dump(build_objects(scene_graph), open(anno_dir / "objects.json", "w"), indent=2)
    json.dump(build_bboxes(scene_graph), open(anno_dir / "bbox.json", "w"), indent=2)
    json.dump(build_relations(scene_graph), open(anno_dir / "relations.json", "w"), indent=2)
    json.dump(build_splits(scene_graph), open(anno_dir / "splits.json", "w"), indent=2)

    print(f"[OK] Scene {scene_id} 完整 Multi3DRefer 数据导出完成！")
