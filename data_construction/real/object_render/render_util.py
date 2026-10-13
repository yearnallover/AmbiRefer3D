import open3d as o3d
import numpy as np
import json
import os
import torch
from PIL import Image, ImageDraw, ImageFilter  # [新增] ImageFilter 用于锐化
import glob
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
# --- PyTorch3D Check ---
try:
    import pytorch3d
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        MeshRenderer, MeshRasterizer, RasterizationSettings, SoftPhongShader,
        FoVPerspectiveCameras, look_at_view_transform, 
        AmbientLights, DirectionalLights,
        Materials, TexturesVertex, BlendParams
    )
    PYTORCH3D_AVAILABLE = True
except ImportError:
    print("PyTorch3D not available.")
    PYTORCH3D_AVAILABLE = False

class StrictViewRenderer:
    def __init__(self, mesh_path, bbox_json_path, meta_file_path=None, scan_id=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.scan_id = scan_id
        
        # 1. Load Mesh (原始加载，不再进行全局细分，节省巨量显存)
        print(f"[{scan_id}] Loading mesh from: {mesh_path}")
        self.mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
        self.mesh_o3d.remove_duplicated_vertices()
        
        if meta_file_path and os.path.exists(meta_file_path):
            align_mat = self._load_scannet_matrix(meta_file_path)
            self.mesh_o3d.transform(align_mat)
        
        verts = np.asarray(self.mesh_o3d.vertices)
        faces = np.asarray(self.mesh_o3d.triangles)
        
        if self.mesh_o3d.has_vertex_colors():
            colors = np.asarray(self.mesh_o3d.vertex_colors)
        else:
            colors = np.ones_like(verts) * 0.75

        self.full_verts = torch.from_numpy(verts).float().to(device)
        self.full_faces = torch.from_numpy(faces).long().to(device)
        self.full_colors = torch.from_numpy(colors).float().to(device)
        self.scene_center = torch.mean(self.full_verts, dim=0).cpu().numpy()

        # 2. Load JSON 
        with open(bbox_json_path, 'r') as f:
            raw_data = json.load(f)
        
        # 3. Initialize Lighting & Materials
        self.image_size = 512 
        self.fov = 60.0 
        
        # [光照] 混合光照：环境底光 + 右上方平行主光
        self.lights = DirectionalLights(
            device=self.device, 
            ambient_color=((0.5, 0.5, 0.5),), 
            diffuse_color=((0.7, 0.7, 0.7),), 
            specular_color=((0.2, 0.2, 0.2),),
            direction=((0.0, -1.0, 1.0),)
        )
        
        # [材质] 赋予物体微量高光反射
        self.materials = Materials(
            device=self.device, 
            ambient_color=[[1.0, 1.0, 1.0]], 
            diffuse_color=[[1.0, 1.0, 1.0]], 
            specular_color=[[0.1, 0.1, 0.1]], 
            shininess=15.0
        )
        
        self.blend_params = BlendParams(background_color=(1.0, 1.0, 1.0))
        self.renderer_fast = self._create_mesh_renderer(use_binning=True)
        self.renderer_safe = self._create_mesh_renderer(use_binning=False)

    def _load_scannet_matrix(self, txt_path):
        matrix = np.eye(4)
        try:
            with open(txt_path, 'r') as f:
                for line in f:
                    if line.startswith('axisAlignment'):
                        parts = line.split('=')[1].strip().split()
                        vals = [float(x) for x in parts]
                        matrix = np.array(vals).reshape(4, 4)
                        break
        except Exception: pass
        return matrix

    def _create_mesh_renderer(self, use_binning=True):
        if use_binning:
            raster_settings = RasterizationSettings(
                image_size=self.image_size, blur_radius=0.0, faces_per_pixel=1, cull_backfaces=False,bin_size=0,max_faces_per_bin=200000
            )
        else:
            raster_settings = RasterizationSettings(
                image_size=self.image_size, blur_radius=0.0, faces_per_pixel=1, cull_backfaces=False, bin_size=0
            )
        return MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings),
            shader=SoftPhongShader(device=self.device, lights=self.lights, materials=self.materials, blend_params=self.blend_params)
        )

    # =========================================================
    # [核心优化] 局部网格细分 (Local Subdivision)
    # =========================================================
    def _crop_scene_mesh(self, center, radius):
        # 1. 先用 PyTorch 在 GPU 上极速裁剪出局部粗糙网格
        faces = self.full_faces
        verts = self.full_verts
        v_idx = faces[:, 0]
        face_centers = verts[v_idx]
        dists_sq = torch.sum((face_centers - center) ** 2, dim=1)
        mask = dists_sq < (radius ** 2)
        valid_faces = faces[mask]
        
        if valid_faces.nelement() == 0: return None
        
        unique_verts_idx, new_face_indices = torch.unique(valid_faces, return_inverse=True)
        new_verts = verts[unique_verts_idx]
        new_colors = self.full_colors[unique_verts_idx]

        # 2. 将这几百个面的小网格转回 CPU/Open3D
        pcd_o3d = o3d.geometry.TriangleMesh()
        pcd_o3d.vertices = o3d.utility.Vector3dVector(new_verts.cpu().numpy())
        pcd_o3d.triangles = o3d.utility.Vector3iVector(new_face_indices.reshape(-1, 3).cpu().numpy())
        pcd_o3d.vertex_colors = o3d.utility.Vector3dVector(new_colors.cpu().numpy())
        
        # 3. [快如闪电] 仅对这一个小区域进行细分平滑 (Subdivision)
        # pcd_o3d = pcd_o3d.subdivide_midpoint(number_of_iterations=1)
        # 只在很小的局部网格上细分，避免面数爆炸
        if len(pcd_o3d.triangles) < 30000:
            pcd_o3d = pcd_o3d.subdivide_midpoint(number_of_iterations=1)

        # 4. 转回 GPU Tensor
        final_verts = torch.from_numpy(np.asarray(pcd_o3d.vertices)).float().to(self.device)
        final_faces = torch.from_numpy(np.asarray(pcd_o3d.triangles)).long().to(self.device)
        final_colors = torch.from_numpy(np.asarray(pcd_o3d.vertex_colors)).float().to(self.device)

        textures = TexturesVertex(verts_features=[final_colors])
        return Meshes(verts=[final_verts], faces=[final_faces], textures=textures)

    def _draw_bbox_on_image_thick(self, image_pil, bbox_min, bbox_max, camera, thickness=6):
        corners = torch.tensor([
            [bbox_min[0], bbox_min[1], bbox_min[2]], [bbox_min[0], bbox_min[1], bbox_max[2]],
            [bbox_min[0], bbox_max[1], bbox_min[2]], [bbox_min[0], bbox_max[1], bbox_max[2]],
            [bbox_max[0], bbox_min[1], bbox_min[2]], [bbox_max[0], bbox_min[1], bbox_max[2]],
            [bbox_max[0], bbox_max[1], bbox_min[2]], [bbox_max[0], bbox_max[1], bbox_max[2]]
        ], device=self.device).float()
        screen_points = camera.transform_points_screen(corners.unsqueeze(0), image_size=((self.image_size, self.image_size),))
        pts = screen_points[0, :, :2].cpu().detach().numpy()
        edges = [(0,1),(0,2),(0,4),(7,6),(7,5),(7,3),(1,3),(1,5),(2,3),(2,6),(4,5),(4,6)]
        draw = ImageDraw.Draw(image_pil)
        margin = 2000
        w, h = image_pil.size
        for start, end in edges:
            x1, y1 = pts[start]; x2, y2 = pts[end]
            if -margin < x1 < w + margin and -margin < y1 < h + margin:
                draw.line([(x1, y1), (x2, y2)], fill=(0, 255, 0), width=thickness)
        return image_pil

    def _get_strict_top_down_views(self, center_np, size_np):
        center_np = center_np.astype(float)
        max_dim = np.max(size_np)
        fov_rad = np.deg2rad(self.fov) / 2.0
        hypotenuse_dist = (max_dim / 2.0 / np.sin(fov_rad)) / 0.9 
        hypotenuse_dist = max(hypotenuse_dist, 0.8)
        z_offset = hypotenuse_dist * 0.707
        xy_radius = hypotenuse_dist * 0.707
        candidates = []
        for azim in range(0, 360, 45):
            rad = np.deg2rad(azim)
            eye_pos = np.array([center_np[0] + xy_radius * np.cos(rad), center_np[1] + xy_radius * np.sin(rad), center_np[2] + z_offset])
            dist_to_room_center = np.linalg.norm(eye_pos - self.scene_center)
            candidates.append({"eye": eye_pos, "score": dist_to_room_center, "name": f"azim_{azim}"})
        candidates.sort(key=lambda x: x["score"])
        return candidates[:3]

# ==============================================================================
# Manager Class
# ==============================================================================
class SceneRenderManager:
    def __init__(self, dataset_root_dir, output_root_dir, bbox_root_dir=None):
        self.dataset_root = dataset_root_dir
        self.output_root = output_root_dir
        self.bbox_root = bbox_root_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"SceneRenderManager initialized on {self.device}")

    def _get_scene_files(self, scene_id):
        scene_dir = os.path.join(self.dataset_root, scene_id)
        ply_path = os.path.join(scene_dir, f"{scene_id}_vh_clean.ply")
        meta_path = os.path.join(scene_dir, f"{scene_id}.txt")
        
        bbox_path = None
        if self.bbox_root and os.path.exists(self.bbox_root):
            possible_paths = [
                os.path.join(self.bbox_root, scene_id, "objects.json"),
                os.path.join(self.bbox_root, "objects.json")
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    bbox_path = p
                    break
        if not bbox_path:
            candidates = [
                os.path.join(scene_dir, f"{scene_id}_bboxes_with_origin.json"),
                os.path.join(scene_dir, "objects.json"),
                os.path.join(scene_dir, f"{scene_id}.aggregation.json")
            ]
            for p in candidates:
                if os.path.exists(p):
                    bbox_path = p
                    break

        if not os.path.exists(ply_path): return None, None, None
        if not bbox_path or not os.path.exists(bbox_path): return None, None, None
        return ply_path, meta_path, bbox_path

    def render_scene_objects(self, scene_id, target_object_names):
        print(f"\n=== Processing Scene: {scene_id} ===")
        ply_path, meta_path, bbox_path = self._get_scene_files(scene_id)
        if not ply_path or not bbox_path: return

        renderer = StrictViewRenderer(
            ply_path, bbox_path, meta_file_path=meta_path, scan_id=scene_id, device=self.device
        )

        scene_output_dir = os.path.join(self.output_root, scene_id)
        rendered_count = 0
        target_names_lower = [name.lower() for name in target_object_names]

        for obj in renderer.objects_list:
            obj_label = obj.get('label', '').lower()
            is_match = any(name in obj_label for name in target_names_lower)
            if is_match:
                renderer.render_object(obj["id"], scene_output_dir)
                rendered_count += 1
        
        print(f"  Finished. Rendered {rendered_count} objects.")

def main():
    if not PYTORCH3D_AVAILABLE: return
    # Path Config
    DATASET_ROOT = "data/scannetv2"  
    BBOX_ROOT = "data/scenegraph" 
    OUTPUT_ROOT = "data/multi_object_renders"

    manager = SceneRenderManager(DATASET_ROOT, OUTPUT_ROOT, bbox_root_dir=BBOX_ROOT)
    tasks = [("scene0011_00", ["chair", "table", "cabinet"])]
    for scene_id, target_names in tasks:
        manager.render_scene_objects(scene_id, target_names)

if __name__ == "__main__":
    main()