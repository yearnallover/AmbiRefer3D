# import torch
# import lightning.pytorch as pl
# from m3drefclip.common_ops.functions import common_ops
# from pytorch3d.structures import Pointclouds
# from pytorch3d.renderer import (
#     look_at_view_transform,
#     FoVOrthographicCameras,
#     PointsRasterizationSettings,
#     PointsRenderer,
#     PointsRasterizer,
#     AlphaCompositor
# )


# class ObjectRenderer(pl.LightningModule):
#     def __init__(self, eye, rasterizer_setting):
#         super().__init__()
#         self.R, self.T = look_at_view_transform(eye=eye, at=((0, 0, 0),), up=((0, 0, 1),), device=self.device)
#         self.image_size = rasterizer_setting.image_size
#         self.views = len(eye)
#         self.renderer = PointsRenderer(
#             rasterizer=PointsRasterizer(
#                 cameras=None, raster_settings=PointsRasterizationSettings(**rasterizer_setting)
#             ), compositor=AlphaCompositor()
#         )
#         for param in self.parameters():
#             param.requires_grad = False

#     def forward(self, data_dict, output_dict):
#         batch_size = len(data_dict["scene_id"])
#         total_num_aabbs = output_dict["pred_aabb_min_max_bounds"].shape[0]
#         output_imgs = torch.zeros(
#             size=(total_num_aabbs * self.views, self.image_size, self.image_size, 3),
#             dtype=torch.float32, device=self.device
#         )

#         for i in range(batch_size):
#             batch_points_start_idx = data_dict["all_point_count_offsets"][i]
#             batch_points_end_idx = data_dict["all_point_count_offsets"][i + 1]
#             current_pcd_xyz = data_dict["all_point_xyz"][batch_points_start_idx:batch_points_end_idx]
#             current_pcd_rgb = data_dict["all_point_rgb"][batch_points_start_idx:batch_points_end_idx]

#             pred_aabb_start_idx = output_dict["proposal_batch_offsets"][i]
#             pred_aabb_end_idx = output_dict["proposal_batch_offsets"][i + 1]

#             output_masks = common_ops.crop_pcd_from_aabbs(
#                 output_dict["pred_aabb_min_max_bounds"][pred_aabb_start_idx:pred_aabb_end_idx],
#                 current_pcd_xyz
#             )
#             aabb_xyz_list = []
#             aabb_rgb_list = []
#             for obj_i in range(output_masks.shape[0]):
#                 current_obj_point_indicies = output_masks[obj_i]
#                 if not current_obj_point_indicies.any():
#                     obj_pcd_xyz = torch.empty(size=(0, 3), device=self.device, dtype=torch.float32)
#                     obj_pcd_rgb = torch.empty(size=(0, 3), device=self.device, dtype=torch.float32)
#                 else:
#                     obj_pcd_xyz = current_pcd_xyz[current_obj_point_indicies]
#                     obj_pcd_rgb = current_pcd_rgb[current_obj_point_indicies]
#                     obj_pcd_xyz -= obj_pcd_xyz.mean(dim=0)
#                     obj_pcd_xyz /= obj_pcd_xyz.abs().max()
#                 for _ in range(self.views):
#                     aabb_xyz_list.append(obj_pcd_xyz)
#                     aabb_rgb_list.append(obj_pcd_rgb)
#             pytorch3d_pcd = Pointclouds(points=aabb_xyz_list, features=aabb_rgb_list)
#             pytorch3d_pcd.device = self.device
#             num_aabbs = len(aabb_xyz_list) // self.views

#             R = self.R.expand(num_aabbs, -1, -1, -1).flatten(0, 1)
#             T = self.T.expand(num_aabbs, -1, -1).flatten(0, 1)

#             output_imgs[pred_aabb_start_idx * self.views:pred_aabb_end_idx * self.views] = self.renderer(
#                 pytorch3d_pcd, dtype=torch.float32, device=self.device,
#                 cameras=FoVOrthographicCameras(device=self.device, R=R, T=T, znear=0.01)
#             )

#             # pytorch3d_pcd = None
#             # aabb_xyz_list = None
#             # aabb_rgb_list = None
#             # torch.cuda.empty_cache()

#         return output_imgs


import torch
import lightning.pytorch as pl
from m3drefclip.common_ops.functions import common_ops
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)


class ObjectRenderer(pl.LightningModule):
    def __init__(self, eye, rasterizer_setting):
        """
        eye: list/tuple of (x, y, z)
        rasterizer_setting: DictConfig, 里面有 image_size 等字段
        """
        super().__init__()

        # 在 CPU 上先构造视角变换，作为 buffer，跟随 module 一起迁移设备
        R, T = look_at_view_transform(
            eye=eye,
            at=((0, 0, 0),),
            up=((0, 0, 1),),
        )  # R: (V, 3, 3), T: (V, 3)

        # Keep camera matrices as runtime buffers but out of checkpoints, so
        # older checkpoints without these keys can still load strictly.
        self.register_buffer("R", R, persistent=False)  # [V, 3, 3]
        self.register_buffer("T", T, persistent=False)  # [V, 3]

        self.image_size = rasterizer_setting.image_size
        self.views = len(eye)

        raster_settings = PointsRasterizationSettings(**rasterizer_setting)

        self.renderer = PointsRenderer(
            rasterizer=PointsRasterizer(
                cameras=None,  # cameras 在 forward 里动态传入
                raster_settings=raster_settings,
            ),
            compositor=AlphaCompositor(),
        )

        # 该模块不参与训练
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, data_dict, output_dict):
        device = self.device
        batch_size = len(data_dict["scene_id"])

        # total_num_aabbs = 所有 batch 的 proposal 总数
        total_num_aabbs = output_dict["pred_aabb_min_max_bounds"].shape[0]

        # 预分配：每个 proposal * views 张图
        output_imgs = torch.zeros(
            size=(total_num_aabbs * self.views, self.image_size, self.image_size, 3),
            dtype=torch.float32,
            device=device,
        )

        # 把 R/T 搬到当前 device
        R_all = self.R.to(device)  # [V, 3, 3]
        T_all = self.T.to(device)  # [V, 3]

        for b in range(batch_size):
            scene_id = data_dict["scene_id"][b]

            # ---------------- 1. 当前 batch 的点云 ----------------
            pts_start = data_dict["all_point_count_offsets"][b].item()
            pts_end = data_dict["all_point_count_offsets"][b + 1].item()

            current_pcd_xyz = data_dict["all_point_xyz"][pts_start:pts_end]  # [N_b, 3]
            current_pcd_rgb = data_dict["all_point_rgb"][pts_start:pts_end]  # [N_b, 3]

            # ---------------- 2. 当前 batch 的 proposal 范围 ----------------
            aabb_start = output_dict["proposal_batch_offsets"][b].item()
            aabb_end = output_dict["proposal_batch_offsets"][b + 1].item()

            cur_bounds = output_dict["pred_aabb_min_max_bounds"][aabb_start:aabb_end]
            num_aabbs_b = cur_bounds.shape[0]

            # 如果这一批没有 proposal，直接跳过
            if num_aabbs_b == 0:
                continue

            # 3. 根据 AABB 裁剪点云：mask: [num_aabbs_b, N_b]
            output_masks = common_ops.crop_pcd_from_aabbs(cur_bounds, current_pcd_xyz)

            if output_masks.shape[0] != num_aabbs_b:
                print(
                    f"[ObjectRenderer][WARN] output_masks.shape[0] != num_aabbs_b: "
                    f"{output_masks.shape[0]} vs {num_aabbs_b}, scene={scene_id}",
                    flush=True,
                )

            # ---------------- 3. 为每个 proposal 构造一个 Pointcloud ----------------
            aabb_xyz_list = []
            aabb_rgb_list = []

            for obj_i in range(num_aabbs_b):
                mask = output_masks[obj_i]  # [N_b]
                num_pts = int(mask.sum().item())

                if num_pts <= 0:
                    # 避免 0 点云触发 Pytorch3D 的边界 bug，构造一个 1 点 dummy cloud
                    obj_xyz = current_pcd_xyz.new_zeros((1, 3))
                    obj_rgb = current_pcd_rgb.new_zeros((1, 3))
                else:
                    obj_xyz = current_pcd_xyz[mask].clone()
                    obj_rgb = current_pcd_rgb[mask].clone()

                    # 中心化 + 归一化到 [-1, 1]
                    obj_xyz = obj_xyz - obj_xyz.mean(dim=0, keepdim=True)
                    max_abs = obj_xyz.abs().max()
                    if max_abs > 0:
                        obj_xyz = obj_xyz / max_abs

                aabb_xyz_list.append(obj_xyz)
                aabb_rgb_list.append(obj_rgb)

            # 这里应该严格等于 num_aabbs_b
            if len(aabb_xyz_list) != num_aabbs_b:
                print(
                    f"[ObjectRenderer][WARN] len(aabb_xyz_list)={len(aabb_xyz_list)} "
                    f"!= num_aabbs_b={num_aabbs_b}, scene={scene_id}",
                    flush=True,
                )

            pytorch3d_pcd = Pointclouds(
                points=aabb_xyz_list,
                features=aabb_rgb_list,
            ).to(device)

            # ---------------- 4. 按 view 循环渲染，并写回 output_imgs ----------------
            # 这一批 proposals 在全局 output_imgs 中的起始位置（以 proposal index 为基准）
            base_idx = aabb_start * self.views  # proposal j 对应 index = j * views + v

            for v in range(self.views):
                # 当前视角的相机参数, 扩展到 num_aabbs_b 个 camera
                Rv = R_all[v].unsqueeze(0).expand(num_aabbs_b, -1, -1)  # [num_aabbs_b, 3, 3]
                Tv = T_all[v].unsqueeze(0).expand(num_aabbs_b, -1)      # [num_aabbs_b, 3]

                cameras = FoVOrthographicCameras(
                    device=device,
                    R=Rv,
                    T=Tv,
                    znear=0.01,
                )

                try:
                    # 渲染: [num_aabbs_b, H, W, 3]
                    imgs = self.renderer(pytorch3d_pcd, cameras=cameras)
                except RuntimeError as e:
                    # 核心容错：一旦 Pytorch3D 在某个 batch 上出错，用全 0 图像顶上，避免训练直接崩溃
                    print(
                        f"[ObjectRenderer][ERROR] rasterizer RuntimeError in batch={b}, "
                        f"view={v}, scene={scene_id}, num_aabbs_b={num_aabbs_b}: {e}",
                        flush=True,
                    )
                    imgs = torch.zeros(
                        (num_aabbs_b, self.image_size, self.image_size, 3),
                        dtype=torch.float32,
                        device=device,
                    )

                # 写回: 对于第 obj_i 个 proposal，全局 index = (aabb_start + obj_i) * views + v
                for obj_i in range(num_aabbs_b):
                    global_idx = base_idx + obj_i * self.views + v
                    output_imgs[global_idx] = imgs[obj_i]

        return output_imgs
