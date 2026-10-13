import unittest

import torch
import torch.nn as nn

from m3drefclip.common_ops.functions import common_ops
from m3drefclip.model.vision_module.pointgroup import (
    PointGroup,
    append_missing_scene_fallback_proposals,
    clusters_voxelization,
)


class _AllBackgroundBackbone(nn.Module):
    def __init__(self, output_dim, semantic_classes):
        super().__init__()
        self.features = nn.Parameter(torch.randn(1, output_dim))
        self.semantic_classes = semantic_classes

    def forward(self, voxel_features, voxel_xyz, voxel_point_map):
        num_points = int(voxel_point_map.numel())
        point_features = self.features.expand(num_points, -1)
        semantic_scores = torch.zeros(
            num_points,
            self.semantic_classes,
            device=point_features.device,
        )
        semantic_scores[:, 0] = 10.0
        offsets = torch.zeros(num_points, 3, device=point_features.device)
        return point_features, semantic_scores, offsets


class PointGroupEmptyProposalTest(unittest.TestCase):
    def test_fallback_is_added_only_to_missing_scene(self):
        cluster_obj_idxs = torch.tensor([0, 0], dtype=torch.int32)
        cluster_point_idxs = torch.tensor([0, 1], dtype=torch.long)
        clusters_offset = torch.tensor([0, 2], dtype=torch.int32)
        vert_batch_ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.uint8)
        semantic_scores = torch.tensor(
            [
                [3.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [3.0, 2.0, 0.1],
                [3.0, 2.0, 0.9],
                [3.0, 2.0, 0.5],
            ]
        )
        obj, points, offsets, fallback = append_missing_scene_fallback_proposals(
            cluster_obj_idxs,
            cluster_point_idxs,
            clusters_offset,
            vert_batch_ids,
            semantic_scores,
            batch_size=2,
            max_points=2,
        )
        self.assertEqual(offsets.tolist(), [0, 2, 4])
        self.assertEqual(obj.tolist(), [0, 0, 1, 1])
        self.assertEqual(set(points[2:].tolist()), {4, 5})
        self.assertEqual(fallback.tolist(), [False, True])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_empty_ball_query_does_not_launch_cuda_kernel(self):
        coords = torch.empty((0, 3), dtype=torch.float32, device="cuda")
        batch_ids = torch.empty(0, dtype=torch.uint8, device="cuda")
        offsets = torch.zeros(1, dtype=torch.int32, device="cuda")
        indices, start_len = common_ops.ballquery_batch_p(
            coords, batch_ids, offsets, 0.03, 50
        )
        self.assertEqual(indices.numel(), 0)
        self.assertEqual(tuple(start_len.shape), (0, 2))
        torch.cuda.synchronize()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_all_empty_clusters_are_replaced_before_voxelization(self):
        device = torch.device("cuda")
        vert_batch_ids = torch.tensor([0, 0, 1, 1], dtype=torch.uint8, device=device)
        semantic_scores = torch.tensor(
            [[2.0, 1.0, 0.1], [2.0, 1.0, 0.2],
             [2.0, 1.0, 0.3], [2.0, 1.0, 0.4]],
            device=device,
        )
        obj, points, offsets, fallback = append_missing_scene_fallback_proposals(
            torch.empty(0, dtype=torch.int32, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
            vert_batch_ids,
            semantic_scores,
            batch_size=2,
            max_points=2,
        )
        coords = torch.tensor(
            [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3],
             [1.0, 1.0, 1.0], [1.2, 1.1, 1.3]],
            device=device,
        )
        sparse, inverse, bounds = clusters_voxelization(
            obj,
            points,
            offsets,
            torch.randn(4, 8, device=device),
            coords,
            scale=50,
            spatial_shape=14,
            device=device,
        )
        self.assertEqual(fallback.tolist(), [True, True])
        self.assertEqual(tuple(bounds.shape), (2, 2, 3))
        self.assertEqual(inverse.numel(), 4)
        self.assertGreater(sparse.features.shape[0], 0)
        torch.cuda.synchronize()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_pointgroup_forward_survives_all_background_predictions(self):
        device = torch.device("cuda")
        model = PointGroup(
            input_channel=3,
            output_channel=32,
            max_proposals=16,
            semantic_class=18,
            use_gt=False,
        ).to(device)
        model.backbone = _AllBackgroundBackbone(32, 18).to(device)
        points_per_scene = 64
        num_points = points_per_scene * 2
        point_xyz = torch.rand(num_points, 3, device=device)
        point_xyz[points_per_scene:] += 2.0
        data = {
            "scene_id": ["scene0", "scene1"],
            "voxel_features": torch.randn(num_points, 3, device=device),
            "voxel_xyz": torch.cat(
                (
                    torch.zeros(num_points, 1, dtype=torch.int32, device=device),
                    (point_xyz * 20).int(),
                ),
                dim=1,
            ),
            "voxel_point_map": torch.arange(num_points, device=device),
            "vert_batch_ids": torch.cat(
                (
                    torch.zeros(points_per_scene, dtype=torch.uint8, device=device),
                    torch.ones(points_per_scene, dtype=torch.uint8, device=device),
                )
            ),
            "point_xyz": point_xyz,
        }
        output = model(data)
        self.assertEqual(output["proposal_masks_dense"].sum(dim=1).tolist(), [1, 1])
        self.assertEqual(
            output["fallback_proposal_masks_dense"].sum(dim=1).tolist(), [1, 1]
        )
        loss = output["aabb_features"].sum() + output["proposal_scores"][0].sum()
        loss.backward()
        self.assertIsNotNone(model.backbone.features.grad)
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
