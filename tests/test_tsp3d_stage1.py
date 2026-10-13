import torch

from m3drefclip.model.stage1_backbones import TSP3DStage1Backbone


def test_tsp3d_anchor_nms_preserves_spatially_distinct_candidates():
    boxes = torch.tensor(
        [
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.1, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ]
    )
    scores = torch.tensor([3.0, 2.0, 1.0])

    keep = TSP3DStage1Backbone._axis_aligned_nms(
        boxes, scores, threshold=0.5, max_keep=3
    )

    assert keep.tolist() == [0, 2]


def test_tsp3d_center_size_conversion():
    boxes = torch.tensor([[1.0, 2.0, 3.0, 2.0, 4.0, 6.0]])
    bounds = TSP3DStage1Backbone._center_size_to_bounds(boxes)
    expected = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]]])
    torch.testing.assert_close(bounds, expected)

