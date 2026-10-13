import unittest

import torch

from m3drefclip.loss.reference_loss import RefBCELoss
from m3drefclip.loss.online_trajectory_loss import OnlineTrajectoryLoss
from m3drefclip.model.trajectory_grounding_adapter import (
    TrajectoryConditionedGroundingAdapter,
)
from m3drefclip.util.utils import get_batch_aabb_pair_ious


class NumericStabilityTest(unittest.TestCase):
    def test_trajectory_zero_terms_are_finite_with_min_value_padding(self):
        loss_fn = OnlineTrajectoryLoss()
        min_value = torch.finfo(torch.float32).min
        scores = torch.tensor(
            [[[[1.0, 0.0, min_value, min_value]]]], requires_grad=True
        )
        losses = loss_fn(
            alive_history=torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]]),
            score_history=scores,
            target_labels=torch.tensor([[[True, False, False, False]]]),
            proposal_mask=torch.tensor([[True, True, False, False]]),
            round_mask=torch.tensor([[[True]]]),
        )

        self.assertTrue(
            all(torch.isfinite(value) for value in losses.values())
        )
        losses["online_trajectory_loss"].backward()
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_aabb_iou_is_finite_for_nonfinite_and_reversed_bounds(self):
        first = torch.tensor(
            [
                [[float("nan"), 1.0, 1.0], [2.0, 2.0, 2.0]],
                [[2.0, 2.0, 2.0], [0.0, 0.0, 0.0]],
            ]
        )
        second = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            ]
        )

        ious = get_batch_aabb_pair_ious(first, second)

        self.assertTrue(torch.isfinite(ious).all())
        self.assertTrue(((ious >= 0.0) & (ious <= 1.0)).all())

    def test_hungarian_matching_ignores_invalid_proposals(self):
        loss_fn = RefBCELoss(
            iou_threshold=0.5,
            matching_strategy="hungarian",
            chunk_size=1,
            max_num_proposals=2,
        )
        output = {
            "proposal_masks_dense": torch.tensor([[True, False]]),
        }
        predicted_boxes = torch.tensor(
            [
                [
                    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                    [[float("nan"), 0.0, 0.0], [1.0, 1.0, 1.0]],
                ]
            ]
        )
        predicted_scores = torch.tensor([[2.0, 2.0]])
        target_boxes = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]]
        )

        loss = loss_fn(
            output,
            predicted_boxes,
            predicted_scores,
            target_boxes,
            torch.tensor([[True]]),
            torch.tensor([0, 1]),
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(output["gt_labels"].tolist(), [[[1.0, 0.0]]])

    def test_relation_features_avoid_size_ratio_overflow(self):
        boxes = torch.tensor(
            [
                [
                    [[-5e19, -5e19, -5e19], [5e19, 5e19, 5e19]],
                    [[-5e-20, -5e-20, -5e-20], [5e-20, 5e-20, 5e-20]],
                ]
            ]
        )

        features = TrajectoryConditionedGroundingAdapter._relation_features(boxes)

        self.assertTrue(torch.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
