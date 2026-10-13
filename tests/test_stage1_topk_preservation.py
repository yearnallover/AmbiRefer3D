import copy
import unittest

from scripts.plot_stage1_topk_preservation import ranked_predictions, summarize


def example():
    return {
        "scene_id": "scene", "ann_id": 1,
        "proposal_valid": [True, True, True, False],
        "proposal_scores": [0.7, 0.8, 0.9, 1.0],
        "proposal_boxes": [
            [[0, 0, 0], [1, 1, 1]],
            [[2, 2, 2], [3, 3, 3]],
            [[4, 4, 4], [5, 5, 5]],
            [[0, 0, 0], [1, 1, 1]],
        ],
        "proposal_object_ids": [1, 2, 3, 1],
        "proposal_object_iou": [1.0, 1.0, 1.0, 1.0],
        "annotated_candidate_ids": [1, 2],
        "target_gt_box": [[0, 0, 0], [1, 1, 1]],
    }


class TopKPreservationTests(unittest.TestCase):
    def test_valid_score_order_and_independent_target_iou(self):
        scores, target, matching = ranked_predictions(example())
        self.assertEqual(scores.tolist(), [0.9, 0.8, 0.7])
        self.assertEqual(target.tolist(), [False, False, True])
        self.assertEqual(matching.tolist(), [False, True, True])

    def test_short_sets_and_threshold_differ_from_topk(self):
        rows, audit = summarize([example()], [1, 2, 3, 5], 0.75)
        self.assertEqual([r["true_target_at_k_percent"] for r in rows], [0, 0, 100, 100])
        self.assertEqual(rows[1]["query_matching_share_at_k_percent"], 50)
        self.assertAlmostEqual(rows[3]["query_matching_share_at_k_percent"], 200 / 3)
        self.assertEqual(rows[3]["queries_with_fewer_than_k_proposals"], 1)
        self.assertEqual(audit["true_target_ts_at_threshold_percent"], 0)
        self.assertEqual(audit["raw_true_target_ts_percent"], 100)

    def test_empty_queries_stay_in_denominator(self):
        empty = copy.deepcopy(example())
        empty["proposal_valid"] = [False] * 4
        rows, audit = summarize([example(), empty], [3], 0.01)
        self.assertEqual(rows[0]["num_queries"], 2)
        self.assertEqual(rows[0]["true_target_at_k_percent"], 50)
        self.assertAlmostEqual(rows[0]["query_matching_share_at_k_percent"], 100 / 3)
        self.assertEqual(audit["num_empty_proposal_queries"], 1)

    def test_missing_gt_and_invalid_scores_rejected(self):
        row = example()
        row["target_gt_box"] = None
        with self.assertRaisesRegex(ValueError, "missing target"):
            ranked_predictions(row)
        row = example()
        row["proposal_scores"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite sigmoid"):
            ranked_predictions(row)


if __name__ == "__main__":
    unittest.main()
