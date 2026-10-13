import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from m3drefclip.data.online_trajectory import load_online_trajectories
from m3drefclip.loss.online_trajectory_loss import OnlineTrajectoryLoss
from m3drefclip.model.trajectory_grounding_adapter import (
    TrajectoryConditionedGroundingAdapter,
)
from m3drefclip.model.m3dref_clip import M3DRefCLIP, freeze_batch_norm_running_stats
from stage2_refactor.data.build_online_round_json import build_round_samples


class TrajectoryGroundingAdapterTest(unittest.TestCase):
    def test_candidate_labels_keep_one_matched_proposal_per_object(self):
        candidate_labels = torch.tensor(
            [[[[True, True, True, False], [True, False, True, False]]]]
        )
        one_to_one = torch.tensor([[[False, True, True, False]]])
        restricted = M3DRefCLIP._restrict_candidate_labels_to_one_to_one(
            candidate_labels, one_to_one
        )

        self.assertTrue(
            torch.equal(
                restricted,
                torch.tensor(
                    [[[[False, True, True, False], [False, False, True, False]]]]
                ),
            )
        )

    def test_freeze_batch_norm_stats_does_not_disable_dropout(self):
        module = nn.Sequential(nn.BatchNorm1d(4), nn.Dropout(0.5)).train()
        freeze_batch_norm_running_stats((module,))
        self.assertFalse(module[0].training)
        self.assertTrue(module[1].training)

    def test_adapter_shapes_masks_and_gradients(self):
        torch.manual_seed(7)
        batch_size, candidates = 4, 6
        adapter = TrajectoryConditionedGroundingAdapter(
            proposal_dim=32, query_dim=128, hidden_dim=64, relation_dim=32, dropout=0.0
        )
        features = torch.randn(batch_size, candidates, 32, requires_grad=True)
        bboxes = torch.sort(torch.randn(batch_size, candidates, 2, 3), dim=2).values
        logits = torch.randn(batch_size, candidates, requires_grad=True)
        query = torch.randn(batch_size, 128, requires_grad=True)
        mask = torch.ones(batch_size, candidates, dtype=torch.bool)
        mask[:, -1] = False

        output = adapter(features, bboxes, logits, query, mask)
        self.assertEqual(output["proposal_features"].shape, features.shape)
        self.assertEqual(output["logits"].shape, logits.shape)
        self.assertEqual(output["belief"].shape, logits.shape)
        self.assertEqual(output["state"].shape, (batch_size, 64))
        self.assertEqual(output["target_summary"].shape, (batch_size, 64))
        self.assertTrue(torch.equal(output["proposal_features"][:, -1], features[:, -1]))
        self.assertTrue(torch.all(output["belief"][:, -1] == 0))
        self.assertLessEqual(float(output["residual_scale"]), 1.0)

        loss = output["proposal_features"].square().mean() + output["logits"].square().mean()
        loss.backward()
        self.assertGreater(float(features.grad.abs().sum()), 0)
        self.assertGreater(float(logits.grad.abs().sum()), 0)
        self.assertGreater(float(query.grad.abs().sum()), 0)
        adapter_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in adapter.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(adapter_grad, 0)
    def test_feedback_changes_trajectory_state(self):
        torch.manual_seed(11)
        adapter = TrajectoryConditionedGroundingAdapter(
            proposal_dim=8, query_dim=12, hidden_dim=16, relation_dim=8, dropout=0.0
        ).eval()
        features = torch.randn(2, 5, 8)
        bboxes = torch.sort(torch.randn(2, 5, 2, 3), dim=2).values
        logits = torch.randn(2, 5)
        query = torch.randn(2, 12)
        mask = torch.ones(2, 5, dtype=torch.bool)
        positive = adapter(features, bboxes, logits, query, mask, feedback=torch.ones(2))
        negative = adapter(features, bboxes, logits, query, mask, feedback=-torch.ones(2))
        self.assertFalse(torch.allclose(positive["state"], negative["state"]))
        self.assertFalse(torch.allclose(positive["logits"], negative["logits"]))
    def test_trajectory_loss_is_finite_and_updates_both_trajectories(self):
        torch.manual_seed(13)
        batch, language, rounds, candidates = 2, 2, 3, 7
        alive = torch.sigmoid(torch.randn(batch, language, rounds, candidates))
        positive = torch.randn(batch, language, rounds, candidates, requires_grad=True)
        negative = torch.randn(batch, language, rounds, candidates, requires_grad=True)
        raw = torch.randn(batch, language, rounds, candidates, requires_grad=True)
        target = torch.zeros(batch, language, candidates, dtype=torch.bool)
        target[..., 0] = True
        compatible = target.clone()
        compatible[..., 1] = True
        proposal_mask = torch.ones(batch, candidates, dtype=torch.bool)
        round_mask = torch.ones(batch, language, rounds, dtype=torch.bool)
        feedback = torch.zeros(batch, language, rounds)
        feedback[:, :, 1:] = 1.0
        criterion = OnlineTrajectoryLoss(
            coverage_weight=1.0,
            transition_weight=0.5,
            rank_improvement_weight=0.5,
            dpo_weight=0.5,
            kl_weight=0.01,
        )

        losses = criterion(
            alive,
            positive,
            target,
            proposal_mask,
            round_mask,
            raw_score_history=raw,
            compatible_labels=compatible,
            feedback=feedback,
            negative_score_history=negative,
        )
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        losses["online_trajectory_loss"].backward()
        self.assertGreater(float(positive.grad.abs().sum()), 0)
        self.assertGreater(float(negative.grad.abs().sum()), 0)
        self.assertIsNone(raw.grad)

    def test_independent_pair_loss_supervises_every_round_without_temporal_terms(self):
        scores = torch.tensor(
            [[[[0.0, 2.0], [2.0, 0.0]]]], requires_grad=True
        )
        alive = torch.tensor([[[[1.0, 0.9], [1.0, 0.1]]]])
        target = torch.tensor([[[True, False]]])
        common = {
            "alive_history": alive,
            "score_history": scores,
            "target_labels": target,
            "proposal_mask": torch.ones(1, 2, dtype=torch.bool),
            "round_mask": torch.ones(1, 1, 2, dtype=torch.bool),
            "feedback": torch.tensor([[[0.0, 1.0]]]),
            "negative_score_history": -scores,
        }
        ordered = OnlineTrajectoryLoss(
            survival_weight=0.0,
            prune_weight=1.0,
            ranking_weight=1.0,
            transition_weight=1.0,
            rank_improvement_weight=1.0,
            dpo_weight=1.0,
        )(**common)
        independent = OnlineTrajectoryLoss(
            survival_weight=0.0,
            prune_weight=1.0,
            ranking_weight=1.0,
            transition_weight=1.0,
            rank_improvement_weight=1.0,
            dpo_weight=1.0,
            independent_rounds=True,
        )(**common)

        self.assertGreater(
            float(independent["online_trajectory_prune_loss"]),
            float(ordered["online_trajectory_prune_loss"]),
        )
        self.assertGreater(
            float(independent["online_trajectory_ranking_loss"]),
            float(ordered["online_trajectory_ranking_loss"]),
        )
        self.assertEqual(float(independent["online_trajectory_transition_loss"]), 0.0)
        self.assertEqual(float(independent["online_trajectory_rank_improvement_loss"]), 0.0)
        self.assertEqual(float(independent["online_trajectory_dpo_loss"]), 0.0)

    def test_balanced_coverage_pushes_compatible_logits_up(self):
        positive = torch.zeros(1, 1, 2, 5, requires_grad=True)
        alive = torch.sigmoid(positive)
        target = torch.tensor([[[True, False, False, False, False]]])
        compatible = torch.tensor([[[True, True, False, False, False]]])
        criterion = OnlineTrajectoryLoss(
            survival_weight=0.0,
            prune_weight=0.0,
            ranking_weight=0.0,
            coverage_weight=1.0,
            initial_survival_weight=0.0,
            coverage_positive_weight=2.0,
            coverage_negative_weight=0.25,
        )
        losses = criterion(
            alive,
            positive,
            target,
            torch.ones(1, 5, dtype=torch.bool),
            torch.ones(1, 1, 2, dtype=torch.bool),
            compatible_labels=compatible,
        )
        losses["online_trajectory_loss"].backward()
        self.assertTrue(torch.all(positive.grad[0, 0, 0, :2] < 0))
        self.assertTrue(torch.all(positive.grad[0, 0, 0, 2:] > 0))

    def test_candidate_set_supervision_preserves_then_prunes_distractor(self):
        scores = torch.zeros(1, 1, 2, 3, requires_grad=True)
        target = torch.tensor([[[True, False, False]]])
        candidate_sets = torch.tensor(
            [[[[True, True, False], [True, False, False]]]]
        )
        criterion = OnlineTrajectoryLoss(
            survival_weight=0.0,
            prune_weight=0.0,
            ranking_weight=0.0,
            coverage_weight=0.0,
            candidate_set_weight=1.0,
            initial_survival_weight=0.0,
            transition_weight=0.0,
            rank_improvement_weight=0.0,
            dpo_weight=0.0,
            kl_weight=0.0,
        )
        losses = criterion(
            torch.sigmoid(scores),
            scores,
            target,
            torch.ones(1, 3, dtype=torch.bool),
            torch.ones(1, 1, 2, dtype=torch.bool),
            candidate_labels=candidate_sets,
        )
        losses["online_trajectory_loss"].backward()
        self.assertLess(float(scores.grad[0, 0, 0, 1]), 0.0)
        self.assertGreater(float(scores.grad[0, 0, 1, 1]), 0.0)

    def test_candidate_set_loss_uses_deployment_threshold(self):
        threshold = 0.01
        boundary = torch.logit(torch.tensor(threshold))
        scores = torch.tensor(
            [[[[boundary + 0.2, boundary - 0.2]]]], requires_grad=True
        )
        target = torch.tensor([[[True, False]]])
        candidate_sets = target.unsqueeze(2)
        criterion = OnlineTrajectoryLoss(
            survival_weight=0.0,
            prune_weight=0.0,
            ranking_weight=0.0,
            candidate_set_weight=1.0,
            decision_threshold=threshold,
        )

        loss = criterion(
            torch.sigmoid(scores),
            scores,
            target,
            torch.ones(1, 2, dtype=torch.bool),
            torch.ones(1, 1, 1, dtype=torch.bool),
            candidate_labels=candidate_sets,
        )["online_trajectory_loss"]
        loss.backward()

        self.assertLess(float(scores.grad[0, 0, 0, 0]), 0.0)
        self.assertGreater(float(scores.grad[0, 0, 0, 1]), 0.0)

    def test_raw_candidate_set_supervision_reaches_stage1_logits(self):
        refined = torch.zeros(1, 1, 2, 3, requires_grad=True)
        raw = torch.zeros(1, 1, 2, 3, requires_grad=True)
        target = torch.tensor([[[True, False, False]]])
        candidate_sets = torch.tensor(
            [[[[True, True, False], [True, False, False]]]]
        )
        criterion = OnlineTrajectoryLoss(
            survival_weight=0.0,
            prune_weight=0.0,
            ranking_weight=0.0,
            coverage_weight=0.0,
            candidate_set_weight=0.0,
            raw_candidate_set_weight=1.0,
            initial_survival_weight=0.0,
            transition_weight=0.0,
            rank_improvement_weight=0.0,
            dpo_weight=0.0,
            kl_weight=0.0,
        )
        losses = criterion(
            torch.sigmoid(refined),
            refined,
            target,
            torch.ones(1, 3, dtype=torch.bool),
            torch.ones(1, 1, 2, dtype=torch.bool),
            raw_score_history=raw,
            candidate_labels=candidate_sets,
        )
        losses["online_trajectory_loss"].backward()
        self.assertLess(float(raw.grad[0, 0, 0, 1]), 0.0)
        self.assertGreater(float(raw.grad[0, 0, 1, 1]), 0.0)

    def test_adapter_residual_is_bounded(self):
        adapter = TrajectoryConditionedGroundingAdapter(
            proposal_dim=4,
            query_dim=6,
            hidden_dim=8,
            relation_dim=4,
            dropout=0.0,
            initial_residual_scale=0.02,
            max_residual_scale=0.1,
            max_logit_residual=2.0,
        )
        with torch.no_grad():
            adapter.logit_residual[-1].weight.fill_(1000.0)
        features = torch.randn(2, 3, 4)
        bboxes = torch.sort(torch.randn(2, 3, 2, 3), dim=2).values
        logits = torch.randn(2, 3)
        output = adapter(
            features,
            bboxes,
            logits,
            torch.randn(2, 6),
            torch.ones(2, 3, dtype=torch.bool),
        )
        delta = (output["logits"] - logits).abs().max()
        self.assertLessEqual(float(delta), 0.2 + 1e-6)

    def test_initial_round_suppresses_negative_logit_residual(self):
        adapter = TrajectoryConditionedGroundingAdapter(
            proposal_dim=4,
            query_dim=6,
            hidden_dim=8,
            relation_dim=4,
            dropout=0.0,
            initial_prune_gate=0.05,
        ).eval()
        with torch.no_grad():
            adapter.logit_residual[-1].weight.zero_()
            adapter.logit_residual[-1].bias.fill_(-1000.0)
        common = (
            torch.randn(1, 3, 4),
            torch.sort(torch.randn(1, 3, 2, 3), dim=2).values,
            torch.zeros(1, 3),
            torch.randn(1, 6),
            torch.ones(1, 3, dtype=torch.bool),
        )
        initial = adapter(*common, feedback=torch.zeros(1))
        clarified = adapter(*common, feedback=torch.ones(1))
        self.assertTrue(torch.all(initial["prune_gate"] == 0.05))
        self.assertLess(
            float(initial["logit_residual"].abs().mean()),
            float(clarified["logit_residual"].abs().mean()),
        )
    def test_disabled_risk_gate_restores_signed_residual_and_gradients(self):
        adapter = TrajectoryConditionedGroundingAdapter(
            proposal_dim=4, query_dim=6, hidden_dim=8, relation_dim=4,
            dropout=0.0, risk_gate_enabled=False,
        ).eval()
        with torch.no_grad():
            adapter.logit_residual[-1].weight.zero_()
            adapter.logit_residual[-1].bias.fill_(-0.5)
        inputs = (
            torch.randn(1, 3, 4),
            torch.sort(torch.randn(1, 3, 2, 3), dim=2).values,
            torch.zeros(1, 3),
            torch.randn(1, 6),
            torch.ones(1, 3, dtype=torch.bool),
        )
        output = adapter(*inputs, feedback=torch.zeros(1))
        expected = adapter.max_logit_residual * torch.tanh(torch.tensor(-0.5))
        self.assertTrue(torch.allclose(output["logit_residual"], expected.expand(1, 3)))
        self.assertTrue(torch.all(output["prune_gate"] == 1))
        output["logits"].sum().backward()
        self.assertGreater(float(adapter.logit_residual[-1].bias.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in adapter.prune_gate.parameters()))

    def test_nested_dialogue_builds_recomputed_negative_query(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "trajectory.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "scene_id": "scene0000_00",
                        "ann_id": 1,
                        "target_id": 5,
                        "dialogues": [
                            {
                                "ambiguity_type": "minimal",
                                "input_query": "the chair",
                                "restored_query": "the chair near the table",
                                "generated_question": "Is it near the table?",
                            }
                        ],
                    }
                ]
            ),
            encoding="utf-8",
        )
        trajectories = load_online_trajectories(path, max_rounds=3, ambiguity_type="minimal")
        trajectory = trajectories[("scene0000_00", "1", 5)][0]
        self.assertEqual(trajectory["queries"][1], "the chair near the table")
        self.assertNotEqual(trajectory["negative_queries"][1], trajectory["queries"][1])
        self.assertEqual(trajectory["feedback"], [0.0, 1.0])
        self.assertEqual(trajectory["negative_feedback"], [0.0, -1.0])

    def test_compositional_stage1_and_stage2_trajectories_are_aligned(self):
        item = {
            "scene_id": "scene0000_00",
            "ann_id": 3,
            "target_id": 5,
            "dialogues": [
                {
                    "ambiguity_type": "minimal",
                    "is_ambiguous": True,
                    "input_query": "the chair",
                    "restored_query": "the red chair near the table",
                    "generated_question": "Is it red and near the table?",
                    "grounding_matches": [5, 6, 7],
                },
                {
                    "ambiguity_type": "no_spatial",
                    "input_query": "the red chair",
                    "grounding_matches": [5, 6],
                },
                {
                    "ambiguity_type": "no_visual",
                    "input_query": "the chair near the table",
                    "grounding_matches": [5],
                },
            ],
        }
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "trajectory.json"
        path.write_text(json.dumps([item]), encoding="utf-8")
        online = load_online_trajectories(
            path, max_rounds=3, ambiguity_type="minimal"
        )[("scene0000_00", "3", 5)]
        stage2_rows = build_round_samples([item], ambiguity_type="minimal")
        stage2_by_id = {}
        for row in stage2_rows:
            stage2_by_id.setdefault(row["trajectory_id"], []).append(row)

        self.assertEqual(
            {trajectory["trajectory_id"] for trajectory in online},
            set(stage2_by_id),
        )
        for trajectory in online:
            rows = sorted(
                stage2_by_id[trajectory["trajectory_id"]],
                key=lambda row: row["round"],
            )
            self.assertEqual(len(rows), len(trajectory["action_queries"]))
            self.assertEqual(
                [row["cand_ids"] for row in rows],
                trajectory["candidate_sets"][:-1],
            )
            self.assertEqual(
                [row["negative_query"] for row in rows],
                trajectory["negative_queries"][1:],
            )
            for action_query, negative_query in zip(
                trajectory["action_queries"],
                trajectory["negative_queries"][1:],
            ):
                self.assertIn(action_query, negative_query)
            self.assertTrue(all(5 in row["cand_ids"] for row in rows))
            self.assertEqual(rows[-1]["stop_y"], 1)
            constraint_keys = [
                json.dumps(row["constraint_gt"], sort_keys=True) for row in rows
            ]
            self.assertEqual(len(constraint_keys), len(set(constraint_keys)))


if __name__ == "__main__":
    unittest.main()
