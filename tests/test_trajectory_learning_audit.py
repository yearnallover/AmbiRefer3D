import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from m3drefclip.loss.online_trajectory_loss import OnlineTrajectoryLoss
from m3drefclip.model.stage2_joint import Stage2JointModule
from m3drefclip.model.stage1_backbones.external import BUTDDETRStage1Backbone, _freeze
from tools.select_best_epoch_checkpoint import select_best
from stage2_refactor.config import TrainConfig
from stage2_refactor.learning.losses_semantic import candidate_filter_loss
from stage2_refactor.model.encoders_semantic import ImageCandidateEncoder
from stage2_refactor.model.stage2_net_semantic import Stage2NetSemantic


class TrajectoryLearningAuditTest(unittest.TestCase):
    def test_frozen_native_butd_stays_frozen_and_in_eval_mode(self):
        native = torch.nn.Module()
        native.text_encoder = torch.nn.Linear(2, 2)
        native.decoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2), torch.nn.Dropout(0.5), torch.nn.Linear(2, 2)
        )
        wrapper = BUTDDETRStage1Backbone.__new__(BUTDDETRStage1Backbone)
        torch.nn.Module.__init__(wrapper)
        wrapper.backbone = native
        wrapper.span_model = torch.nn.Linear(2, 2)
        wrapper.visual_projection = torch.nn.Linear(2, 2)
        wrapper.freeze_native_backbone = True
        _freeze(wrapper.backbone)

        wrapper.train(True)

        self.assertFalse(wrapper.backbone.training)
        self.assertFalse(wrapper.span_model.training)
        self.assertTrue(wrapper.visual_projection.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in wrapper.backbone.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in wrapper.visual_projection.parameters()))
    @staticmethod
    def _online_losses(scores, detach=False, **kwargs):
        return OnlineTrajectoryLoss(detach_temporal_baseline=detach)(
            alive_history=scores.sigmoid(),
            score_history=scores,
            target_labels=torch.tensor([[[True, False, False]]]),
            proposal_mask=torch.tensor([[True, True, False]]),
            round_mask=torch.ones(1, 1, scores.shape[2], dtype=torch.bool),
            feedback=torch.tensor([[[0.0, 1.0]]]),
            **kwargs,
        )

    def test_temporal_losses_cannot_improve_by_degrading_initial_scores(self):
        for key in (
            "online_trajectory_rank_improvement_loss",
            "online_trajectory_transition_loss",
        ):
            with self.subTest(loss=key):
                scores = torch.tensor(
                    [[[[0.0, 2.0, 0.0], [2.0, 0.0, 0.0]]]], requires_grad=True
                )
                legacy = self._online_losses(scores)[key]
                fixed = self._online_losses(scores, detach=True)[key]
                legacy_grad = torch.autograd.grad(legacy, scores)[0]
                fixed_grad = torch.autograd.grad(fixed, scores)[0]
                torch.testing.assert_close(legacy, fixed)
                self.assertGreater(legacy_grad[0, 0, 0, 0].item(), 0)
                torch.testing.assert_close(fixed_grad[:, :, 0], torch.zeros(1, 1, 3))
                self.assertLess(fixed_grad[0, 0, 1, 0].item(), 0)
                torch.testing.assert_close(fixed_grad[..., 2], torch.zeros(1, 1, 2))

    def test_detached_comparator_preserves_recurrent_credit_assignment(self):
        initial = torch.tensor([[[0.0, 2.0, 0.0]]], requires_grad=True)
        final = initial + torch.tensor([[[2.0, -2.0, 0.0]]])
        scores = torch.stack((initial, final), dim=2)
        loss = self._online_losses(scores, detach=True)[
            "online_trajectory_rank_improvement_loss"
        ]
        loss.backward()
        self.assertLess(initial.grad[0, 0, 0].item(), 0)
        self.assertTrue(torch.isfinite(initial.grad).all())

    def test_online_loss_does_not_mutate_candidate_labels(self):
        scores = torch.zeros(1, 1, 2, 3, requires_grad=True)
        labels = torch.ones_like(scores, dtype=torch.bool)
        original = labels.clone()
        losses = self._online_losses(scores, candidate_labels=labels)
        losses["online_trajectory_candidate_set_loss"].backward()
        self.assertTrue(torch.equal(labels, original))
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_filter_loss_does_not_mutate_target_labels(self):
        labels = torch.tensor([[True, True]])
        logits = torch.zeros(1, 2, requires_grad=True)
        original = labels.clone()
        candidate_filter_loss(
            logits, torch.tensor([0]), torch.tensor([[True, False]]),
            target_mask=labels,
        ).backward()
        self.assertTrue(torch.equal(labels, original))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_geometry_gradients_finite_for_degenerate_candidate_sets(self):
        for count, valid_count in ((1, 1), (3, 1), (3, 3), (3, 0)):
            with self.subTest(count=count, valid_count=valid_count):
                boxes = torch.tensor([[[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]])
                boxes = boxes.repeat(1, count, 1).requires_grad_()
                mask = torch.zeros(1, count)
                mask[:, :valid_count] = 1
                features = ImageCandidateEncoder._bbox_features(boxes, mask)
                features.sum().backward()
                self.assertTrue(torch.isfinite(features).all())
                self.assertTrue(torch.isfinite(boxes.grad).all())

    @staticmethod
    def _stage2(pre_action=True):
        torch.manual_seed(123)
        cfg = TrainConfig(
            d_text=16, d_cand=16, d_constraint=16, d_state=16, imgfeat_dim=8,
            use_bbox_geom=True, use_stage1_features=True, stage1_feat_dim=8,
            use_scene_context_attention=True, use_constraint_conditioned_context=True,
            selector_pre_action_state=pre_action,
        )
        net = Stage2NetSemantic(32, cfg, 4).eval()
        batch = {
            "query_ids": torch.tensor([[1, 2, 3]]),
            "c_ctype_id": torch.tensor([0]), "c_rel_id": torch.tensor([0]),
            "c_attr_id": torch.tensor([0]), "field_ids": torch.tensor([[4, 5]]),
            "cand_mask": torch.ones(1, 3), "img": torch.zeros(1, 3, 8),
            "cand_bboxes": torch.randn(1, 3, 6),
            "cand_stage1_features": torch.randn(1, 3, 8, requires_grad=True),
            "cand_belief": torch.full((1, 3), 1 / 3),
        }
        return net, batch

    def test_selector_invariant_to_chosen_action_but_filter_is_conditioned(self):
        net, batch = self._stage2()
        changed = dict(batch, field_ids=torch.tensor([[6, 7]]), c_rel_id=torch.tensor([1]))
        first, second = net(batch), net(changed)
        torch.testing.assert_close(first["selector_logits"], second["selector_logits"], rtol=0, atol=0)
        self.assertGreater((first["mask_logits"] - second["mask_logits"]).abs().max().item(), 1e-5)

    def test_legacy_mode_reproduces_action_dependent_selector(self):
        net, batch = self._stage2(pre_action=False)
        changed = dict(batch, field_ids=torch.tensor([[6, 7]]), c_rel_id=torch.tensor([1]))
        difference = net(batch)["selector_logits"] - net(changed)["selector_logits"]
        self.assertGreater(difference.abs().max().item(), 1e-5)

    def test_selector_gradient_reaches_stage1_input_not_chosen_action_encoder(self):
        net, batch = self._stage2()
        F.cross_entropy(net(batch)["selector_logits"], torch.tensor([0])).backward()
        gradient = batch["cand_stage1_features"].grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in net.cenc.parameters()))

    def test_filter_gradient_still_reaches_chosen_action_encoder(self):
        net, batch = self._stage2()
        F.cross_entropy(net(batch)["mask_logits"], torch.tensor([0])).backward()
        gradients = [p.grad for p in net.cenc.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
        self.assertGreater(batch["cand_stage1_features"].grad.abs().sum().item(), 0)

    def test_fix_does_not_change_checkpoint_keys(self):
        legacy, _ = self._stage2(pre_action=False)
        fixed, _ = self._stage2(pre_action=True)
        result = fixed.load_state_dict(legacy.state_dict(), strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])

    def test_shared_stage1_logprob_cancels_in_pairwise_dpo(self):
        module = Stage2JointModule.__new__(Stage2JointModule)
        torch.nn.Module.__init__(module)
        module.trajectory_grounding_logprob_weight = 1.0
        module.trajectory_dpo_beta = 0.1
        scores = torch.tensor([[0.2, -0.1, 0.4]], requires_grad=True)
        batch = {
            "constraint_id": torch.tensor([0]), "target_index": torch.tensor([1]),
            "target_candidate_mask": torch.tensor([[False, True, False]]),
            "cand_stage1_scores": scores, "cand_mask": torch.ones(1, 3),
            "stop_y": torch.zeros(1),
        }
        out = {
            "selector_logits": torch.tensor([[0.3, -0.2]], requires_grad=True),
            "mask_logits": torch.zeros(1, 3, requires_grad=True),
            "stop_logit": torch.zeros(1, requires_grad=True),
        }
        lengths = torch.tensor([1])
        positive = module._trajectory_logprob(None, batch, out, lengths)
        negative = module._trajectory_logprob(
            None, batch, out, lengths, constraint_ids=torch.tensor([1])
        )
        module._dpo_pair_loss(
            positive, negative, positive.detach(), negative.detach()
        ).mean().backward()
        torch.testing.assert_close(scores.grad, torch.zeros_like(scores))
        self.assertGreater(out["selector_logits"].grad.abs().sum().item(), 0)


class AuditLauncherTest(unittest.TestCase):
    def test_select_best_epoch_checkpoint_uses_joint_score(self):
        with tempfile.TemporaryDirectory() as directory:
            training_dir = Path(directory)
            (training_dir / "epoch_metrics.log").write_text(
                "[VAL-MT][epoch 005] F1@IoU0.50=70.00% Joint=62.50%\n"
                "[VAL-MT][epoch 010] F1@IoU0.50=69.00% Joint=64.25%\n"
                "[VAL-MT][epoch 015] F1@IoU0.50=75.00% Joint=63.10%\n",
                encoding="utf-8",
            )
            expected = training_dir / "epoch=9.ckpt"
            expected.touch()
            checkpoint, epoch, score = select_best(training_dir)
            self.assertEqual(checkpoint, expected.resolve())
            self.assertEqual(epoch, 10)
            self.assertEqual(score, 64.25)

    def test_dry_run_variants_preserve_user_overrides_and_do_not_launch_training(self):
        root = Path(__file__).resolve().parents[1]
        for variant, selector, temporal in (
            ("legacy", "False", "False"), ("selector_only", "True", "False"),
            ("temporal_only", "False", "True"), ("fixed", "True", "True"),
        ):
            with self.subTest(variant=variant):
                result = subprocess.run(
                    ["bash", str(root / "scripts/run_trajectory_audit_control.sh"),
                     variant, "m3drefclip", "ordered_trajectory", "train", "fresh",
                     "trainer.limit_train_batches=1"],
                    env={**os.environ, "ROOT": str(root), "DRY_RUN": "1"},
                    check=True, capture_output=True, text=True,
                )
                self.assertIn(f"model.stage2.selector_pre_action_state={selector}", result.stdout)
                self.assertIn(f"detach_temporal_baseline={temporal}", result.stdout)
                self.assertIn("trainer.limit_train_batches=1", result.stdout)

    def test_invalid_variant_fails_before_starting_training(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            ["bash", str(root / "scripts/run_trajectory_audit_control.sh"),
             "invalid", "m3drefclip", "ordered_trajectory", "train"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
