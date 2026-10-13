import unittest

import torch
import torch.nn as nn

from m3drefclip.model.stage2_joint import Stage2JointModule
from stage2_refactor.eval_rollout import (
    eval_rollout_batch,
    risk_sensitive_candidate_update,
)
from validate_multi_ckpts import (
    _accumulate_alive_counts,
    _count_queries_without_target_proposal,
)
from stage2_refactor.learning.losses_semantic import (
    candidate_filter_loss,
    selector_ce_loss,
    stop_bce_loss,
)
from stage2_refactor.model.heads_selector import ConstraintSelectorHead
from stage2_refactor.model.constraint_embed_semantic import SemanticConstraintEmbedder
from stage2_refactor.model.encoders_semantic import QueryEncoder
from stage2_refactor.model.stage2_net_semantic import Stage2NetSemantic
from stage2_refactor.config import TrainConfig
from stage2_refactor.learning.constraint_table import (
    ConstraintTable,
    constraint_field_text,
    constraint_to_key,
)
from tools.analyze_interactive_candidate_utility import _query_metrics
from validate_multi_ckpts import _trajectory_grounding_sums


class TestStage2EvaluationAccounting(unittest.TestCase):
    def test_counts_targetless_active_queries(self):
        query_mask = torch.tensor([[True, True, False]])
        target_labels = torch.tensor(
            [[[False, False], [False, True], [False, False]]]
        )

        self.assertEqual(
            _count_queries_without_target_proposal(query_mask, target_labels),
            1,
        )

    def test_rejects_target_labels_without_proposal_dimension(self):
        with self.assertRaisesRegex(ValueError, "proposal dimension"):
            _count_queries_without_target_proposal(
                torch.ones((1, 2), dtype=torch.bool),
                torch.ones((1, 2), dtype=torch.bool),
            )


class _DummyStage2Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        marker = batch["marker"].float() * self.scale
        zeros = torch.zeros_like(marker)
        return {
            "selector_logits": torch.stack((marker, -marker), dim=1),
            "mask_logits": torch.stack((marker, zeros, -marker), dim=1),
            "stop_logit": -marker,
        }


class _RoundAwareNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, batch):
        used = batch["used_constraint_mask"].sum(dim=1)
        first = torch.tensor([1.0, 10.0], device=used.device)
        second = torch.tensor([9.0, 0.0], device=used.device)
        mask_logits = torch.where((used <= 1).unsqueeze(1), first, second)
        selector_logits = torch.tensor(
            [[2.0, 1.0]], device=used.device
        ).expand(used.shape[0], -1)
        selector_logits = selector_logits.masked_fill(
            batch["used_constraint_mask"] > 0, -1e9
        )
        return {
            "selector_logits": selector_logits + self.anchor * 0.0,
            "mask_logits": mask_logits + self.anchor * 0.0,
            "stop_logit": torch.full_like(used, -10.0) + self.anchor * 0.0,
            "cand_feat": torch.zeros(used.shape[0], 2, 4, device=used.device),
        }


class _BeliefRecordingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.beliefs = []
        self.alive_probabilities = []

    def forward(self, batch):
        self.beliefs.append(batch["cand_belief"].detach().cpu().clone())
        self.alive_probabilities.append(
            batch["cand_alive_probs"].detach().cpu().clone()
        )
        size = batch["cand_mask"].shape[0]
        selector_logits = torch.tensor(
            [[2.0, 1.0]], device=batch["cand_mask"].device
        ).expand(size, -1)
        selector_logits = selector_logits.masked_fill(
            batch["used_constraint_mask"] > 0, -1e9
        )
        probabilities = torch.tensor(
            [[0.8, 0.2]], device=batch["cand_mask"].device
        ).expand(size, -1)
        return {
            "selector_logits": selector_logits + self.anchor * 0.0,
            "mask_logits": torch.logit(probabilities) + self.anchor * 0.0,
            "stop_logit": torch.full(
                (size,), -10.0, device=batch["cand_mask"].device
            ) + self.anchor * 0.0,
            "cand_feat": torch.zeros(size, 2, 4, device=batch["cand_mask"].device),
        }


class _TransitionGradientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.round_logits = []

    def forward(self, batch):
        mask_logits = self.scale * batch["cand_belief"]
        mask_logits.retain_grad()
        self.round_logits.append(mask_logits)
        size = mask_logits.shape[0]
        return {
            "selector_logits": torch.zeros(
                size, 2, device=mask_logits.device
            ) + self.scale * 0.0,
            "mask_logits": mask_logits,
            "stop_logit": torch.zeros(size, device=mask_logits.device),
            "cand_feat": torch.zeros(size, 2, 4, device=mask_logits.device),
        }


class _TinyConstraintTable:
    class Row:
        ctype_id = 0
        rel_id = 0
        attr_id = 0
        field_text = "item"

    def num_constraints(self):
        return 2

    def row(self, _):
        return self.Row()


class _OneConstraintTable(_TinyConstraintTable):
    def num_constraints(self):
        return 1


class _TinyVocab:
    def encode(self, _text, max_len):
        return [1] + [0] * (max_len - 1)


class _ActionConstraintTable:
    @staticmethod
    def id_from_constraint_gt(constraint):
        return int(constraint["id"])

    @staticmethod
    def apply_id_to_batch(batch, cid, vocab, max_len):
        batch["c_ctype_id"] = cid.clone()
        batch["c_rel_id"] = cid.clone()
        batch["c_attr_id"] = cid.clone()
        batch["field_ids"] = cid.unsqueeze(1).expand(-1, max_len).clone()
        return batch


class _StopStatsProbe(nn.Module):
    def forward(self, value):
        return value[:, -4:].sum(dim=1)


class _SelectorTargetProbe(nn.Module):
    def forward(
        self,
        state,
        txt,
        cand_feat,
        alive_mask,
        target_summary=None,
        used_mask=None,
        constraint_bank=None,
    ):
        self.target_summary = target_summary.detach().clone()
        return state.new_zeros((state.shape[0], 2))


class _DpoModeProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.rnn = nn.GRU(2, 2, batch_first=True)

    def forward(self, batch):
        self.dropout_training_during_forward = self.dropout.training
        self.rnn_training_during_forward = self.rnn.training
        value, _ = self.rnn(batch["value"])
        return self.dropout(value)


class TrajectoryMetricTest(unittest.TestCase):
    def test_cardinality_makes_repeated_semantic_constraints_distinct(self):
        base = {"ctype": "REL", "rel": "closest", "landmark_text": "table"}
        keep_two = {**base, "selection_mode": "top_k", "keep_count": 2}
        keep_one = {**base, "selection_mode": "top_k", "keep_count": 1}

        self.assertEqual(constraint_to_key(base), "REL|closest|table")
        self.assertNotEqual(constraint_to_key(keep_two), constraint_to_key(keep_one))
        self.assertIn("keep 2 candidates", constraint_field_text(keep_two))
        table = ConstraintTable.build_from_samples(
            [{"constraint_gt": keep_two}, {"constraint_gt": keep_one}],
            rel2id={"closest": 0},
            attr2id={"other": 0},
        )
        self.assertEqual(table.num_constraints(), 2)

    def test_independent_pair_mode_only_flattens_training(self):
        module = Stage2JointModule.__new__(Stage2JointModule)
        nn.Module.__init__(module)
        module.supervision_mode = "shuffled_pairs"

        module.train()
        self.assertTrue(module._uses_independent_pair_training())
        shuffled = module._shuffle_independent_rows([0, 1, 2], seed=7)
        self.assertCountEqual(shuffled, [0, 1, 2])
        self.assertNotEqual(shuffled, [0, 1, 2])
        self.assertEqual(
            shuffled, module._shuffle_independent_rows([0, 1, 2], seed=7)
        )

        module.eval()
        self.assertFalse(module._uses_independent_pair_training())

    def test_stage2_gradient_bridge_preserves_values_and_scales_credit(self):
        value = torch.tensor([1.0, -2.0], requires_grad=True)
        bridged = Stage2JointModule._scale_stage1_gradient(value, 0.2)

        self.assertTrue(torch.equal(bridged.detach(), value.detach()))
        bridged.square().sum().backward()
        self.assertTrue(
            torch.allclose(value.grad, 0.2 * 2.0 * value.detach())
        )

    @staticmethod
    def _stage2_trajectory_batch(marker):
        marker = torch.tensor(marker, dtype=torch.float32)
        num_steps = marker.numel()
        return {
            "marker": marker,
            "constraint_id": torch.zeros(num_steps, dtype=torch.long),
            "target_index": torch.zeros(num_steps, dtype=torch.long),
            "target_candidate_mask": torch.tensor(
                [[True, False, False]] * num_steps
            ),
            "cand_stage1_scores": torch.stack(
                (marker, torch.zeros_like(marker), -marker), dim=1
            ),
            "cand_mask": torch.ones(num_steps, 3, dtype=torch.bool),
            "stop_y": torch.tensor([0.0] * (num_steps - 1) + [1.0]),
        }

    @staticmethod
    def _stage2_dpo_module(require_explicit=True):
        module = Stage2JointModule.__new__(Stage2JointModule)
        nn.Module.__init__(module)
        module.net = _DummyStage2Net()
        module.reference_net = None
        module.num_constraints = 2
        module.trajectory_dpo_beta = 0.1
        module.trajectory_grounding_logprob_weight = 1.0
        module.trajectory_require_explicit_negatives = require_explicit
        return module

    def test_raw_metrics_use_fixed_c0(self):
        refined = torch.tensor([[[[4.0, 3.0, 2.0, 1.0], [1.0, 4.0, 3.0, 2.0]]]])
        trajectory_output = {
            "refined_score_history": refined,
            "round_mask": torch.ones(1, 1, 2, dtype=torch.bool),
            "proposal_mask": torch.ones(1, 4, dtype=torch.bool),
        }
        target = torch.tensor([[[False, True, False, False]]])
        compatible = torch.tensor([[[True, True, False, False]]])
        query_mask = torch.ones(1, 1, dtype=torch.bool)
        sums = _trajectory_grounding_sums(
            trajectory_output,
            target,
            compatible,
            query_mask,
            threshold=0.5,
        )
        self.assertEqual(sums["raw_num"], 1)
        self.assertEqual(sums["raw_acc_r0"], 0)
        self.assertEqual(sums["raw_acc_rt"], 1)
        self.assertEqual(sums["raw_target_rank_r0"], 2)
        self.assertEqual(sums["raw_target_rank_rt"], 1)
        self.assertAlmostEqual(sums["compatible_recall@1"], 0.5)
        self.assertEqual(sums["compatible_precision@1"], 1.0)

    def test_single_action_stage2_trajectory_is_retained(self):
        module = Stage2JointModule.__new__(Stage2JointModule)
        samples = [{
            "scene_id": "scene",
            "ann_id": "1",
            "trajectory_id": "one_step",
            "round": 0,
            "cand_ids": [3, 4],
            "target_index": 0,
        }]
        self.assertEqual(module._build_trajectories(samples, False), [[0]])
        self.assertEqual(module._build_trajectories(samples, True), [])

    def test_selector_return_prefers_shorter_successful_trajectory(self):
        module = Stage2JointModule.__new__(Stage2JointModule)
        module.constraint_table = _ActionConstraintTable()
        module.selector_return_discount = 0.9
        module.selector_step_cost = 0.1
        module.selector_information_gain_weight = 0.25
        module.samples = [
            {
                "scene_id": "scene",
                "ann_id": "1",
                "trajectory_id": "short",
                "round": 0,
                "cand_ids": [1, 2, 3],
                "target_index": 0,
                "constraint_gt": {"id": 0},
                "stop_y": 1,
            },
            {
                "scene_id": "scene",
                "ann_id": "1",
                "trajectory_id": "long",
                "round": 0,
                "cand_ids": [1, 2, 3],
                "target_index": 0,
                "constraint_gt": {"id": 1},
                "stop_y": 0,
            },
            {
                "scene_id": "scene",
                "ann_id": "1",
                "trajectory_id": "long",
                "round": 1,
                "cand_ids": [1, 2],
                "target_index": 0,
                "constraint_gt": {"id": 2},
                "stop_y": 1,
            },
        ]
        trajectories = module._build_trajectories(module.samples, False)
        utilities = module._build_selector_action_utilities(trajectories)
        initial_state = ("scene", "1", 1, (1, 2, 3))
        self.assertGreater(
            utilities[initial_state][0], utilities[initial_state][1]
        )

    def test_return_conditioned_negative_uses_lower_utility_action(self):
        rejected, valid = Stage2JointModule._return_conditioned_rejected_action(
            {3: 1.2, 7: 0.98, 9: 1.1},
            current_action=3,
            valid_actions={3, 7, 9},
            min_utility_gap=0.01,
        )
        self.assertTrue(valid)
        self.assertEqual(rejected, 7)

        rejected, valid = Stage2JointModule._return_conditioned_rejected_action(
            {3: 1.2, 7: 1.2},
            current_action=3,
            valid_actions={3, 7},
            min_utility_gap=0.01,
        )
        self.assertFalse(valid)
        self.assertEqual(rejected, 3)

    def test_counterfactual_batch_updates_action_id_and_semantics(self):
        module = Stage2JointModule.__new__(Stage2JointModule)
        module.constraint_table = _ActionConstraintTable()
        module.vocab = _TinyVocab()
        module.max_len = 2
        batch = {
            "constraint_id": torch.tensor([1, 1]),
            "c_ctype_id": torch.tensor([1, 1]),
            "c_rel_id": torch.tensor([1, 1]),
            "c_attr_id": torch.tensor([1, 1]),
            "field_ids": torch.ones(2, 2, dtype=torch.long),
            "alive_mask": torch.ones(2, 3),
            "used_constraint_mask": torch.zeros(2, 4),
        }
        counterfactual = module._make_counterfactual_batch(
            batch, torch.tensor([2, 3])
        )
        self.assertTrue(
            torch.equal(counterfactual["constraint_id"], torch.tensor([2, 3]))
        )
        self.assertTrue(
            torch.equal(counterfactual["c_ctype_id"], torch.tensor([2, 3]))
        )

    def test_deterministic_dpo_keeps_rnn_in_training_mode(self):
        net = _DpoModeProbe().train()
        Stage2JointModule._deterministic_dpo_forward(
            net, {"value": torch.randn(1, 2, 2)}
        )
        self.assertFalse(net.dropout_training_during_forward)
        self.assertTrue(net.rnn_training_during_forward)
        self.assertTrue(net.dropout.training)
        self.assertTrue(net.rnn.training)

    def test_union_candidate_state_keeps_teacher_and_model_false_positives(self):
        teacher = torch.tensor([[True, False, False, True]])
        model_alive = torch.tensor([[0.2, 0.8, 0.6, 0.9]])
        proposal = torch.tensor([[True, True, True, False]])
        merged = Stage2JointModule._merge_candidate_state_masks(
            teacher, model_alive, proposal, mode="union"
        )
        self.assertTrue(
            torch.equal(merged, torch.tensor([[True, True, True, False]]))
        )
        predicted = Stage2JointModule._merge_candidate_state_masks(
            teacher, model_alive, proposal, mode="model"
        )
        self.assertTrue(
            torch.equal(predicted, torch.tensor([[False, True, True, False]]))
        )

    def test_initial_union_then_teacher_preserves_trajectory_dynamics(self):
        teacher = torch.tensor(
            [[True, True, False], [True, False, False]]
        )
        model_alive = torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        )
        proposal = torch.ones(2, 3, dtype=torch.bool)
        state = Stage2JointModule._trajectory_candidate_state_masks(
            teacher,
            model_alive,
            proposal,
            round_indices=torch.tensor([0, 1]),
            mode="initial_union_then_teacher",
        )
        self.assertTrue(torch.equal(state[0], torch.tensor([True, True, True])))
        self.assertTrue(torch.equal(state[1], torch.tensor([True, False, False])))

    def test_terminal_action_filters_to_true_target_not_pre_action_state(self):
        candidate_labels = torch.tensor(
            [[[[True, True, True], [True, True, False]]]]
        )
        target = torch.tensor([True, False, False])
        intermediate = Stage2JointModule._post_action_filter_targets(
            candidate_labels,
            target,
            batch_idx=0,
            lang_idx=0,
            round_idx=0,
            final_round_idx=1,
            terminal=False,
        )
        terminal = Stage2JointModule._post_action_filter_targets(
            candidate_labels,
            target,
            batch_idx=0,
            lang_idx=0,
            round_idx=1,
            final_round_idx=1,
            terminal=True,
        )
        self.assertTrue(torch.equal(intermediate, torch.tensor([True, True, False])))
        self.assertTrue(torch.equal(terminal, target))

    def test_rollout_context_candidates_cannot_be_selected(self):
        model = _RoundAwareNet()
        batch = {
            "cand_mask": torch.ones(1, 2),
            "context_mask": torch.ones(1, 2),
            "alive_mask": torch.tensor([[1.0, 0.0]]),
            "target_index": torch.tensor([0]),
            "target_candidate_mask": torch.tensor([[True, False]]),
            "available_constraint_mask": torch.ones(1, 2),
        }
        pred_idx, _, _, _, _ = eval_rollout_batch(
            model=model,
            batch=batch,
            constraint_table=_TinyConstraintTable(),
            vocab=_TinyVocab(),
            max_len=4,
            max_round=1,
            eval_mode="agent",
            selector_strategy="ours",
            risk_sensitive_pruning=False,
        )
        self.assertEqual(int(pred_idx[0]), 0)

    def test_stage2_trajectory_logprob_updates_stage1_scores(self):
        module = Stage2JointModule.__new__(Stage2JointModule)
        nn.Module.__init__(module)
        module.trajectory_grounding_logprob_weight = 1.0
        stage1_scores = torch.tensor(
            [[0.2, -0.1, 0.4], [0.3, 0.1, -0.2]], requires_grad=True
        )
        batch = {
            "constraint_id": torch.tensor([0, 1]),
            "target_index": torch.tensor([1, 0]),
            "target_candidate_mask": torch.tensor(
                [[False, True, False], [True, False, False]]
            ),
            "cand_stage1_scores": stage1_scores,
            "cand_mask": torch.ones(2, 3),
            "stop_y": torch.zeros(2),
        }
        output = {
            "selector_logits": torch.zeros(2, 2, requires_grad=True),
            "mask_logits": torch.zeros(2, 3, requires_grad=True),
            "stop_logit": torch.zeros(2, requires_grad=True),
        }
        logprob = module._trajectory_logprob(
            None, batch, output, lengths=torch.tensor([2])
        )
        (-logprob.mean()).backward()
        self.assertIsNotNone(stage1_scores.grad)
        self.assertGreater(float(stage1_scores.grad.abs().sum()), 0)

    def test_terminal_trajectory_logprob_updates_target_grounding(self):
        module = Stage2JointModule.__new__(Stage2JointModule)
        nn.Module.__init__(module)
        module.trajectory_grounding_logprob_weight = 1.0
        mask_logits = torch.zeros(1, 3, requires_grad=True)
        batch = {
            "constraint_id": torch.tensor([0]),
            "target_index": torch.tensor([1]),
            "target_candidate_mask": torch.tensor([[False, True, False]]),
            "cand_stage1_scores": torch.tensor([[0.2, 0.1, -0.1]]),
            "cand_mask": torch.ones(1, 3),
            "stop_y": torch.ones(1),
        }
        output = {
            "selector_logits": torch.zeros(1, 2, requires_grad=True),
            "mask_logits": mask_logits,
            "stop_logit": torch.zeros(1, requires_grad=True),
        }
        (-module._trajectory_logprob(
            None, batch, output, lengths=torch.tensor([1])
        ).mean()).backward()
        self.assertLess(float(mask_logits.grad[0, 1]), 0.0)
        self.assertIsNotNone(output["selector_logits"].grad)
        self.assertLess(float(output["selector_logits"].grad[0, 0]), 0.0)

    def test_trajectory_logprob_uses_intermediate_candidate_set_only_before_stop(self):
        module = Stage2JointModule.__new__(Stage2JointModule)
        nn.Module.__init__(module)
        module.trajectory_grounding_logprob_weight = 0.0
        mask_logits = torch.zeros(2, 3, requires_grad=True)
        batch = {
            "constraint_id": torch.tensor([0, 0]),
            "target_index": torch.tensor([0, 0]),
            "target_candidate_mask": torch.tensor(
                [[True, False, False], [True, False, False]]
            ),
            "filter_candidate_mask": torch.tensor(
                [[True, True, False], [True, False, False]]
            ),
            "state_candidate_mask": torch.tensor(
                [[True, True, False], [True, False, False]]
            ),
            "cand_mask": torch.ones(2, 3),
            "stop_y": torch.tensor([0.0, 1.0]),
        }
        output = {
            "selector_logits": torch.zeros(2, 1, requires_grad=True),
            "mask_logits": mask_logits,
            "stop_logit": torch.zeros(2, requires_grad=True),
        }
        (-module._trajectory_logprob(
            None, batch, output, lengths=torch.tensor([2])
        ).mean()).backward()
        self.assertLess(float(mask_logits.grad[0, 1]), 0.0)
        self.assertGreater(float(mask_logits.grad[1, 1]), 0.0)

    def test_stage1_grounding_uses_current_not_future_candidate_set(self):
        module = Stage2JointModule.__new__(Stage2JointModule)
        nn.Module.__init__(module)
        module.trajectory_grounding_logprob_weight = 1.0
        stage1_scores = torch.zeros(2, 3, requires_grad=True)
        batch = {
            "constraint_id": torch.tensor([0, 0]),
            "target_index": torch.tensor([0, 0]),
            "target_candidate_mask": torch.tensor(
                [[True, False, False], [True, False, False]]
            ),
            "filter_candidate_mask": torch.tensor(
                [[True, False, False], [True, False, False]]
            ),
            "state_candidate_mask": torch.tensor(
                [[True, True, False], [True, False, False]]
            ),
            "cand_stage1_scores": stage1_scores,
            "cand_mask": torch.ones(2, 3),
            "stop_y": torch.tensor([0.0, 1.0]),
        }
        output = {
            "selector_logits": torch.zeros(2, 1, requires_grad=True),
            "mask_logits": torch.zeros(2, 3, requires_grad=True),
            "stop_logit": torch.zeros(2, requires_grad=True),
        }
        (-module._trajectory_logprob(
            None, batch, output, lengths=torch.tensor([2])
        ).mean()).backward()
        self.assertLess(float(stage1_scores.grad[0, 1]), 0.0)
        self.assertGreater(float(stage1_scores.grad[1, 1]), 0.0)

    def test_unavailable_constraints_are_masked_as_used(self):
        module = Stage2JointModule.__new__(Stage2JointModule)
        nn.Module.__init__(module)
        module.num_constraints = 3
        batch = {
            "cand_mask": torch.ones(1, 2),
            "available_constraint_mask": torch.tensor(
                [[True, False, True]]
            ),
        }
        prepared = module._prepare_runtime_masks(batch)
        self.assertTrue(
            torch.equal(
                prepared["used_constraint_mask"],
                torch.tensor([[0.0, 1.0, 0.0]]),
            )
        )

    def test_selector_trajectory_return_distribution_prefers_higher_value_action(self):
        logits = torch.zeros(1, 3, requires_grad=True)
        target_distribution = torch.tensor([[0.8, 0.2, 0.0]])
        loss = selector_ce_loss(
            logits,
            torch.tensor([0]),
            target_distribution=target_distribution,
        )
        loss.backward()
        self.assertGreater(float(loss), 0.0)
        self.assertLess(float(logits.grad[0, 0]), float(logits.grad[0, 1]))
        self.assertGreater(float(logits.grad[0, 2]), 0.0)

    def test_stage2_mask_loss_supervises_terminal_round(self):
        module = Stage2JointModule.__new__(Stage2JointModule)
        nn.Module.__init__(module)
        module.num_constraints = 2
        module.net = _DummyStage2Net()
        module.candidate_filter_loss = candidate_filter_loss
        module.selector_ce_loss = selector_ce_loss
        module.stop_bce_loss = stop_bce_loss
        module.mask_ce_weight = 0.5
        module.mask_bce_weight = 1.0
        module.mask_positive_weight = 2.0
        module.mask_negative_weight = 0.25
        module.use_preference_loss = False
        module.use_metric_loss = False
        module.use_counterfactual_trajectory_loss = False
        module.use_trajectory_dpo_loss = False
        module.use_selector_loss = True
        module.use_mask_loss = True
        module.use_stop_loss = True
        module.w_selector = 1.0
        module.w_mask = 1.0
        module.w_stop = 0.2
        module.w_preference = 0.0
        module.w_metric = 0.0
        module.w_counterfactual = 0.0
        module.w_trajectory_dpo = 0.0
        batch = self._stage2_trajectory_batch([1.0, -1.0])
        loss = module.loss(batch, use_offline_trajectory=False)["stage2_mask_loss"]
        loss.backward()
        # The final row has marker=-1. It must influence the shared scale.
        self.assertIsNotNone(module.net.scale.grad)
        self.assertGreater(float(module.net.scale.grad.abs()), 0.0)

    def test_stage2_selector_supervises_terminal_action(self):
        module = Stage2JointModule.__new__(Stage2JointModule)
        nn.Module.__init__(module)
        module.num_constraints = 2
        module.net = _DummyStage2Net()
        module.candidate_filter_loss = candidate_filter_loss
        module.selector_ce_loss = selector_ce_loss
        module.stop_bce_loss = stop_bce_loss
        module.mask_ce_weight = 0.5
        module.mask_bce_weight = 1.0
        module.mask_positive_weight = 2.0
        module.mask_negative_weight = 0.25
        module.use_preference_loss = False
        module.use_metric_loss = False
        module.use_counterfactual_trajectory_loss = False
        module.use_trajectory_dpo_loss = False
        module.use_selector_loss = True
        module.use_mask_loss = False
        module.use_stop_loss = False
        module.w_selector = 1.0
        module.w_mask = 0.0
        module.w_stop = 0.0
        module.w_preference = 0.0
        module.w_metric = 0.0
        module.w_counterfactual = 0.0
        module.w_trajectory_dpo = 0.0
        batch = self._stage2_trajectory_batch([1.0])
        batch["stop_y"] = torch.ones(1)
        loss = module.loss(batch, use_offline_trajectory=False)[
            "stage2_selector_loss"
        ]
        loss.backward()
        self.assertGreater(float(module.net.scale.grad.abs()), 0.0)

    def test_rollout_composes_round_evidence_instead_of_historical_max(self):
        batch = {
            "cand_mask": torch.ones(1, 2),
            "target_index": torch.tensor([0]),
            "target_candidate_mask": torch.tensor([[True, False]]),
            "constraint_id": torch.tensor([0]),
            "c_ctype_id": torch.tensor([0]),
            "c_rel_id": torch.tensor([0]),
            "c_attr_id": torch.tensor([0]),
            "field_ids": torch.tensor([[1, 0]]),
        }
        pred_idx, _, rounds, _, _, trace = eval_rollout_batch(
            model=_RoundAwareNet(),
            batch=batch,
            constraint_table=_TinyConstraintTable(),
            vocab=_TinyVocab(),
            max_len=2,
            max_round=2,
            thr=0.0,
            stop_thr=1.0,
            eval_mode="agent",
            selector_strategy="ours",
            risk_sensitive_pruning=False,
            return_trace=True,
        )
        self.assertEqual(int(rounds[0]), 2)
        self.assertEqual(int(pred_idx[0]), 0)
        self.assertEqual(int(trace["pred_idx_history"][0][0]), 1)
        self.assertEqual(int(trace["pred_idx_history"][1][0]), 0)

    def test_rollout_accepts_training_aligned_candidate_belief_summary(self):
        batch = {
            "cand_mask": torch.ones(1, 2),
            "cand_belief": torch.tensor([[0.8, 0.2]]),
            "target_index": torch.tensor([0]),
            "target_candidate_mask": torch.tensor([[True, False]]),
            "constraint_id": torch.tensor([0]),
            "c_ctype_id": torch.tensor([0]),
            "c_rel_id": torch.tensor([0]),
            "c_attr_id": torch.tensor([0]),
            "field_ids": torch.tensor([[1, 0]]),
        }
        pred_idx, _, _, _, _ = eval_rollout_batch(
            model=_RoundAwareNet(),
            batch=batch,
            constraint_table=_TinyConstraintTable(),
            vocab=_TinyVocab(),
            max_len=2,
            max_round=2,
            thr=0.0,
            stop_thr=1.0,
            eval_mode="agent",
            v_target_strategy="candidate_belief",
            selector_strategy="ours",
            risk_sensitive_pruning=False,
        )
        self.assertEqual(int(pred_idx[0]), 0)

    def test_rollout_propagates_belief_and_alive_probability_between_rounds(self):
        model = _BeliefRecordingNet()
        batch = {
            "cand_mask": torch.ones(1, 2),
            "cand_belief": torch.tensor([[0.5, 0.5]]),
            "cand_alive_probs": torch.ones(1, 2),
            "target_index": torch.tensor([0]),
            "target_candidate_mask": torch.tensor([[True, False]]),
            "constraint_id": torch.tensor([0]),
            "c_ctype_id": torch.tensor([0]),
            "c_rel_id": torch.tensor([0]),
            "c_attr_id": torch.tensor([0]),
            "field_ids": torch.tensor([[1, 0]]),
        }

        eval_rollout_batch(
            model=model,
            batch=batch,
            constraint_table=_TinyConstraintTable(),
            vocab=_TinyVocab(),
            max_len=2,
            max_round=2,
            thr=0.0,
            stop_thr=1.0,
            eval_mode="agent",
            v_target_strategy="candidate_belief",
            selector_strategy="ours",
            risk_sensitive_pruning=False,
        )

        self.assertTrue(torch.allclose(model.beliefs[0], torch.tensor([[0.5, 0.5]])))
        self.assertTrue(torch.allclose(model.beliefs[2], torch.tensor([[0.8, 0.2]])))
        self.assertTrue(
            torch.allclose(
                model.alive_probabilities[2], torch.tensor([[0.8, 0.2]])
            )
        )

    def test_dynamic_trajectory_state_uses_each_row_round(self):
        trajectory_output = {
            "alive_history": torch.tensor([[[[0.9, 0.8], [0.7, 0.1]]]]),
            "belief_history": torch.tensor([[[[0.6, 0.4], [0.95, 0.05]]]]),
        }
        alive, belief = Stage2JointModule._gather_dynamic_trajectory_state(
            trajectory_output,
            torch.tensor([0, 0]),
            torch.tensor([0, 0]),
            torch.tensor([0, 1]),
        )

        self.assertTrue(
            torch.allclose(alive, torch.tensor([[0.9, 0.8], [0.7, 0.1]]))
        )
        self.assertTrue(
            torch.allclose(belief, torch.tensor([[0.6, 0.4], [0.95, 0.05]]))
        )

    def test_autoregressive_training_mixes_predicted_state_into_next_round(self):
        module = Stage2JointModule.__new__(Stage2JointModule)
        nn.Module.__init__(module)
        module.end_to_end_autoregressive_teacher_mix = 0.5
        model = _BeliefRecordingNet()
        batch = {
            "cand_mask": torch.ones(2, 2),
            "alive_mask": torch.ones(2, 2),
            "cand_belief": torch.tensor([[0.5, 0.5], [0.1, 0.9]]),
            "cand_alive_probs": torch.tensor([[1.0, 1.0], [0.6, 0.4]]),
            "used_constraint_mask": torch.zeros(2, 2),
        }

        output = module._autoregressive_trajectory_forward(
            model, batch, torch.tensor([2])
        )

        self.assertEqual(tuple(output["mask_logits"].shape), (2, 2))
        self.assertTrue(
            torch.allclose(model.beliefs[1], torch.tensor([[0.45, 0.55]]))
        )
        self.assertTrue(
            torch.allclose(
                model.alive_probabilities[1], torch.tensor([[0.7, 0.3]])
            )
        )

    def test_autoregressive_training_backpropagates_future_loss_to_prior_action(self):
        module = Stage2JointModule.__new__(Stage2JointModule)
        nn.Module.__init__(module)
        module.end_to_end_autoregressive_teacher_mix = 0.5
        model = _TransitionGradientNet()
        batch = {
            "cand_mask": torch.ones(2, 2),
            "alive_mask": torch.ones(2, 2),
            "cand_belief": torch.tensor([[0.5, 0.5], [0.1, 0.9]]),
            "cand_alive_probs": torch.ones(2, 2),
            "used_constraint_mask": torch.zeros(2, 2),
        }

        output = module._autoregressive_trajectory_forward(
            model, batch, torch.tensor([2])
        )
        output["mask_logits"][1, 0].backward()

        self.assertIsNotNone(model.round_logits[0].grad)
        self.assertGreater(float(model.round_logits[0].grad.abs().sum()), 0.0)

    def test_rollout_stops_when_available_actions_are_exhausted(self):
        batch = {
            "cand_mask": torch.ones(1, 2),
            "target_index": torch.tensor([0]),
            "target_candidate_mask": torch.tensor([[True, False]]),
            "constraint_id": torch.tensor([0]),
            "c_ctype_id": torch.tensor([0]),
            "c_rel_id": torch.tensor([0]),
            "c_attr_id": torch.tensor([0]),
            "field_ids": torch.tensor([[1, 0]]),
            "available_constraint_mask": torch.ones(1, 1, dtype=torch.bool),
        }

        class OneActionNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.tensor(0.0))

            def forward(self, current):
                size = current["cand_mask"].shape[0]
                return {
                    "selector_logits": torch.zeros(size, 1) + self.anchor * 0.0,
                    "mask_logits": torch.full((size, 2), 10.0) + self.anchor * 0.0,
                    "stop_logit": torch.full((size,), -10.0) + self.anchor * 0.0,
                    "cand_feat": torch.zeros(size, 2, 4),
                }

        _, _, rounds, _, _ = eval_rollout_batch(
            model=OneActionNet(),
            batch=batch,
            constraint_table=_OneConstraintTable(),
            vocab=_TinyVocab(),
            max_len=2,
            max_round=3,
            thr=0.0,
            stop_thr=1.0,
            eval_mode="agent",
            selector_strategy="ours",
            risk_sensitive_pruning=False,
        )
        self.assertEqual(int(rounds[0]), 1)

    def test_scheduled_rollout_cannot_stop_before_required_actions(self):
        class EarlyStopNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.tensor(0.0))

            def forward(self, current):
                used = current["used_constraint_mask"]
                selector_logits = torch.zeros_like(used).masked_fill(used > 0, -1e9)
                size = used.shape[0]
                return {
                    "selector_logits": selector_logits + self.anchor * 0.0,
                    "mask_logits": torch.tensor(
                        [[10.0, -10.0]], device=used.device
                    ).expand(size, -1) + self.anchor * 0.0,
                    "stop_logit": torch.full(
                        (size,), 10.0, device=used.device
                    ) + self.anchor * 0.0,
                    "cand_feat": torch.zeros(size, 2, 4, device=used.device),
                }

        batch = {
            "cand_mask": torch.ones(1, 2),
            "target_index": torch.tensor([0]),
            "target_candidate_mask": torch.tensor([[True, False]]),
            "constraint_id": torch.tensor([0]),
            "c_ctype_id": torch.tensor([0]),
            "c_rel_id": torch.tensor([0]),
            "c_attr_id": torch.tensor([0]),
            "field_ids": torch.tensor([[1, 0]]),
            "available_constraint_mask": torch.tensor([[True, False]]),
            "state_constraint_ids": torch.tensor([[0, 1]]),
            "minimum_rollout_rounds": torch.tensor([2]),
            "required_rollout_rounds": torch.tensor([2]),
        }
        _, _, rounds, _, _, trace = eval_rollout_batch(
            model=EarlyStopNet(),
            batch=batch,
            constraint_table=_TinyConstraintTable(),
            vocab=_TinyVocab(),
            max_len=2,
            max_round=3,
            thr=0.5,
            stop_thr=0.5,
            eval_mode="agent",
            selector_strategy="ours",
            risk_sensitive_pruning=False,
            return_trace=True,
        )

        self.assertEqual(int(rounds[0]), 2)
        self.assertEqual(
            [int(step[0]) for step in trace["selected_constraint_history"]],
            [0, 1],
        )

    def test_explicit_negative_dpo_includes_early_stop(self):
        module = self._stage2_dpo_module()
        losses = module._trajectory_dpo_losses(
            {
                "flat_batch": self._stage2_trajectory_batch([1.0, 1.0]),
                "negative_flat_batch": self._stage2_trajectory_batch([-1.0, -1.0]),
                "lengths": torch.tensor([2]),
            }
        )
        self.assertGreater(float(losses["stage2_dpo_wrong_loss"]), 0.0)
        self.assertGreater(float(losses["stage2_dpo_early_stop_loss"]), 0.0)
        losses["stage2_trajectory_dpo_loss"].backward()
        self.assertGreater(float(module.net.scale.grad.abs()), 0.0)

    def test_terminal_only_trajectory_has_no_early_stop_preference(self):
        module = self._stage2_dpo_module()
        losses = module._trajectory_dpo_losses(
            {
                "flat_batch": self._stage2_trajectory_batch([1.0]),
                "negative_flat_batch": self._stage2_trajectory_batch([-1.0]),
                "lengths": torch.tensor([1]),
                "preference_step_mask": torch.tensor([True]),
            }
        )
        self.assertEqual(float(losses["stage2_dpo_early_stop_loss"]), 0.0)
        self.assertEqual(
            float(losses["stage2_trajectory_dpo_loss"]),
            float(losses["stage2_dpo_wrong_loss"]),
        )

    def test_strict_dpo_does_not_fabricate_missing_negative(self):
        module = self._stage2_dpo_module(require_explicit=True)
        losses = module._trajectory_dpo_losses(
            {
                "flat_batch": self._stage2_trajectory_batch([1.0, 1.0]),
                "lengths": torch.tensor([2]),
            }
        )
        self.assertEqual(float(losses["stage2_trajectory_dpo_loss"]), 0.0)

    def test_effective_candidate_count_uses_configured_threshold(self):
        common = {
            "true_target_id": 7,
            "num_compatible_targets": 2,
            "eval_type": "mt",
            "is_any_target_iou25": 1,
            "is_true_target_iou25": 1,
            "is_true_target_iou50": 1,
            "is_any_target_iou50": 1,
            "matched_target_ids_iou25": {7},
            "matched_target_ids_iou50": {7},
            "_has_identity_matches": 1,
        }
        rows = [
            {**common, "probability": 0.8, "selected_at_threshold": 1},
            {
                **common,
                "probability": 0.2,
                "selected_at_threshold": 0,
                "is_any_target_iou50": 0,
                "is_true_target_iou50": 0,
                "matched_target_ids_iou50": set(),
            },
            {
                **common,
                "probability": 0.1,
                "selected_at_threshold": 0,
                "is_any_target_iou50": 0,
                "is_true_target_iou50": 0,
                "matched_target_ids_iou50": set(),
            },
        ]
        metrics = _query_metrics(
            "test", ("scene", 1), rows, None, topks=[1, 3]
        )
        self.assertEqual(metrics["effective_candidate_count"], 1)
        self.assertEqual(
            metrics["query_matched_share_in_effective_candidates_iou50"], 1.0
        )

    def test_candidate_filter_loss_calibrates_target_logit(self):
        logits = torch.zeros(1, 4, requires_grad=True)
        loss = candidate_filter_loss(
            logits,
            target_index=torch.tensor([1]),
            cand_mask=torch.ones(1, 4),
            target_mask=torch.tensor([[False, True, False, False]]),
        )
        loss.backward()
        self.assertLess(float(logits.grad[0, 1]), 0.0)
        self.assertTrue(torch.all(logits.grad[0, [0, 2, 3]] > 0))

    def test_candidate_filter_loss_is_centered_on_rollout_threshold(self):
        threshold = 0.4
        boundary = torch.logit(torch.tensor(threshold))
        raw_logits = torch.tensor(
            [[boundary - 0.2, boundary + 0.3]], requires_grad=True
        )
        shifted_logits = (raw_logits.detach() - boundary).requires_grad_(True)
        kwargs = {
            "target_index": torch.tensor([1]),
            "cand_mask": torch.ones(1, 2),
            "target_mask": torch.tensor([[False, True]]),
            "ce_weight": 0.0,
        }

        aligned = candidate_filter_loss(
            raw_logits, decision_threshold=threshold, **kwargs
        )
        reference = candidate_filter_loss(
            shifted_logits, decision_threshold=0.5, **kwargs
        )

        self.assertTrue(torch.allclose(aligned, reference, atol=1e-7))

    def test_selector_loss_accepts_multiple_valid_actions(self):
        logits = torch.zeros(1, 3, requires_grad=True)
        loss = selector_ce_loss(
            logits,
            constraint_id=torch.tensor([0]),
            target_mask=torch.tensor([[True, True, False]]),
        )
        loss.backward()
        self.assertLess(float(logits.grad[0, 0]), 0.0)
        self.assertLess(float(logits.grad[0, 1]), 0.0)
        self.assertGreater(float(logits.grad[0, 2]), 0.0)

    def test_risk_sensitive_pruning_limits_single_round_deletion(self):
        probability = torch.tensor([[0.9, 0.3, 0.2, 0.1]])
        alive = torch.ones_like(probability)
        updated = risk_sensitive_candidate_update(
            probability,
            alive,
            threshold=0.4,
            selector_confidence=torch.tensor([0.1]),
            min_keep_ratio=0.5,
            min_keep_count=1,
        )
        self.assertGreaterEqual(int(updated.sum()), 2)
        self.assertEqual(float(updated[0, 0]), 1.0)

    def test_terminal_action_can_prune_below_risk_keep_ratio(self):
        probability = torch.tensor(
            [[0.9, 0.3, 0.2, 0.1], [0.9, 0.3, 0.2, 0.1]]
        )
        alive = torch.ones_like(probability)
        updated = risk_sensitive_candidate_update(
            probability,
            alive,
            threshold=0.4,
            selector_confidence=torch.tensor([1.0, 1.0]),
            min_keep_ratio=0.5,
            min_keep_count=1,
            terminal_ready=torch.tensor([True, False]),
        )

        self.assertEqual(int(updated[0].sum()), 1)
        self.assertGreaterEqual(int(updated[1].sum()), 2)

    def test_alive_count_metrics_carry_stopped_state_forward(self):
        totals = {1: 0, 2: 0, 3: 0}
        _accumulate_alive_counts(
            totals,
            [torch.tensor([2, 1]), torch.tensor([1, 1])],
            max_round=3,
        )

        self.assertEqual(totals, {1: 3, 2: 2, 3: 2})

    def test_semantic_selector_scores_constraint_bank_and_backpropagates(self):
        selector = ConstraintSelectorHead(
            d_state=8,
            d_text=8,
            d_cand=8,
            num_constraints=4,
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_ff=32,
            dropout=0.0,
            vocab_size=12,
            num_rel=3,
            num_attr=4,
            use_semantic_constraint_scoring=True,
        )
        bank = {
            "ctype_ids": torch.tensor([0, 0, 1, 1]),
            "rel_ids": torch.tensor([0, 1, 0, 0]),
            "attr_ids": torch.tensor([3, 3, 1, 2]),
            "field_ids": torch.tensor(
                [[1, 2, 0], [1, 3, 0], [4, 5, 0], [4, 6, 0]]
            ),
        }
        logits = selector(
            state=torch.randn(2, 8),
            txt=torch.randn(2, 8),
            cand_feat=torch.randn(2, 3, 8),
            alive_mask=torch.ones(2, 3),
            constraint_bank=bank,
        )
        self.assertEqual(tuple(logits.shape), (2, 4))
        logits[:, 0].sum().backward()
        self.assertGreater(float(selector.constraint_word.weight.grad.abs().sum()), 0)
        self.assertGreater(float(selector.out.weight.grad.abs().sum()), 0)

    def test_stage2_text_encoders_ignore_trailing_padding(self):
        short = torch.tensor([[2, 3, 0, 0]])
        long = torch.tensor([[2, 3, 0, 0, 0, 0]])

        query_encoder = QueryEncoder(vocab_size=10, d_text=8).eval()
        self.assertTrue(
            torch.allclose(
                query_encoder(short), query_encoder(long), atol=1e-6
            )
        )

        constraint_encoder = SemanticConstraintEmbedder(
            d=8, vocab_size=10, num_rel=3, num_attr=4
        ).eval()
        short_embedding = constraint_encoder(
            torch.tensor([0]), torch.tensor([1]), torch.tensor([2]), short
        )
        long_embedding = constraint_encoder(
            torch.tensor([0]), torch.tensor([1]), torch.tensor([2]), long
        )
        self.assertTrue(
            torch.allclose(short_embedding, long_embedding, atol=1e-6)
        )

    def test_stage2_stop_and_mask_follow_current_candidate_state(self):
        cfg = TrainConfig(
            imgfeat_dim=6,
            d_text=8,
            d_constraint=8,
            d_cand=8,
            d_state=8,
        )
        model = Stage2NetSemantic(
            vocab_size=12,
            cfg=cfg,
            num_constraints=2,
            num_rel=3,
            num_attr=4,
        ).eval()
        model.stop = _StopStatsProbe()
        batch = {
            "query_ids": torch.tensor([[1, 2, 0]]),
            "c_ctype_id": torch.tensor([0]),
            "c_rel_id": torch.tensor([1]),
            "c_attr_id": torch.tensor([2]),
            "field_ids": torch.tensor([[3, 4, 0]]),
            "img": torch.randn(1, 3, 6),
            "cand_bboxes": torch.zeros(1, 3, 7),
            "cand_mask": torch.ones(1, 3),
        }
        all_alive = model({**batch, "alive_mask": torch.ones(1, 3)})
        one_alive = model(
            {**batch, "alive_mask": torch.tensor([[1.0, 0.0, 0.0]])}
        )
        self.assertNotEqual(
            float(all_alive["stop_logit"]), float(one_alive["stop_logit"])
        )
        self.assertTrue(torch.all(one_alive["mask_logits"][0, 1:] < -1e8))

    def test_constraint_conditioned_context_backpropagates_landmark_geometry(self):
        cfg = TrainConfig(
            imgfeat_dim=6,
            d_text=8,
            d_constraint=8,
            d_cand=8,
            d_state=8,
            use_bbox_geom=True,
            use_constraint_conditioned_context=True,
        )
        model = Stage2NetSemantic(
            vocab_size=12,
            cfg=cfg,
            num_constraints=2,
            num_rel=3,
            num_attr=4,
        )
        batch = {
            "query_ids": torch.tensor([[1, 2, 0]]),
            "c_ctype_id": torch.tensor([0]),
            "c_rel_id": torch.tensor([1]),
            "c_attr_id": torch.tensor([2]),
            "field_ids": torch.tensor([[3, 4, 0]]),
            "img": torch.randn(1, 3, 6),
            "cand_bboxes": torch.tensor(
                [[[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                  [1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0],
                  [0.0, 2.0, 0.0, 2.0, 1.0, 1.0, 0.0]]]
            ),
            "cand_mask": torch.ones(1, 3),
            "alive_mask": torch.tensor([[1.0, 1.0, 0.0]]),
            "context_mask": torch.ones(1, 3),
        }

        output = model(batch)
        output["mask_logits"][:, :2].sum().backward()

        self.assertTrue(torch.isfinite(output["mask_logits"][:, :2]).all())
        self.assertGreater(
            float(model.constraint_context_condition.weight.grad.abs().sum()), 0.0
        )
        self.assertGreater(
            float(model.constraint_relation_geom[0].weight.grad.abs().sum()), 0.0
        )

    def test_stage2_empty_alive_state_falls_back_to_valid_candidates(self):
        cfg = TrainConfig(
            imgfeat_dim=4,
            d_text=8,
            d_constraint=8,
            d_cand=8,
            d_state=8,
        )
        model = Stage2NetSemantic(
            vocab_size=10,
            cfg=cfg,
            num_constraints=2,
            num_rel=3,
            num_attr=4,
        ).eval()
        output = model({
            "query_ids": torch.tensor([[1, 0]]),
            "c_ctype_id": torch.tensor([0]),
            "c_rel_id": torch.tensor([0]),
            "c_attr_id": torch.tensor([0]),
            "field_ids": torch.tensor([[1, 0]]),
            "img": torch.randn(1, 2, 4),
            "cand_bboxes": torch.zeros(1, 2, 7),
            "cand_mask": torch.ones(1, 2),
            "alive_mask": torch.zeros(1, 2),
        })
        self.assertTrue(torch.isfinite(output["stop_logit"]).all())
        self.assertTrue(torch.isfinite(output["mask_logits"]).all())

    def test_stage2_belief_summary_excludes_eliminated_candidates(self):
        cfg = TrainConfig(
            imgfeat_dim=4,
            d_text=8,
            d_constraint=8,
            d_cand=8,
            d_state=8,
        )
        model = Stage2NetSemantic(
            vocab_size=10,
            cfg=cfg,
            num_constraints=2,
            num_rel=3,
            num_attr=4,
        ).eval()
        probe = _SelectorTargetProbe()
        model.selector = probe
        output = model({
            "query_ids": torch.tensor([[1, 0]]),
            "c_ctype_id": torch.tensor([0]),
            "c_rel_id": torch.tensor([0]),
            "c_attr_id": torch.tensor([0]),
            "field_ids": torch.tensor([[1, 0]]),
            "img": torch.randn(1, 2, 4),
            "cand_bboxes": torch.zeros(1, 2, 7),
            "cand_mask": torch.ones(1, 2),
            "alive_mask": torch.tensor([[1.0, 0.0]]),
            "cand_belief": torch.tensor([[0.1, 0.9]]),
        })
        self.assertTrue(
            torch.allclose(
                probe.target_summary,
                output["cand_feat"][:, 0],
                atol=1e-6,
            )
        )


if __name__ == "__main__":
    unittest.main()
