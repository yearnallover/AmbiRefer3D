import os
import json
import torch
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from m3drefclip.util.utils import get_batch_aabb_pair_ious
from m3drefclip.evaluation.general_evaluator import GeneralEvaluator


class Multi3DReferEvaluator(GeneralEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluation_types = {"zt_w_d": 0, "zt_wo_d": 1, "st_w_d": 2, "st_wo_d": 3, "mt": 4}
        self.last_query_results = []

    def _set_ground_truths_from_files(self, path):
        self.ground_truths = {}
        scene_files = os.listdir(path)
        for scene_file in scene_files:
            with open(os.path.join(path, scene_file), "r") as f:
                gt_json = json.load(f)
            for query in gt_json:
                self.ground_truths[(scene_file[:-5], 0, query["ann_id"])] = {
                    "aabb_bound": np.array(query["aabb_bound"], dtype=np.float32),
                    "eval_type": query["eval_type"]
                }

    def evaluate(self, predictions):
        all_gt_info_len = len(predictions)
        eval_type_mask = np.empty(all_gt_info_len, dtype=np.uint8)
        scores = {
            "f1@0.25": np.zeros(all_gt_info_len, dtype=np.float32),
            "f1@0.5": np.zeros(all_gt_info_len, dtype=np.float32),
            "precision@0.25": np.zeros(all_gt_info_len, dtype=np.float32),
            "precision@0.5": np.zeros(all_gt_info_len, dtype=np.float32),
            "recall@0.25": np.zeros(all_gt_info_len, dtype=np.float32),
            "recall@0.5": np.zeros(all_gt_info_len, dtype=np.float32),
            "target_survival@0.25": np.full(all_gt_info_len, np.nan, dtype=np.float32),
            "target_survival@0.5": np.full(all_gt_info_len, np.nan, dtype=np.float32),
            "target_survival_no_iou": np.full(all_gt_info_len, np.nan, dtype=np.float32),
            "raw_proposal_target_survival@0.25": np.full(all_gt_info_len, np.nan, dtype=np.float32),
            "raw_proposal_target_survival@0.5": np.full(all_gt_info_len, np.nan, dtype=np.float32),
            "raw_proposal_target_survival_no_iou": np.full(all_gt_info_len, np.nan, dtype=np.float32),
        }
        query_records = []
        iterator = enumerate(tqdm(predictions.items(), desc="Evaluating") if self.verbose else predictions.items())
        for i, (key, value) in iterator:
            gt_info = self.ground_truths[key]
            eval_type_mask[i] = self.evaluation_types[gt_info["eval_type"]]
            query_scores = {metric_name: np.nan for metric_name in scores}
            if gt_info["eval_type"] in ("zt_wo_d", "zt_w_d"):
                zero_gt_score = self.evaluate_one_zero_gt_query(value)
                for metric_name in ("f1", "precision", "recall"):
                    scores[f"{metric_name}@0.25"][i] = zero_gt_score
                    scores[f"{metric_name}@0.5"][i] = zero_gt_score
                    query_scores[f"{metric_name}@0.25"] = zero_gt_score
                    query_scores[f"{metric_name}@0.5"] = zero_gt_score
            else:
                query_scores = self.evaluate_one_query(value, gt_info)
                for metric_name, metric_value in query_scores.items():
                    scores[metric_name][i] = metric_value
            query_records.append({
                "scene_id": str(key[0]),
                "object_id": int(key[1]),
                "ann_id": int(key[2]),
                "eval_type": str(gt_info["eval_type"]),
                "num_predictions": int(len(value["aabb_bound"])),
                "num_targets": int(len(gt_info["aabb_bound"])),
                "target_volume": self._mean_aabb_volume(gt_info["aabb_bound"]),
                **{name: self._as_optional_float(query_scores.get(name, np.nan)) for name in scores},
            })

        results = {}
        for metric_name, metric_scores in scores.items():
            metric_results = {}
            for sub_group in ("zt_w_d", "zt_wo_d", "st_w_d", "st_wo_d", "mt"):
                selected_indices = eval_type_mask == self.evaluation_types[sub_group]
                metric_results[sub_group] = self._safe_mean(metric_scores[selected_indices])
            metric_results["overall"] = self._safe_mean(metric_scores)
            results[metric_name] = metric_results

        if self.verbose:
            self._print_results(results)

        self.last_query_results = query_records
        return results

    @staticmethod
    def _safe_mean(values):
        values = np.asarray(values, dtype=np.float32)
        if values.size == 0 or np.isnan(values).all():
            return np.nan
        return float(np.nanmean(values))

    def _print_results(self, results):
        print(f"{'=' * 100}")
        print("{0:<24}{1:<12}{2:<12}{3:<12}{4:<12}{5:<12}{6:<12}"
              .format("Metric", "zt_w_d", "zt_wo_d", "st_w_d", "st_wo_d", "mt", "overall"))
        print(f"{'-' * 100}")
        for metric_name, metric_results in results.items():
            line = f"{metric_name:<24}"
            for sub_group_type in ("zt_w_d", "zt_wo_d", "st_w_d", "st_wo_d", "mt", "overall"):
                score = metric_results[sub_group_type]
                line += f"{score * 100:<12.1f}" if not np.isnan(score) else f"{'nan':<12}"
            print(line)
        print(f"{'=' * 100}\n")

    @staticmethod
    def evaluate_one_query(pred_info, gt_info):
        pred_bboxes_count = len(pred_info["aabb_bound"])
        gt_bboxes_count = len(gt_info["aabb_bound"])

        tp = {0.25: 0, 0.5: 0}

        # initialize the cost matrix
        square_matrix_len = max(gt_bboxes_count, pred_bboxes_count)
        iou_matrix = np.zeros(shape=(square_matrix_len, square_matrix_len), dtype=np.float32)
        # TODO: convert to batch process
        for pred_aabb_idx, pred_aabb in enumerate(pred_info["aabb_bound"]):
            for gt_aabb_idx, gt_aabb in enumerate(gt_info["aabb_bound"]):
                iou_matrix[pred_aabb_idx, gt_aabb_idx] = get_batch_aabb_pair_ious(
                    torch.from_numpy(gt_aabb).unsqueeze(0), torch.from_numpy(pred_aabb).unsqueeze(0)
                )

        # apply matching algorithm
        row_idx, col_idx = linear_sum_assignment(iou_matrix * -1)

        # iterate matched pairs, check ious
        for i in range(pred_bboxes_count):
            iou = iou_matrix[row_idx[i], col_idx[i]]
            for threshold in tp:
                if iou >= threshold:
                    tp[threshold] += 1

        max_iou_per_gt = iou_matrix[:pred_bboxes_count, :gt_bboxes_count].max(axis=0) \
            if pred_bboxes_count > 0 and gt_bboxes_count > 0 else np.zeros(gt_bboxes_count, dtype=np.float32)

        out = {}
        for threshold, true_positive in tp.items():
            precision = true_positive / pred_bboxes_count if pred_bboxes_count > 0 else 0.0
            recall = true_positive / gt_bboxes_count if gt_bboxes_count > 0 else 0.0
            f1_score = 2 * true_positive / (pred_bboxes_count + gt_bboxes_count) \
                if pred_bboxes_count + gt_bboxes_count > 0 else 0.0
            survival = float(np.any(max_iou_per_gt >= threshold)) if gt_bboxes_count > 0 else np.nan
            suffix = "0.25" if threshold == 0.25 else "0.5"
            out[f"precision@{suffix}"] = precision
            out[f"recall@{suffix}"] = recall
            out[f"f1@{suffix}"] = f1_score
            out[f"target_survival@{suffix}"] = survival
        out["target_survival_no_iou"] = Multi3DReferEvaluator._as_optional_float(
            pred_info.get("target_survival_no_iou", np.nan)
        )
        for metric_name in (
            "raw_proposal_target_survival@0.25",
            "raw_proposal_target_survival@0.5",
            "raw_proposal_target_survival_no_iou",
        ):
            out[metric_name] = Multi3DReferEvaluator._as_optional_float(pred_info.get(metric_name, np.nan))
        return out

    @staticmethod
    def _as_optional_float(value):
        if value is None:
            return np.nan
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    @staticmethod
    def _mean_aabb_volume(aabbs):
        aabbs = np.asarray(aabbs, dtype=np.float32)
        if aabbs.size == 0:
            return np.nan
        sizes = np.maximum(aabbs[:, 1] - aabbs[:, 0], 0.0)
        return float(np.prod(sizes, axis=1).mean())

    @staticmethod
    def evaluate_one_zero_gt_query(pred_info):
        return 1 if len(pred_info["aabb_bound"]) == 0 else 0
