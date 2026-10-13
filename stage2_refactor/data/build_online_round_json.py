from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from .convert_dialogue_json_semantic import (
    _candidate_ids_from_dialogue,
    infer_semantic_constraint,
)
from m3drefclip.data.online_trajectory import (
    build_minimal_compositional_trajectories,
)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _negative_feedback_query(query: str, question: str) -> str:
    query = _clean_text(query).rstrip(" .!?")
    question = _clean_text(question).rstrip(" ?")
    if not query or not question:
        return ""
    return f"{query}. No, the answer to '{question}' is no."


def _query(dialogue: Dict[str, Any]) -> str:
    return _clean_text(
        dialogue.get("input_query")
        or dialogue.get("query_before")
        or dialogue.get("degraded_query")
    )


def _restored_query(dialogue: Dict[str, Any]) -> str:
    return _clean_text(
        dialogue.get("restored_query")
        or dialogue.get("query_after")
        or dialogue.get("restored_query_raw")
    )


def _constraint(text: str) -> Dict[str, Any] | None:
    value = infer_semantic_constraint(text)
    field = value.get("landmark_text") or value.get("attr_val_text")
    return value if _clean_text(field) else None


def _candidate_ids(dialogue: Dict[str, Any], target_id: int) -> List[int]:
    values = {int(value) for value in _candidate_ids_from_dialogue(dialogue, target_id)}
    values.add(int(target_id))
    return sorted(value for value in values if value >= 0)


def _minimal_compositional_samples(
    item: Dict[str, Any], scene_id: str, ann_id: str, target_id: int
) -> List[Dict[str, Any]]:
    output = []
    for trajectory in build_minimal_compositional_trajectories(item, max_rounds=3):
        queries = trajectory["queries"]
        action_queries = trajectory["action_queries"]
        candidate_sets = trajectory["candidate_sets"]
        question = trajectory["generated_question"]
        constraints = []
        for action_query in action_queries:
            constraint = _constraint(action_query) or _constraint(queries[-1])
            if constraint is None:
                constraint = {"ctype": "ATTR", "attr_key": "description", "attr_val_text": action_query}
            if constraints and constraint == constraints[-1]:
                constraint = {
                    "ctype": "ATTR",
                    "attr_key": "description",
                    "attr_val_text": action_query,
                }
            if constraints and constraint == constraints[-1]:
                follow_up = _clean_text(
                    question
                    or f"follow-up clarification: {action_query}"
                )
                constraint = {
                    "ctype": "ATTR",
                    "attr_key": "follow_up",
                    "attr_val_text": follow_up,
                }
            constraints.append(constraint)
        initial_query = queries[0]
        trajectory_id = trajectory["trajectory_id"]
        negative_queries = trajectory["negative_queries"]
        for round_idx, constraint_gt in enumerate(constraints):
            candidates = candidate_sets[round_idx]
            terminal = round_idx == len(constraints) - 1
            post_action_candidates = (
                [target_id]
                if terminal
                else list(candidate_sets[round_idx + 1])
            )
            output.append({
                "scene_id": scene_id,
                "ann_id": ann_id,
                "trajectory_id": trajectory_id,
                "ambiguity_type": "minimal",
                "round": round_idx,
                # Rollout retains the initial language state; applied constraints
                # and candidate dynamics carry the interaction history.
                "query": initial_query,
                "state_query": (
                    queries[min(round_idx, len(queries) - 1)]
                ),
                "negative_query": negative_queries[
                    min(round_idx + 1, len(negative_queries) - 1)
                ],
                "generated_question": question,
                "cand_ids": list(candidates),
                "filter_cand_ids": post_action_candidates,
                "target_id": target_id,
                "target_survived_stage1": True,
                "constraint_gt": constraint_gt,
                "stop_y": int(terminal),
                "target_index": list(candidates).index(target_id),
            })
    return output


def build_round_samples(items: List[Dict[str, Any]], ambiguity_type: str = "") -> List[Dict[str, Any]]:
    output = []
    for item in items:
        scene_id = str(item.get("scene_id", ""))
        ann_id = str(item.get("ann_id", ""))
        target_id = int(item.get("target_id", item.get("true_target_id", -1)))
        if ambiguity_type == "minimal":
            output.extend(
                _minimal_compositional_samples(
                    item, scene_id, ann_id, target_id
                )
            )
            continue
        for dialogue_idx, dialogue in enumerate(item.get("dialogues") or []):
            current_type = str(dialogue.get("ambiguity_type", "unknown"))
            if ambiguity_type and current_type != ambiguity_type:
                continue
            if not bool(dialogue.get("is_ambiguous", False)):
                continue
            query_before = _clean_text(
                dialogue.get("input_query")
                or dialogue.get("query_before")
                or dialogue.get("degraded_query")
            )
            query_after = _clean_text(
                dialogue.get("restored_query")
                or dialogue.get("query_after")
                or dialogue.get("restored_query_raw")
            )
            if not query_before or not query_after or query_before.lower() == query_after.lower():
                continue
            candidate_ids = _candidate_ids_from_dialogue(dialogue, target_id)
            # Grounding-match generation occasionally omits the annotated target.
            # Repair that source-label conflict before enforcing monotonic filtering.
            if target_id not in candidate_ids:
                candidate_ids = list(candidate_ids) + [target_id]
            generated_question = _clean_text(dialogue.get("generated_question"))
            teacher = generated_question or query_after
            constraint = infer_semantic_constraint(teacher)
            field_text = constraint.get("landmark_text") or constraint.get(
                "attr_val_text"
            )
            if not field_text and query_after:
                question_constraint = infer_semantic_constraint(
                    query_after
                )
                question_field = question_constraint.get(
                    "landmark_text"
                ) or question_constraint.get("attr_val_text")
                if question_field:
                    constraint = question_constraint
            trajectory_id = f"{current_type}:{dialogue_idx}"
            rejected_query = _negative_feedback_query(
                query_before, generated_question
            )
            shuffled_ids = list(candidate_ids)
            output.append({
                "scene_id": scene_id,
                "ann_id": ann_id,
                "trajectory_id": trajectory_id,
                "ambiguity_type": current_type,
                "round": 0,
                "query": query_before,
                "negative_query": rejected_query,
                "generated_question": generated_question,
                "cand_ids": shuffled_ids,
                "filter_cand_ids": [target_id],
                "target_id": target_id,
                "target_survived_stage1": True,
                "constraint_gt": constraint,
                "stop_y": 1,
                "target_index": shuffled_ids.index(target_id),
            })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--ambiguity_type", default="")
    args = parser.parse_args()
    with open(args.input_json, "r", encoding="utf-8") as f:
        items = json.load(f)
    samples = build_round_samples(items, ambiguity_type=args.ambiguity_type)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    trajectory_count = len({
        (x["scene_id"], x["ann_id"], x["target_id"], x["trajectory_id"])
        for x in samples
    })
    print(f"saved {len(samples)} rounds from {trajectory_count} trajectories to {args.output_json}")


if __name__ == "__main__":
    main()
