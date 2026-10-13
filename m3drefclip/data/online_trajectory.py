import json
from collections import defaultdict


def _ann_key(value):
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _clean_text(value):
    return " ".join(str(value or "").strip().split())


def _append_unique_round(rounds, text):
    text = _clean_text(text)
    if text and (not rounds or text.lower() != rounds[-1].lower()):
        rounds.append(text)


def _negative_feedback_query(query, question):
    query = _clean_text(query).rstrip(" .!?")
    question = _clean_text(question).rstrip(" ?")
    if not query or not question:
        return ""
    return f"{query}. No, the answer to '{question}' is no."


def _candidate_ids(dialogue, target_id):
    values = dialogue.get("grounding_matches") or []
    candidates = {int(value) for value in values if int(value) >= 0}
    if target_id >= 0:
        candidates.add(int(target_id))
    return sorted(candidates)


def build_minimal_compositional_trajectories(item, max_rounds):
    dialogues = {
        str(dialogue.get("ambiguity_type", "")): dialogue
        for dialogue in (item.get("dialogues") or [])
    }
    minimal = dialogues.get("minimal") or {}
    if not bool(minimal.get("is_ambiguous", False)):
        return []

    scene_id = str(item.get("scene_id", ""))
    ann_id = _ann_key(item.get("ann_id", ""))
    target_id = int(item.get("target_id", item.get("true_target_id", -1)))
    initial = _clean_text(
        minimal.get("input_query")
        or minimal.get("query_before")
        or minimal.get("degraded_query")
    )
    restored = _clean_text(
        minimal.get("restored_query")
        or minimal.get("query_after")
        or minimal.get("restored_query_raw")
    )
    if not scene_id or target_id < 0 or not initial or not restored:
        return []

    initial_candidates = _candidate_ids(minimal, target_id)
    branches = (
        (
            "visual_first",
            dialogues.get("no_spatial") or {},
            dialogues.get("no_visual") or {},
        ),
        (
            "spatial_first",
            dialogues.get("no_visual") or {},
            dialogues.get("no_spatial") or {},
        ),
    )
    output = []
    seen_queries = set()
    for branch_name, partial_dialogue, second_dialogue in branches:
        partial = _clean_text(
            partial_dialogue.get("input_query")
            or partial_dialogue.get("query_before")
            or partial_dialogue.get("degraded_query")
        )
        if not partial or partial.lower() == initial.lower():
            continue
        partial_candidates = sorted(
            set(_candidate_ids(partial_dialogue, target_id))
            & set(initial_candidates)
        )
        if target_id not in partial_candidates:
            partial_candidates.append(target_id)
            partial_candidates.sort()

        queries = [initial, partial]
        candidate_sets = [initial_candidates, partial_candidates]
        if (
            len(partial_candidates) > 1
            and restored.lower() not in {query.lower() for query in queries}
        ):
            queries.append(restored)
            candidate_sets.append([target_id])
        queries = queries[:max_rounds]
        candidate_sets = candidate_sets[: len(queries)]
        query_key = tuple(query.lower() for query in queries)
        if len(queries) < 2 or query_key in seen_queries:
            continue
        seen_queries.add(query_key)

        question = _clean_text(minimal.get("generated_question"))
        second_action = _clean_text(
            second_dialogue.get("input_query")
            or second_dialogue.get("query_before")
            or second_dialogue.get("degraded_query")
        )
        action_queries = [partial]
        if len(queries) > 2:
            action_queries.append(second_action or restored)
        negative_queries = [initial]
        for action_index, action_query in enumerate(action_queries):
            state_query = queries[min(action_index, len(queries) - 1)]
            action_question = f"Does this description apply: {action_query}"
            negative_queries.append(
                _negative_feedback_query(state_query, action_question)
            )
        output.append({
            "queries": queries,
            "action_queries": action_queries,
            "negative_queries": negative_queries,
            "feedback": [0.0] + [1.0] * (len(queries) - 1),
            "negative_feedback": [0.0] + [-1.0] * (len(queries) - 1),
            "candidate_sets": candidate_sets,
            "target_id": target_id,
            "trajectory_type": "minimal",
            "trajectory_id": f"minimal:{branch_name}",
            "source_ann_id": ann_id,
            "generated_question": question,
        })
    return output


def _nested_dialogue_trajectories(items, max_rounds, ambiguity_type=""):
    trajectories = defaultdict(list)
    for item in items:
        scene_id = str(item.get("scene_id", ""))
        ann_id = _ann_key(item.get("ann_id", ""))
        target_id = item.get("target_id", item.get("true_target_id", -1))
        if not scene_id:
            continue
        try:
            target_id = int(target_id)
        except (TypeError, ValueError):
            target_id = -1

        if ambiguity_type == "minimal":
            options = build_minimal_compositional_trajectories(item, max_rounds)
            if options:
                trajectories[(scene_id, ann_id, target_id)].extend(options)
                continue

        for dialogue_idx, dialogue in enumerate(item.get("dialogues") or []):
            current_type = str(dialogue.get("ambiguity_type", "unknown"))
            if ambiguity_type and ambiguity_type != "all" and current_type != ambiguity_type:
                continue
            rounds = []
            _append_unique_round(
                rounds,
                dialogue.get("input_query")
                or dialogue.get("query_before")
                or dialogue.get("degraded_query"),
            )
            _append_unique_round(
                rounds,
                dialogue.get("restored_query")
                or dialogue.get("query_after")
                or dialogue.get("restored_query_raw"),
            )
            if len(rounds) < 2:
                continue
            question = _clean_text(dialogue.get("generated_question"))
            negative_rounds = list(rounds)
            rejected_query = _negative_feedback_query(rounds[0], question)
            if rejected_query and len(negative_rounds) > 1:
                negative_rounds[1] = rejected_query
            trajectories[(scene_id, ann_id, target_id)].append({
                "queries": rounds[:max_rounds],
                "negative_queries": negative_rounds[:max_rounds] if rejected_query else [],
                "feedback": [0.0] + [1.0] * (min(len(rounds), max_rounds) - 1),
                "negative_feedback": [0.0] + [-1.0] * (min(len(rounds), max_rounds) - 1),
                "target_id": target_id,
                "trajectory_type": current_type,
                "trajectory_id": f"{current_type}:{dialogue_idx}",
                "source_ann_id": ann_id,
                "generated_question": question,
            })
    return dict(trajectories)


def _flat_round_trajectories(items, max_rounds):
    grouped = defaultdict(list)
    for item in items:
        try:
            round_id = int(item.get("round", -1))
        except (TypeError, ValueError):
            continue
        key = (
            str(item.get("scene_id", "")),
            _ann_key(item.get("ann_id", "")),
            int(item.get("target_id", -1)),
        )
        grouped[key].append((round_id, _clean_text(item.get("query"))))

    trajectories = defaultdict(list)
    for (scene_id, ann_id, target_id), rows in grouped.items():
        by_round = {}
        for round_id, query in rows:
            if query:
                by_round.setdefault(round_id, query)
        round_ids = sorted(by_round)
        # Repeated round-0 rows are independent samples, not a trajectory.
        if len(round_ids) < 2:
            continue
        queries = [by_round[round_id] for round_id in round_ids]
        trajectories[(scene_id, ann_id, target_id)].append({
            "queries": queries[:max_rounds],
            "negative_queries": [],
            "feedback": [0.0] * min(len(queries), max_rounds),
            "negative_feedback": [0.0] * min(len(queries), max_rounds),
            "target_id": target_id,
            "trajectory_type": "flat_rounds",
            "trajectory_id": "flat_rounds",
            "source_ann_id": ann_id,
            "generated_question": "",
        })
    return dict(trajectories)


def _prebuilt_trajectories(items, max_rounds, ambiguity_type=""):
    trajectories = defaultdict(list)
    for item in items:
        current_type = str(item.get("trajectory_type", item.get("ambiguity_type", "")))
        if ambiguity_type and ambiguity_type != "all" and current_type != ambiguity_type:
            continue
        scene_id = str(item.get("scene_id", ""))
        ann_id = _ann_key(item.get("source_ann_id", item.get("ann_id", "")))
        try:
            target_id = int(item.get("target_id", -1))
        except (TypeError, ValueError):
            continue
        queries = [_clean_text(value) for value in (item.get("queries") or [])]
        queries = [value for value in queries if value][:max_rounds]
        if not scene_id or target_id < 0 or not queries:
            continue

        option = dict(item)
        option["queries"] = queries
        option["negative_queries"] = [
            _clean_text(value)
            for value in (item.get("negative_queries") or [])[:max_rounds]
        ]
        option["feedback"] = [
            float(value) for value in (item.get("feedback") or [])[:max_rounds]
        ]
        option["negative_feedback"] = [
            float(value)
            for value in (item.get("negative_feedback") or [])[:max_rounds]
        ]
        option["candidate_sets"] = [
            sorted({int(value) for value in values if int(value) >= 0})
            for values in (item.get("candidate_sets") or [])[:max_rounds]
        ]
        option["target_id"] = target_id
        option["source_ann_id"] = ann_id
        option["trajectory_type"] = current_type
        option["trajectory_id"] = str(
            item.get("trajectory_id", current_type or "prebuilt")
        )
        trajectories[(scene_id, ann_id, target_id)].append(option)
    return dict(trajectories)


def load_online_trajectories(path, max_rounds=3, ambiguity_type=""):
    """Load real multi-round queries keyed by ``(scene_id, ann_id, target_id)``.

    Nested dialogue files become two-step trajectories from degraded to restored
    queries. Flat round files are accepted only when they contain at least two
    distinct round ids, preventing duplicated round-0 rows from masquerading as
    trajectories.
    """
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError(f"Trajectory file must contain a JSON list: {path}")
    if not items:
        return {}
    if any("queries" in item for item in items):
        return _prebuilt_trajectories(
            items, int(max_rounds), str(ambiguity_type or "")
        )
    if any(item.get("dialogues") for item in items):
        return _nested_dialogue_trajectories(items, int(max_rounds), str(ambiguity_type or ""))
    return _flat_round_trajectories(items, int(max_rounds))
