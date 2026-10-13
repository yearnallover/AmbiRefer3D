import json
import random
import warnings
import clip
import torch
import numpy as np
from m3drefclip.data.dataset.general_dataset import GeneralDataset
from m3drefclip.data.online_trajectory import load_online_trajectories
from m3drefclip.data.counterfactual_pairing import pack_pair_aware_indices


def _normalize_text(value):
    return " ".join(str(value or "").strip().lower().split())


def _deduplicate_trajectories(options):
    unique = []
    seen = set()
    for option in options:
        key = (
            str(option.get("source_ann_id", "")),
            str(option.get("trajectory_id", "")),
            tuple(_normalize_text(query) for query in option.get("queries") or []),
        )
        if key not in seen:
            seen.add(key)
            unique.append(option)
    return unique


class Multi3DRefer(GeneralDataset):
    def __int__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_language_data(self):
        # create a map, skip invalid labels to make the final semantic labels consecutive
        filtered_label_map = {}
        for i, valid_id in enumerate(self.data_cfg.scene_metadata.valid_semantic_mapping):
            filtered_label_map[valid_id] = i

        # load language data
        file_path = getattr(self.data_cfg.lang_metadata, f"{self.split}_language_data")
        with open(file_path, "r") as f:
            raw_data = json.load(f)

        trajectory_cfg = getattr(self.data_cfg, "online_trajectory", None)
        self.online_trajectory_enabled = bool(
            trajectory_cfg is not None and getattr(trajectory_cfg, "enabled", False)
        )
        self.online_trajectory_max_rounds = int(
            getattr(trajectory_cfg, "max_rounds", 3) if trajectory_cfg is not None else 3
        )
        self.online_trajectory_strict_alignment = bool(
            getattr(trajectory_cfg, "strict_query_alignment", True)
            if trajectory_cfg is not None else True
        )
        self.online_trajectory_repair_target_set = bool(
            getattr(trajectory_cfg, "repair_target_set", True)
            if trajectory_cfg is not None else True
        )
        self.online_trajectory_require_train_match = bool(
            getattr(trajectory_cfg, "require_train_match", False)
            if trajectory_cfg is not None else False
        )
        self.online_trajectories = {}
        self.online_trajectories_by_query = {}
        self.online_trajectories_by_question = {}
        if self.online_trajectory_enabled:
            trajectory_path = str(getattr(trajectory_cfg, "path", "") or "")
            if not trajectory_path:
                raise ValueError("data.online_trajectory.path is required when online trajectory training is enabled.")
            self.online_trajectories = load_online_trajectories(
                trajectory_path,
                max_rounds=self.online_trajectory_max_rounds,
                ambiguity_type=str(getattr(trajectory_cfg, "ambiguity_type", "") or ""),
            )
            if not self.online_trajectories:
                raise ValueError(
                    f"No valid multi-round trajectories found in {trajectory_path}. "
                    "A trajectory needs at least two distinct query rounds."
                )
            for (scene_id, _, target_id), options in self.online_trajectories.items():
                for option in options:
                    queries = option.get("queries") or []
                    if not queries:
                        continue
                    query_key = _normalize_text(queries[0])
                    question_key = _normalize_text(option.get("generated_question", ""))
                    self.online_trajectories_by_query.setdefault(
                        (scene_id, int(target_id), query_key), []
                    ).append(option)
                    if question_key:
                        self.online_trajectories_by_question.setdefault(
                            (scene_id, int(target_id), query_key, question_key), []
                        ).append(option)

        self.language_data = {}
        self.scene_data_split_by_scene = {}
        scene_ids = {}
        alignment_stats = {
            "samples": 0,
            "matched": 0,
            "unmatched": 0,
            "rejected_direct": 0,
            "repaired_target_sets": 0,
        }
        for item in raw_data:
            alignment_stats["samples"] += 1
            scene_ids[item["scene_id"]] = True
            scene_data_split = str(item.get("scene_data_split") or self.split)
            if scene_data_split not in {"train", "val", "test"}:
                raise ValueError(f"invalid scene_data_split {scene_data_split!r}")
            previous_scene_split = self.scene_data_split_by_scene.setdefault(
                item["scene_id"], scene_data_split
            )
            if previous_scene_split != scene_data_split:
                raise ValueError(
                    f"scene {item['scene_id']} has conflicting scene_data_split values"
                )
            if item["scene_id"] not in self.language_data:
                self.language_data[item["scene_id"]] = []

            object_name = item["object_name"].replace("_", " ")
            ann_id = item["ann_id"]
            true_target_id = int(item.get("true_target_id", -1))
            source_ann_id = item.get(
                "trajectory_source_ann_id", item.get("source_ann_id", ann_id)
            )
            direct_trajectory_options = self.online_trajectories.get(
                (str(item["scene_id"]), str(int(source_ann_id)), true_target_id), []
            ) if self.online_trajectory_enabled else []
            query_key = _normalize_text(item["description"])
            question_key = _normalize_text(item.get("generated_question", ""))
            trajectory_options = []
            if self.online_trajectory_enabled:
                exact_direct = [
                    option
                    for option in direct_trajectory_options
                    if option.get("queries")
                    and _normalize_text(option["queries"][0]) == query_key
                    and (
                        not question_key
                        or _normalize_text(option.get("generated_question", ""))
                        == question_key
                    )
                ]
                alignment_stats["rejected_direct"] += (
                    len(direct_trajectory_options) - len(exact_direct)
                )
                trajectory_options.extend(exact_direct)
                if not trajectory_options and question_key:
                    trajectory_options.extend(
                        self.online_trajectories_by_question.get(
                            (
                                str(item["scene_id"]), true_target_id,
                                query_key, question_key,
                            ),
                            [],
                        )
                    )
                if not trajectory_options:
                    trajectory_options.extend(
                        self.online_trajectories_by_query.get(
                            (str(item["scene_id"]), true_target_id, query_key), []
                        )
                    )
                if not trajectory_options and not self.online_trajectory_strict_alignment:
                    trajectory_options.extend(direct_trajectory_options)
                trajectory_options = _deduplicate_trajectories(trajectory_options)

            positive_key = str(getattr(self.data_cfg, "positive_object_key", "object_ids"))
            object_ids = np.array(item.get(positive_key, item["object_ids"]), dtype=np.int16)
            if (
                self.online_trajectory_enabled
                and trajectory_options
                and true_target_id >= 0
                and true_target_id not in object_ids
            ):
                if self.online_trajectory_repair_target_set:
                    object_ids = np.unique(
                        np.append(object_ids, np.int16(true_target_id))
                    ).astype(np.int16)
                    alignment_stats["repaired_target_sets"] += 1
                else:
                    trajectory_options = []

            if trajectory_options:
                alignment_stats["matched"] += 1
            elif self.online_trajectory_enabled:
                alignment_stats["unmatched"] += 1
            self.language_data[item["scene_id"]].append(
                {
                    "object_ids": object_ids,
                    "object_name": object_name,
                    "ann_id": ann_id,
                    "eval_type": item["eval_type"],
                    "query_text": item["description"].strip(),
                    "clip_tokens": clip.tokenize(item["description"].strip(), truncate=True)[0],
                    "trajectory_options": trajectory_options,
                    "true_target_id": true_target_id,
                    "diagnostic_sample_id": str(item.get("diagnostic_sample_id") or ""),
                    "counterfactual_pair_id": str(item.get("counterfactual_pair_id") or ""),
                    "sample_weight": float(item.get("recommended_loss_weight", 1.0)),
                    "is_formal_target_leaf": bool(item.get("is_formal_target_leaf", False)),
                }
            )
        if self.online_trajectory_enabled:
            if (
                self.split == "train"
                and self.online_trajectory_require_train_match
                and alignment_stats["unmatched"]
            ):
                raise ValueError(
                    "Strict online trajectory alignment failed: "
                    f"{alignment_stats['unmatched']}/{alignment_stats['samples']} train "
                    "language rows have no trajectory with the same scene, target, and "
                    "initial query. Build aligned trajectory metadata or disable "
                    "data.online_trajectory.require_train_match."
                )
            warnings.warn(
                "Online trajectory alignment "
                f"({self.split}): matched={alignment_stats['matched']}/"
                f"{alignment_stats['samples']}, unmatched={alignment_stats['unmatched']}, "
                f"rejected_direct={alignment_stats['rejected_direct']}, "
                f"repaired_target_sets={alignment_stats['repaired_target_sets']}",
                stacklevel=2,
            )
        self.scene_ids = list(scene_ids.keys())

    def _pack_data_to_chunks(self):
        pair_cfg = getattr(self.data_cfg, "counterfactual_pair_packing", None)
        if pair_cfg is None or not bool(getattr(pair_cfg, "enabled", False)):
            return super()._pack_data_to_chunks()
        chunk_size = int(self.data_cfg.chunk_size)
        self.chunk_lang_indices = np.empty(shape=(0, chunk_size), dtype=np.uint16)
        self.chunk_scene_indices = np.empty(shape=0, dtype=np.uint16)
        base_seed = str(getattr(pair_cfg, "seed", "3dvg-counterfactual-pairs-v1"))
        for scene_index, scene_id in enumerate(self.scene_ids):
            if self.split == "train":
                rng = random.Random(int(np.random.randint(0, 2**31 - 1)))
            else:
                rng = random.Random(f"{base_seed}|{self.split}|{scene_id}")
            chunks = pack_pair_aware_indices(self.language_data[scene_id], chunk_size, rng)
            if not chunks:
                continue
            self.chunk_lang_indices = np.concatenate(
                (self.chunk_lang_indices, np.asarray(chunks, dtype=np.uint16)), axis=0
            )
            self.chunk_scene_indices = np.concatenate(
                (self.chunk_scene_indices, np.full(len(chunks), scene_index, dtype=np.uint16)), axis=0
            )

    def __getitem__(self, index):
        data_dict = super().__getitem__(index)  # get scene data from parent class

        # prepare language data
        scene_id = self.scene_ids[self.chunk_scene_indices[index]]
        language_data_indices = self.chunk_lang_indices[index]
        language_data_in_scene = self.language_data[scene_id]
        num_language_data_in_scene = len(language_data_in_scene)

        data_dict["ann_id"] = np.empty(shape=self.data_cfg.chunk_size, dtype=np.int32)
        data_dict["object_id"] = np.empty(shape=self.data_cfg.chunk_size, dtype=np.int16)
        data_dict["true_target_id"] = np.full(
            shape=self.data_cfg.chunk_size, fill_value=-1, dtype=np.int16
        )
        data_dict["sample_weight"] = np.ones(shape=self.data_cfg.chunk_size, dtype=np.float32)
        data_dict["is_formal_target_leaf"] = np.zeros(shape=self.data_cfg.chunk_size, dtype=bool)
        data_dict["diagnostic_sample_id"] = [""] * self.data_cfg.chunk_size
        data_dict["counterfactual_pair_id"] = [""] * self.data_cfg.chunk_size

        data_dict["gt_target_obj_id_mask"] = np.zeros(
            shape=(self.data_cfg.chunk_size, data_dict["gt_aabb_obj_ids"].shape[0]), dtype=bool
        )

        data_dict["clip_tokens"] = torch.empty(size=(self.data_cfg.chunk_size, 77), dtype=torch.int32)
        data_dict["query_text"] = [""] * self.data_cfg.chunk_size
        if self.online_trajectory_enabled:
            data_dict["trajectory_clip_tokens"] = torch.zeros(
                size=(self.data_cfg.chunk_size, self.online_trajectory_max_rounds, 77), dtype=torch.int32
            )
            data_dict["trajectory_round_mask"] = torch.zeros(
                size=(self.data_cfg.chunk_size, self.online_trajectory_max_rounds), dtype=torch.bool
            )
            data_dict["trajectory_feedback"] = torch.zeros(
                size=(self.data_cfg.chunk_size, self.online_trajectory_max_rounds), dtype=torch.float32
            )
            data_dict["negative_trajectory_clip_tokens"] = torch.zeros(
                size=(self.data_cfg.chunk_size, self.online_trajectory_max_rounds, 77), dtype=torch.int32
            )
            data_dict["negative_trajectory_round_mask"] = torch.zeros(
                size=(self.data_cfg.chunk_size, self.online_trajectory_max_rounds), dtype=torch.bool
            )
            data_dict["negative_trajectory_feedback"] = torch.zeros(
                size=(self.data_cfg.chunk_size, self.online_trajectory_max_rounds), dtype=torch.float32
            )
            data_dict["trajectory_target_obj_id"] = torch.full(
                size=(self.data_cfg.chunk_size,), fill_value=-1, dtype=torch.int16
            )
            data_dict["trajectory_source_ann_id"] = torch.full(
                size=(self.data_cfg.chunk_size,), fill_value=-1, dtype=torch.int32
            )
            data_dict["trajectory_id"] = [""] * self.data_cfg.chunk_size
            data_dict["trajectory_query_texts"] = [
                [""] * self.online_trajectory_max_rounds
                for _ in range(self.data_cfg.chunk_size)
            ]
            data_dict["negative_trajectory_query_texts"] = [
                [""] * self.online_trajectory_max_rounds
                for _ in range(self.data_cfg.chunk_size)
            ]
            data_dict["trajectory_candidate_gt_mask"] = np.zeros(
                (
                    self.data_cfg.chunk_size,
                    self.online_trajectory_max_rounds,
                    data_dict["gt_aabb_obj_ids"].shape[0],
                ),
                dtype=bool,
            )
        data_dict["eval_type"] = []
        for i, index in enumerate(language_data_indices):
            real_idx = index % num_language_data_in_scene  # pad the last chunk
            data = language_data_in_scene[real_idx]
            data_dict["ann_id"][i] = data["ann_id"]
            data_dict["object_id"][i] = 0  # dummy value for multi3drefer dataset
            data_dict["true_target_id"][i] = data["true_target_id"]
            data_dict["sample_weight"][i] = data["sample_weight"]
            data_dict["is_formal_target_leaf"][i] = data["is_formal_target_leaf"]
            data_dict["diagnostic_sample_id"][i] = data["diagnostic_sample_id"]
            data_dict["counterfactual_pair_id"][i] = data["counterfactual_pair_id"]
            data_dict["gt_target_obj_id_mask"][i] = np.in1d(data_dict["gt_aabb_obj_ids"], data["object_ids"])
            data_dict["clip_tokens"][i] = data["clip_tokens"]
            data_dict["query_text"][i] = data["query_text"]
            data_dict["eval_type"].append(data["eval_type"])
            if self.online_trajectory_enabled and data["trajectory_options"]:
                option_idx = np.random.randint(len(data["trajectory_options"])) if self.split == "train" else 0
                trajectory = data["trajectory_options"][option_idx]
                queries = trajectory["queries"][:self.online_trajectory_max_rounds]
                feedback = trajectory.get("feedback") or []
                for round_idx, query in enumerate(queries):
                    data_dict["trajectory_query_texts"][i][round_idx] = query
                    data_dict["trajectory_clip_tokens"][i, round_idx] = clip.tokenize(
                        query, truncate=True
                    )[0]
                    data_dict["trajectory_round_mask"][i, round_idx] = True
                    if round_idx < len(feedback):
                        data_dict["trajectory_feedback"][i, round_idx] = float(feedback[round_idx])
                negative_queries = trajectory.get("negative_queries") or []
                negative_feedback = trajectory.get("negative_feedback") or []
                for round_idx, query in enumerate(
                    negative_queries[:self.online_trajectory_max_rounds]
                ):
                    data_dict["negative_trajectory_query_texts"][i][round_idx] = query
                    data_dict["negative_trajectory_clip_tokens"][i, round_idx] = clip.tokenize(
                        query, truncate=True
                    )[0]
                    data_dict["negative_trajectory_round_mask"][i, round_idx] = True
                    if round_idx < len(negative_feedback):
                        data_dict["negative_trajectory_feedback"][i, round_idx] = float(
                            negative_feedback[round_idx]
                        )
                target_id = int(trajectory.get("target_id", data["true_target_id"]))
                data_dict["trajectory_target_obj_id"][i] = target_id
                data_dict["trajectory_source_ann_id"][i] = int(
                    trajectory.get("source_ann_id", data["ann_id"])
                )
                data_dict["trajectory_id"][i] = str(trajectory.get("trajectory_id", ""))
                candidate_sets = trajectory.get("candidate_sets") or []
                for round_idx in range(min(len(queries), self.online_trajectory_max_rounds)):
                    if round_idx < len(candidate_sets):
                        candidate_ids = candidate_sets[round_idx]
                    elif round_idx == 0:
                        candidate_ids = data["object_ids"]
                    else:
                        candidate_ids = [target_id]
                    data_dict["trajectory_candidate_gt_mask"][i, round_idx] = np.isin(
                        data_dict["gt_aabb_obj_ids"],
                        np.asarray(candidate_ids, dtype=np.int64),
                    )
        return data_dict
