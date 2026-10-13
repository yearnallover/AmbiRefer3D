import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stage2_refactor.data.convert_dialogue_json_semantic import convert
from tools.validate_stage1_metadata import validate_rows as validate_stage1
from tools.validate_stage2_inputs import validate_rows as validate_stage2


class ReleaseContractsTest(unittest.TestCase):
    def test_mt_preserves_compatible_set_and_target(self):
        rows = [{'scene_id': 's', 'ann_id': '0', 'description': 'the chair',
                 'eval_type': 'mt', 'object_ids': [4, 5], 'true_target_id': 5}]
        self.assertEqual(validate_stage1(rows), {'s'})
        rows[0]['true_target_id'] = 9
        with self.assertRaises(ValueError):
            validate_stage1(rows)

    def test_converter_keeps_missed_target_and_aligned_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input.json'
            output = root / 'rounds.json'
            source.write_text(json.dumps([{'scene_id': 's', 'ann_id': '0', 'target_id': 99,
                'dialogues': [{'input_query': 'the chair', 'candidate_ids': [4, 5],
                'grounding_matches': [99], 'cand_stage1_scores': [0.4, 0.5],
                'cand_bboxes': [[4, 0, 0, 1, 1, 1, 5], [5, 0, 0, 1, 1, 1, 5]],
                'generated_question': 'Is it near the table?', 'is_ambiguous': True}]}]))
            convert(str(source), str(output), seed=3)
            row = json.loads(output.read_text())[0]
            self.assertEqual(row['target_index'], -1)
            self.assertEqual(set(row['cand_ids']), {4, 5})
            self.assertNotIn(99, row['cand_ids'])
            for candidate, bbox, score in zip(row['cand_ids'], row['cand_bboxes'], row['cand_stage1_scores']):
                self.assertEqual(candidate, bbox[0])
                self.assertAlmostEqual(score, candidate / 10)
            before = output.read_text()
            convert(str(source), str(output), seed=3)
            self.assertEqual(before, output.read_text())

    def test_stage2_requires_real_cached_features(self):
        with tempfile.TemporaryDirectory() as directory:
            np.savez(Path(directory) / 's.npz', obj_ids=np.array([4, 5]), feats=np.ones((2, 768)))
            row = {'scene_id': 's', 'query': 'the chair', 'cand_ids': [4, 5],
                   'target_id': 99, 'target_index': -1}
            self.assertEqual(validate_stage2([row], directory)['initial_target_survival'], 0)
            row['cand_ids'] = [4, 6]
            with self.assertRaises(ValueError):
                validate_stage2([row], directory)


if __name__ == '__main__':
    unittest.main()
