"""Check candidate alignment, valid blending, and development/holdout isolation."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from experiment_committee import ARMS, align, check_disjoint, fit_policy, load_predictions, mix


class CommitteeTests(unittest.TestCase):
    def rows(self):
        row = {'id': 'question', 'task': 'relation', 'source_dataset': 'source',
               'source_id': 'document', 'gold': 'a', 'prediction': 'a',
               'probabilities': {'a': .9, 'b': .1}, 'scores': {'a': 2., 'b': 0.}}
        arms = {arm: {row['id']: copy.deepcopy(row)} for arm in ARMS}
        arms['control']['question']['probabilities'] = {'b': .3, 'a': .7}
        arms['v2']['question']['probabilities'] = {'a': .2, 'b': .8}
        return arms

    def test_alignment_and_blending_use_candidate_identity(self):
        arms = self.rows()
        blended = mix(arms, {'relation': [.5, .5, 0.]})[0]
        self.assertAlmostEqual(blended['probabilities']['a'], .8)
        self.assertEqual(blended['prediction'], 'a')
        self.assertNotIn('scores', blended)
        arms['control']['question']['probabilities'] = {'a': .7, 'c': .3}
        with self.assertRaises(ValueError):
            align(arms)

    def test_policy_selection_and_runtime_do_not_route_using_gold(self):
        arms = self.rows()
        weights, selection = fit_policy(arms)
        self.assertEqual(weights['relation'], [1., 0., 0.])
        self.assertEqual(selection['relation']['trials'], 66)
        expected = mix(arms, weights)[0]['probabilities']
        for rows in arms.values():
            rows['question']['gold'] = 'b'
        self.assertEqual(mix(arms, weights)[0]['probabilities'], expected)

    def test_reject_metadata_and_document_leakage(self):
        arms = self.rows()
        altered = copy.deepcopy(arms)
        altered['v2']['question']['task'] = 'another-task'
        with self.assertRaises(ValueError):
            align(altered)
        heldout = {arm: {'different-id': {**row, 'id': 'different-id'}}
                   for arm, rows in arms.items() for row in rows.values()}
        with self.assertRaises(ValueError):
            check_disjoint(arms, heldout)

    def test_reject_invalid_probability_files(self):
        row = self.rows()['v1']['question']
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'predictions.jsonl'
            for probabilities in ({'a': float('nan'), 'b': .1}, {'a': .1, 'b': .1}):
                path.write_text(json.dumps({**row, 'probabilities': probabilities}))
                with self.assertRaises(ValueError):
                    load_predictions(path)
            path.write_text(json.dumps(row) + '\n' + json.dumps(row))
            with self.assertRaises(ValueError):
                load_predictions(path)
