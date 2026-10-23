"""Check transitive evidence dependence and reject mismatched paired metadata."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_uncertainty import components, component_interval


class UncertaintyTests(unittest.TestCase):
    def test_transitive_shared_groups_form_one_component(self):
        groups = components([{'id': 'a', 'split_groups': ['article:1']},
                             {'id': 'b', 'split_groups': ['article:1', 'article:2']},
                             {'id': 'c', 'split_groups': ['article:2']},
                             {'id': 'd', 'split_groups': ['article:3']}])
        self.assertEqual(groups['a'], groups['c'])
        self.assertNotEqual(groups['a'], groups['d'])

    def test_identical_predictions_have_zero_difference_and_interval(self):
        row = {'id': 'a', 'gold': 'yes', 'task': 'test', 'source_dataset': 'test',
               'source_id': 'doc', 'probabilities': {'yes': .6, 'no': .4}}
        result = component_interval({'a': row}, {'a': row}, {'a': 'group'})
        self.assertEqual(result['difference'], 0)
        self.assertEqual(result['connected_evidence_group_bootstrap_95'], [0, 0])
        with self.assertRaisesRegex(ValueError, 'metadata'):
            component_interval({'a': row}, {'a': dict(row, gold='no')}, {'a': 'group'})


if __name__ == '__main__':
    unittest.main()
