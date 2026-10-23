"""Verify run lineage, immutable outputs and detection of data tampering."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from reproduce import commands, prepare_stage, ROOT
from verify_reproduction import verify_profile


class ReproductionTests(unittest.TestCase):
    def test_training_recipe_and_parent_lineage(self):
        work = Path('/tmp/reproduction-plan')
        for stage, epochs, rows, lr in [('train-v1', [2], [6000], ['1e-5']),
                                        ('train-v2', [4, 3], [6000, 8000], ['5e-6', '5e-6'])]:
            calls = [(argv, runtime) for argv, runtime in commands(stage, work) if Path(argv[1]).name == 'finetune.py']
            self.assertEqual(len(calls), len(epochs))
            for (argv, runtime), epoch, count, rate in zip(calls, epochs, rows, lr):
                value = lambda flag: argv[argv.index(flag) + 1]
                self.assertEqual(int(value('--epochs')), epoch)
                self.assertEqual(value('--lr'), rate)
                self.assertEqual(int(value('--batch-size')) * int(value('--grad-accum')), 16)
                self.assertIn('--fp16', argv)
                self.assertIn('--data-parallel', argv)
                expected_parent = work / ('praxis-base' if stage == 'train-v1' else 'runs/v1/defense-praxis')
                self.assertEqual(Path(value('--model-dir')), expected_parent)
                expected_data = 'v1' if count == 6000 else 'v2'
                self.assertEqual(Path(value('--train-file')).parent, work / 'data' / expected_data)
                self.assertEqual(runtime, 'v1' if stage == 'train-v1' else 'current')

    def test_policy_replay_uses_archived_policy(self):
        argv, _ = commands('prepare-v3', Path('/tmp/reproduction-plan'))[0]
        self.assertEqual(Path(argv[argv.index('--policy-file') + 1]), ROOT / 'reproducibility/archived_teacher_policy.json')

    def test_completed_or_partial_data_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            (work / 'data/v1').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
                prepare_stage('prepare-v1', work)

    def test_verifier_rejects_tampered_actual_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            row = {'id': 'one', 'split_groups': ['wiki:one'], 'label': 'a',
                   'options': {'a': 'A', 'b': 'B'}, 'source_dataset': 'test'}
            content = json.dumps(row) + '\n'
            path = folder / 'train.jsonl'
            path.write_text(content)
            expected = {'files': {'train.jsonl': {'sha256': hashlib.sha256(content.encode()).hexdigest(), 'rows': 1}}}
            (folder / 'data_manifest.json').write_text(json.dumps(expected))
            self.assertEqual(verify_profile(folder, expected)['train.jsonl']['rows'], 1)
            path.write_text(content + '\n')
            with self.assertRaisesRegex(ValueError, 'Paper partition differs'):
                verify_profile(folder, expected)


if __name__ == '__main__':
    unittest.main()
