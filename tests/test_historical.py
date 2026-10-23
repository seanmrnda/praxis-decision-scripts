"""Execute original hard-label trainers against real, tiny local checkpoints."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from decision_moe import DecisionMoE
from test_model import tiny_julia
from test_pipeline import tiny_tokenizer

ROOT = Path(__file__).resolve().parents[1]


class HistoricalTests(unittest.TestCase):
    def test_v1_and_v2_original_training_clis(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / 'model'
            DecisionMoE(tiny_julia(), [1], 2, 2).save_pretrained(model)
            tiny_tokenizer().save_pretrained(model / 'tokenizer')
            for version in ['v1', 'v2']:
                source = ROOT / ('historical/v1/src' if version == 'v1' else 'src')
                out = root / version
                env = dict(os.environ, PYTHONPATH=str(source), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
                command = [sys.executable, str(ROOT / f'historical/{version}/scripts/finetune.py'),
                           '--model-dir', str(model), '--output-dir', str(out), '--epochs', '1',
                           '--batch-size', '2', '--grad-accum', '1', '--max-length', '128', '--head-length', '64',
                           '--train-file', str(ROOT / 'examples/train.jsonl'),
                           '--validation-file', str(ROOT / 'examples/validation.jsonl'),
                           '--calibration-file', str(ROOT / 'examples/calibration.jsonl'), '--device', 'cpu']
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                report = json.loads((out / 'training_report.json').read_text())
                self.assertEqual(report['status'], 'complete')
                self.assertGreater(report['optimizer_updates'], 0)
                self.assertIn('calibration_after', report)


if __name__ == '__main__':
    unittest.main()
