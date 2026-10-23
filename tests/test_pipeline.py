"""Exercise the CLIs using a local Julia-format checkpoint and tokenizer."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from safetensors.torch import load_file
from decision_moe.data import DecisionCollator, marker_sequence
from test_model import tiny_julia

ROOT = Path(__file__).resolve().parents[1]


def tiny_tokenizer():
    vocab = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3, "[MASK]": 4,
             "choice": 5, "question": 6, ":": 7, "route": 8,
             "A": 9, "B": 10, "state": 11}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", cls_token="[CLS]",
                                   sep_token="[SEP]", unk_token="[UNK]", mask_token="[MASK]")


class PipelineTests(unittest.TestCase):
    def test_marker_serialization_and_overflow(self):
        tokenizer = tiny_tokenizer()
        row = {"state": "state", "question": "route", "options": {"a": "A", "b": "B"}}
        item = marker_sequence(tokenizer, row, 128, 64)
        self.assertEqual(item["ids"], [1, 5, 6, 7, 8, 2, 4, 9, 4, 10, 2, 11, 2])
        self.assertEqual(item["markers"], [6, 8])
        batch = DecisionCollator(tokenizer, 128, 64, labeled=False)([row, row])
        self.assertEqual(batch["input_ids"].shape[0], 2)  # One sequence per question.
        self.assertTrue(torch.all(batch["input_ids"].gather(1, batch["marker_pos"]) == 4))
        for invalid in (dict(row, state="[MASK] injected"), dict(row, state="state " * 150),
                        dict(row, options={"a": "A " * 49, "b": "B"}),
                        dict(row, question="route " * 64)):
            with self.assertRaises(ValueError):
                marker_sequence(tokenizer, invalid, 128, 64)

    def test_convert_train_calibrate_predict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            dense = tiny_julia()
            dense.temperature.copy_(torch.tensor([1.2, 1.6, 0.8]))
            dense.save_pretrained(base)
            tiny_tokenizer().save_pretrained(base / "tokenizer")
            env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            def run(script, *args):
                result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
                                        cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout
            converted = root / "converted"
            trained = root / "trained"
            run("convert_julia.py", "--base-model", base, "--output-dir", converted,
                "--num-experts", 3, "--top-k", 2, "--moe-layers", "all", "--model-name", "praxis")
            self.assertEqual(json.loads((converted / 'config.json').read_text())['model_name'], 'praxis')
            saved = load_file(str(converted / "model.safetensors"))
            for key, value in dense.state_dict().items():
                if not key.startswith("encoder.layers.") or ".mlp." not in key:
                    torch.testing.assert_close(saved[key], value, atol=0, rtol=0)
            run("finetune.py", "--model-dir", converted,
                "--train-file", ROOT / "examples/train.jsonl",
                "--validation-file", ROOT / "examples/validation.jsonl",
                "--calibration-file", ROOT / "examples/calibration.jsonl",
                "--output-dir", trained, "--epochs", 1, "--batch-size", 2,
                "--grad-accum", 2, "--max-length", 128, "--head-length", 64, "--device", "cpu",
                "--model-name", "defense-praxis", "--save-each-epoch")
            self.assertEqual(json.loads((trained / 'config.json').read_text())['model_name'], 'defense-praxis')
            self.assertTrue((trained / 'epoch-001/model.safetensors').is_file())
            self.assertEqual(json.loads((trained / 'progress.json').read_text())['status'], 'complete')
            report = json.loads((trained / "training_report.json").read_text())
            self.assertEqual(set(report["temperatures"]), {"choice", "score", "noul"})
            self.assertTrue(all(0.05 <= value <= 20 for value in report["temperatures"].values()))
            self.assertLessEqual(report["calibration_after"]["nll"], report["calibration_before"]["nll"] + 1e-5)
            output = root / "predictions.jsonl"
            run("predict.py", "--model-dir", trained, "--input-file", ROOT / "examples/requests.jsonl",
                "--output-file", output, "--max-length", 128, "--head-length", 64,
                "--batch-size", 2, "--device", "cpu")
            predictions = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([p["id"] for p in predictions], ["route", "refund", "urgency"])
            self.assertIn(predictions[0]["choice"], ["billing", "technical"])
            self.assertAlmostEqual(sum(predictions[0]["probabilities"].values()), 1, places=5)
            self.assertTrue(0 <= predictions[1]["noul"] <= 1)
            self.assertTrue(0 <= predictions[2]["score"] <= 2)


if __name__ == "__main__":
    unittest.main()
