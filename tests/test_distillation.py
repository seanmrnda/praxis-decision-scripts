"""Verify masked KL, exact teacher provenance and a real training CLI update."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import tempfile
import unittest

import torch
from decision_moe import DecisionMoE
from decision_moe.distillation import attach_teachers, distillation_kl, file_sha256
from test_model import tiny_julia
from test_pipeline import tiny_tokenizer
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from distillation_results import build_comparison
from prepare_distillation import unused_groups

ROOT = Path(__file__).resolve().parents[1]


class DistillationTests(unittest.TestCase):
    def test_fresh_groups_exclude_partial_evidence_overlap(self):
        rows = [{'id': 'overlap', 'split_groups': ['wiki:old', 'claim:new']},
                {'id': 'unused', 'split_groups': ['wiki:new', 'claim:new']}]
        self.assertEqual([r['id'] for r in unused_groups(rows, {'wiki:old'})], ['unused'])

    def test_publication_rejects_unmatched_control_or_incomplete_windows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'teacher_policy.json').write_text('{}')
            recipe = {'planned_optimizer_windows': 500, 'distill_weight': .5, 'distill_temperature': 2.,
                      'teacher_policy_sha256': file_sha256(root / 'teacher_policy.json'), 'teacher_model_sha256': {'v1': 'hash'}}
            receipt = {'train_sha256': 'trainhash', 'policy_sha256': recipe['teacher_policy_sha256'],
                       'teacher_model_sha256': recipe['teacher_model_sha256']}
            manifest = {'planned_recipe': recipe, 'files': {'train.jsonl': {'sha256': 'trainhash'}}}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            matched = {key: 1 for key in ['model_dir', 'epochs', 'batch_size', 'grad_accum', 'lr', 'head_lr', 'seed', 'fp16',
                       'data_parallel', 'max_length', 'head_length', 'train_file', 'validation_file', 'calibration_file',
                       'brier_weight', 'aux_weight', 'weight_decay', 'warmup_ratio']}
            candidate = {'status': 'complete', 'optimizer_updates': 495, 'skipped_updates': 5,
                         'args': {**matched, 'distill_weight': .5, 'distill_temperature': 2.},
                         'distillation': {'teacher_receipt': receipt}}
            control = {'status': 'complete', 'optimizer_updates': 495, 'skipped_updates': 5, 'args': dict(matched)}
            (root / 'training_report.json').write_text(json.dumps(candidate))
            (root / 'control_c').mkdir()
            control['optimizer_updates'] = 494
            (root / 'control_c/training_report.json').write_text(json.dumps(control))
            with self.assertRaisesRegex(ValueError, 'all planned windows'):
                build_comparison(root, root / 'manifest.json')
            control['optimizer_updates'] = 495
            control['args']['lr'] = 2
            (root / 'control_c/training_report.json').write_text(json.dumps(control))
            with self.assertRaisesRegex(ValueError, 'Unmatched control recipe'):
                build_comparison(root, root / 'manifest.json')

    def test_kl_matches_distribution_and_ignores_padding_gradients(self):
        logits = torch.tensor([[2., 0., -99.], [0., 1., 2.]], requires_grad=True)
        mask = torch.tensor([[True, True, False], [True, True, True]])
        teacher = logits.detach().masked_fill(~mask, float('-inf')).softmax(-1)
        loss = distillation_kl(logits, teacher, mask, 2.)
        self.assertAlmostEqual(loss.item(), 0., places=6)
        teacher[0, :2] = torch.tensor([.2, .8])
        loss = distillation_kl(logits, teacher, mask, 2.)
        self.assertGreater(loss.item(), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[0, 2].item(), 0.)
        self.assertGreater(logits.grad[0, 0].item(), 0.)

    def test_provenance_order_and_gold_retention(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = {'type': 'choice', 'id': 'x', 'task': 'relation', 'source_dataset': 'source', 'source_id': 'doc',
                   'state': 'state', 'question': 'route', 'options': {'a': 'A', 'b': 'B'}, 'label': 'a'}
            (root / 'train.jsonl').write_text(json.dumps(row) + '\n')
            teacher = {'id': 'x', 'gold': 'a', 'task': 'relation', 'source_dataset': 'source', 'source_id': 'doc',
                       'probabilities': {'b': .8, 'a': .2}}
            (root / 'predictions.jsonl').write_text(json.dumps(teacher) + '\n')
            receipt = {'role': 'train', 'train_sha256': file_sha256(root / 'train.jsonl'),
                       'predictions_sha256': file_sha256(root / 'predictions.jsonl')}
            (root / 'receipt.json').write_text(json.dumps(receipt))
            joined, _ = attach_teachers([row], root / 'train.jsonl', root / 'predictions.jsonl', root / 'receipt.json')
            self.assertEqual(joined[0]['teacher_probs'], [.2, .8])
            self.assertEqual(joined[0]['label'], 'a')
            self.assertNotIn('teacher_probs', row)
            (root / 'train.jsonl').write_text(json.dumps(row) + '\n\n')
            with self.assertRaises(ValueError):
                attach_teachers([row], root / 'train.jsonl', root / 'predictions.jsonl', root / 'receipt.json')

    def test_real_distillation_cli_updates_and_calibrates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = DecisionMoE(tiny_julia(), [1], 3, 2)
            before = model.scorer[-1].weight.detach().clone()
            model.save_pretrained(root / 'model')
            tiny_tokenizer().save_pretrained(root / 'model/tokenizer')
            for role, count in [('train', 4), ('validation', 2), ('calibration', 2)]:
                rows = [{'type': 'choice', 'id': role + str(i), 'task': 'relation', 'source_dataset': 'test',
                         'source_id': role + str(i), 'state': role + ' state ' + str(i), 'question': 'route',
                         'options': {'a': 'A', 'b': 'B'}, 'label': 'a' if i % 2 else 'b'} for i in range(count)]
                (root / (role + '.jsonl')).write_text(''.join(json.dumps(row) + '\n' for row in rows))
                if role == 'train':
                    teachers = [{**{key: row[key] for key in ['id', 'task', 'source_dataset', 'source_id']},
                                 'gold': row['label'], 'probabilities': {'a': .3, 'b': .7}} for row in rows]
                    (root / 'teacher.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in teachers))
            env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            for arm in ['v1', 'control', 'v2']:
                shutil.copytree(root / 'model', root / 'teachers' / arm)
            (root / 'policy.json').write_text(json.dumps({'arms': ['v1', 'control', 'v2'], 'weights': {'relation': [.6, .4, 0.]}}))
            cached = subprocess.run([sys.executable, str(ROOT / 'scripts/cache_teachers.py'),
                                     '--teacher-root', str(root / 'teachers'), '--train-file', str(root / 'train.jsonl'),
                                     '--policy-file', str(root / 'policy.json'), '--output-dir', str(root / 'cache'), '--device', 'cpu'],
                                    env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(cached.returncode, 0, cached.stdout + cached.stderr)
            receipt = json.loads((root / 'cache/teacher_manifest.json').read_text())
            self.assertEqual(receipt['teacher_model_sha256']['v1'], file_sha256(root / 'model/model.safetensors'))
            command = [sys.executable, str(ROOT / 'scripts/finetune.py'), '--model-dir', str(root / 'model'),
                       '--train-file', str(root / 'train.jsonl'), '--validation-file', str(root / 'validation.jsonl'),
                       '--calibration-file', str(root / 'calibration.jsonl'), '--output-dir', str(root / 'student'),
                       '--teacher-predictions', str(root / 'cache/teacher_predictions.jsonl'), '--teacher-manifest', str(root / 'cache/teacher_manifest.json'),
                       '--epochs', '1', '--batch-size', '2', '--grad-accum', '2', '--max-length', '128', '--head-length', '64', '--device', 'cpu']
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((root / 'student/training_report.json').read_text())
            self.assertEqual(report['optimizer_updates'], 1)
            self.assertEqual(report['distillation']['teacher_receipt'], receipt)
            trained = DecisionMoE.from_pretrained(root / 'student')
            self.assertFalse(torch.equal(before, trained.scorer[-1].weight))
            self.assertTrue(all(torch.isfinite(p).all() for p in trained.parameters()))
