"""Offline teacher alignment and masked candidate KL for decision distillation."""
import hashlib
import json
import math
from pathlib import Path

import torch
from .data import DecisionCollator, candidates


def file_sha256(path):
    """Hash an artifact without materializing model-sized files in memory."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def attach_teachers(rows, train_file, predictions_file, manifest_file):
    """Attach probabilities by candidate identity while retaining gold targets.

    Receipt binds predictions to exact training bytes. Metadata and exact ID
    set must match; prediction order/candidate map order need not match.
    Returns new row dictionaries and receipt, leaving original rows unchanged.
    """
    receipt = json.loads(Path(manifest_file).read_text())
    if receipt['train_sha256'] != file_sha256(train_file) or receipt['predictions_sha256'] != file_sha256(predictions_file):
        raise ValueError('Teacher receipt does not match training/cache bytes')
    if receipt.get('role') != 'train':
        raise ValueError('Teacher cache must be generated for the training role')
    cached = {}
    for line in Path(predictions_file).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row['id'] in cached:
            raise ValueError('Duplicate teacher question ID')
        cached[row['id']] = row
    if len({r['id'] for r in rows}) != len(rows) or set(cached) != {r['id'] for r in rows}:
        raise ValueError('Training and teacher question IDs must match exactly')
    result = []
    for row in rows:
        teacher = cached[row['id']]
        if any(row[field] != teacher[other] for field, other in
               [('label', 'gold'), ('task', 'task'), ('source_dataset', 'source_dataset'), ('source_id', 'source_id')]):
            raise ValueError('Teacher metadata differs from training row')
        keys, _ = candidates(row)
        probabilities = teacher['probabilities']
        if set(keys) != set(probabilities):
            raise ValueError('Teacher candidate identities differ')
        values = [probabilities[key] for key in keys]
        if any(not math.isfinite(p) or p < 0 for p in values) or not math.isclose(sum(values), 1., abs_tol=1e-5):
            raise ValueError('Teacher probabilities must be finite, nonnegative and normalized')
        if 'target_probs' in row:
            raise ValueError('This distillation experiment requires separate hard gold labels')
        result.append({**row, 'teacher_probs': values})
    return result, receipt


class DistillationCollator(DecisionCollator):
    """Add padded offline teacher targets; gold targets retain their usual tensor."""
    def __call__(self, rows):
        batch = super().__call__(rows)
        teacher = torch.zeros_like(batch['targets'])
        for index, row in enumerate(rows):
            values = row['teacher_probs']
            teacher[index, :len(values)] = torch.tensor(values, dtype=torch.float32)
        batch['teacher_targets'] = teacher
        return batch


def distillation_kl(logits, teacher, candidate_mask, temperature=2.):
    """Mean KL(soft teacher || soft student), multiplied by T squared.

    Teachers are calibrated probability mixtures at T=1; soften by p**(1/T)
    followed by renormalization over valid candidates. Padding contributes
    neither probability nor gradient. Calculation uses FP32 under CUDA AMP.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Distillation temperature must be finite and positive')
    if logits.shape != teacher.shape or logits.shape != candidate_mask.shape:
        raise ValueError('Student, teacher and candidate mask shapes must match')
    valid = candidate_mask.bool()
    scores = logits.float().masked_fill(~valid, float('-inf')) / temperature
    log_student = scores.log_softmax(-1).masked_fill(~valid, 0.)
    softened = teacher.float().clamp_min(0).pow(1. / temperature).masked_fill(~valid, 0.)
    softened = softened / softened.sum(-1, keepdim=True)
    log_teacher = softened.clamp_min(1e-15).log()
    return (softened * (log_teacher - log_student)).sum(-1).mean() * temperature ** 2
