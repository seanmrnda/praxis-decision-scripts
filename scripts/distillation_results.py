"""Validate the v3 pilot and summarize matched control and frozen baselines."""
import json
import math
from pathlib import Path

from decision_moe.distillation import file_sha256
from experiment_committee import load_predictions, paired_interval


def build_comparison(checkpoint, manifest_file):
    """Require complete matched training windows and finite, counted holdouts.

    Called in Kaggle and again locally before publication. Fresh regression
    flags are descriptive release gates; this never promotes to main.
    """
    checkpoint = Path(checkpoint)
    manifest = json.loads(Path(manifest_file).read_text())
    recipe = manifest['planned_recipe']
    candidate = json.loads((checkpoint / 'training_report.json').read_text())
    control = json.loads((checkpoint / 'control_c/training_report.json').read_text())
    for name, report in [('candidate', candidate), ('control', control)]:
        if report.get('status') != 'complete' or report.get('optimizer_updates', 0) < 1:
            raise ValueError('Incomplete v3 training arm: ' + name)
        if report['optimizer_updates'] + report['skipped_updates'] != recipe['planned_optimizer_windows']:
            raise ValueError('Training did not execute all planned windows: ' + name)
        if report['skipped_updates'] > .05 * recipe['planned_optimizer_windows']:
            raise ValueError('More than 5% FP16 optimizer windows were skipped: ' + name)
    for key in ['model_dir', 'epochs', 'batch_size', 'grad_accum', 'lr', 'head_lr', 'seed', 'fp16', 'data_parallel', 'max_length', 'head_length', 'train_file', 'validation_file', 'calibration_file', 'brier_weight', 'aux_weight', 'weight_decay', 'warmup_ratio']:
        if candidate['args'][key] != control['args'][key]:
            raise ValueError('Unmatched control recipe: ' + key)
    teacher = candidate.get('distillation', {}).get('teacher_receipt', {})
    if control['args'].get('teacher_predictions') or candidate['args']['distill_weight'] != recipe['distill_weight'] or candidate['args']['distill_temperature'] != recipe['distill_temperature']:
        raise ValueError('Distillation/control treatment differs from frozen recipe')
    if teacher.get('train_sha256') != manifest['files']['train.jsonl']['sha256'] or teacher.get('policy_sha256') != recipe['teacher_policy_sha256'] or teacher.get('teacher_model_sha256') != recipe['teacher_model_sha256']:
        raise ValueError('Teacher cache provenance differs from frozen v3 recipe')
    if file_sha256(checkpoint / 'teacher_policy.json') != recipe['teacher_policy_sha256']:
        raise ValueError('Output teacher policy changed')
    role_files = {'historical': 'test.jsonl', 'confirmation': 'confirmation.jsonl', 'fresh': 'fresh.jsonl', 'development': 'validation.jsonl'}
    reports, full = {}, {}
    for arm in ['dense', 'v1', 'teacher_a', 'v2', 'committee', 'control', 'v3']:
        reports[arm], full[arm] = {}, {}
        partitions = ['fresh'] if arm == 'dense' else ['historical', 'confirmation', 'fresh']
        if arm in ['control', 'v3']:
            partitions += ['development']
        for partition in partitions:
            report = json.loads((checkpoint / f'{arm}_{partition}_evaluation.json').read_text())
            if report['examples'] != manifest['files'][role_files[partition]]['rows']:
                raise ValueError('Evaluation row count differs: ' + arm + ':' + partition)
            if any(not math.isfinite(s['nll']) for s in report['slices'].values()):
                raise ValueError('Nonfinite evaluation: ' + arm + ':' + partition)
            full[arm][partition] = report
            reports[arm][partition] = {key: {metric: value[metric] for metric in
                ['examples', 'accuracy', 'macro_f1', 'balanced_accuracy', 'nll', 'brier', 'ece', 'per_class'] if metric in value}
                for key, value in report['slices'].items()}
    diagnostics = checkpoint / 'diagnostics'
    rows = load_predictions(diagnostics / 'v3_fresh_predictions.jsonl')
    paired = {}
    for arm in ['v1', 'v2', 'control', 'committee']:
        baseline = load_predictions(diagnostics / f'{arm}_fresh_predictions.jsonl')
        if set(rows) != set(baseline):
            raise ValueError('Fresh paired prediction IDs differ')
        paired[arm] = paired_interval(list(rows.values()), baseline)
    warnings = []
    for baseline in ['v1', 'v2', 'control']:
        for key, current in reports['v3']['fresh'].items():
            if key.startswith('dataset:') and current['accuracy'] < reports[baseline]['fresh'][key]['accuracy'] - .02:
                warnings.append(f'{key}: v3 fresh accuracy more than two percentage points below {baseline}')
    result = {'arms': reports, 'fresh_paired_accuracy_vs': paired, 'fresh_regression_flags': warnings,
              'candidate_optimizer_updates': candidate['optimizer_updates'], 'control_optimizer_updates': control['optimizer_updates'],
              'planned_optimizer_windows': recipe['planned_optimizer_windows'],
              'status': 'Experimental v3 pilot only; main and experimental-v2 preserved.',
              'scope': 'Distillation vs matched hard-label continuation of v1. Dense Julia is not equally fine-tuned; no MoE architecture benefit claim. Historical/confirmation results are retrospective.'}
    (checkpoint / 'experiment_comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
