"""Kaggle v3 pilot: cached committee teachers, matched control and one student."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from experiment_committee import ARMS, load_predictions, mix, summarize
from decision_moe.distillation import file_sha256
from distillation_results import build_comparison


def main():
    """Finish both training/calibration arms before opening fresh evaluations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--working', type=Path, required=True)
    args = parser.parse_args()
    root, working = args.root, args.working
    scripts, data = root / 'code/scripts', root / 'data'
    output = working / 'defense-praxis'
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = output / 'diagnostics'
    diagnostics.mkdir()
    recipe = json.loads((root / 'experiment.json').read_text())
    teachers = root / 'teachers'
    (teachers / 'v1').symlink_to(root / 'praxis', target_is_directory=True)
    for arm in ARMS:
        if file_sha256(teachers / arm / 'model.safetensors') != recipe['teacher_model_sha256'][arm]:
            raise ValueError('Frozen teacher model hash changed')
    if file_sha256(data / 'teacher_policy.json') != recipe['teacher_policy_sha256']:
        raise ValueError('Frozen teacher policy changed')
    def run(command):
        print(json.dumps({'stage_command': command}), flush=True)
        subprocess.run(command, check=True)
    cache = output / 'teacher_cache'
    run([sys.executable, str(scripts / 'cache_teachers.py'), '--teacher-root', str(teachers),
         '--train-file', str(data / 'train.jsonl'), '--policy-file', str(data / 'teacher_policy.json'),
         '--output-dir', str(cache), '--fp16'])
    shutil.copyfile(cache / 'teacher_manifest.json', output / 'teacher_manifest.json')
    shutil.copyfile(data / 'teacher_policy.json', output / 'teacher_policy.json')
    base = [sys.executable, str(scripts / 'finetune.py'), '--model-dir', str(root / 'praxis'),
            '--train-file', str(data / 'train.jsonl'), '--validation-file', str(data / 'validation.jsonl'),
            '--calibration-file', str(data / 'calibration.jsonl'), '--epochs', str(recipe['epochs']),
            '--batch-size', '4', '--grad-accum', '4', '--max-length', '1024', '--head-length', '256',
            '--lr', str(recipe['lr']), '--head-lr', str(recipe['head_lr']), '--fp16',
            '--log-steps', '10', '--seed', str(recipe['seed'])]
    import torch
    if torch.cuda.device_count() > 1:
        base.append('--data-parallel')
    run(base + ['--output-dir', str(output / 'control_c'), '--model-name', 'defense-praxis-control-c'])
    run(base + ['--output-dir', str(output), '--model-name', recipe['model_name'],
                '--teacher-predictions', str(cache / 'teacher_predictions.jsonl'),
                '--teacher-manifest', str(cache / 'teacher_manifest.json'),
                '--distill-weight', str(recipe['distill_weight']), '--distill-temperature', str(recipe['distill_temperature'])])
    checkpoints = {'v1': teachers / 'v1', 'teacher_a': teachers / 'control', 'v2': teachers / 'v2',
                   'control': output / 'control_c', 'v3': output, 'dense': root / 'dense'}
    roles = {'historical': 'test', 'confirmation': 'confirmation', 'fresh': 'fresh', 'development': 'validation'}
    for arm, checkpoint in checkpoints.items():
        partitions = ['fresh'] if arm == 'dense' else ['historical', 'confirmation', 'fresh']
        if arm in ['control', 'v3']:
            partitions = ['development'] + partitions
        for partition in partitions:
            name = arm + '_' + partition
            run([sys.executable, str(scripts / 'evaluate_gotham.py'), '--checkpoint', str(checkpoint),
                 '--test-file', str(data / (roles[partition] + '.jsonl')),
                 '--output-file', str(output / (name + '_evaluation.json')),
                 '--predictions-file', str(diagnostics / (name + '_predictions.jsonl')),
                 '--model-kind', 'dense' if arm == 'dense' else 'moe', '--fp16'])
    policy = json.loads((data / 'teacher_policy.json').read_text())
    for partition in ['historical', 'confirmation', 'fresh']:
        arms = {arm: load_predictions(diagnostics / f'{"teacher_a" if arm == "control" else arm}_{partition}_predictions.jsonl') for arm in ARMS}
        rows = mix(arms, policy['weights'])
        (diagnostics / f'committee_{partition}_predictions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        (output / f'committee_{partition}_evaluation.json').write_text(json.dumps({'examples': len(rows), 'slices': summarize(rows), 'kind': 'Fixed probability committee; no fitted v3 parameters'}, indent=2) + '\n')
    shutil.copyfile(output / 'v3_fresh_evaluation.json', output / 'evaluation.json')
    shutil.copyfile(output / 'v1_historical_evaluation.json', working / 'baseline_evaluation.json')
    comparison = build_comparison(output, data / 'data_manifest.json')
    print(json.dumps({'comparison': comparison}), flush=True)


if __name__ == '__main__':
    main()
