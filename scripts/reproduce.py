#!/usr/bin/env python3
"""Plan or run the original Julia-1 -> Praxis Decision v1/v2/v3 study.

Planning is dependency-free and never executes downloads or training. Running
uses fresh stage outputs and preserves the original two-GPU FP16 recipe.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('download', 'convert', 'prepare-v1', 'train-v1', 'prepare-v2',
          'train-v2', 'committee', 'prepare-v3', 'train-v3', 'verify')


def commands(stage, work):
    """Return argv/runtime pairs. No imports of torch or mutable state."""
    scripts = ROOT / 'scripts'
    raw, data = work / 'raw', work / 'data'
    dense, base = work / 'julia-dense', work / 'praxis-base'
    v1, v2 = work / 'runs/v1/defense-praxis', work / 'runs/v2/defense-praxis'
    diag = v2 / 'diagnostics'
    entries = []

    def add(script, *args, runtime='current'):
        entries.append(([sys.executable, str(script), *map(str, args)], runtime))

    def evaluate(checkpoint, partition, folder, name, kind='moe'):
        add(scripts / 'evaluate_gotham.py', '--checkpoint', checkpoint,
            '--test-file', folder / (partition + '.jsonl'), '--output-file', v2 / (name + '_evaluation.json'),
            '--predictions-file', diag / (name + '_predictions.jsonl'), '--model-kind', kind, '--fp16')

    def train(script, model, folder, out, name, epochs, lr, head_lr, runtime='current', save=False):
        args = ['--model-dir', model, '--train-file', folder / 'train.jsonl',
                '--validation-file', folder / 'validation.jsonl', '--calibration-file', folder / 'calibration.jsonl',
                '--output-dir', out, '--model-name', name, '--epochs', epochs,
                '--batch-size', 4, '--grad-accum', 4, '--max-length', 1024, '--head-length', 256,
                '--lr', lr, '--head-lr', head_lr, '--fp16', '--data-parallel', '--seed', 42,
                '--log-steps', 10]
        if save:
            args += ['--save-each-epoch']
        add(script, *args, runtime=runtime)

    if stage == 'download':
        add(scripts / 'download_base.py', '--output-dir', dense)
        add(scripts / 'download_gotham_sources.py', '--output-dir', raw)
    elif stage == 'convert':
        add(ROOT / 'historical/v1/scripts/convert_julia.py', '--base-model', dense,
            '--output-dir', base, '--model-name', 'praxis', '--num-experts', 4, '--top-k', 2,
            '--moe-layers', 'every-other', '--expert-noise-std', 0, '--seed', 42, runtime='v1')
    elif stage == 'prepare-v1':
        add(scripts / 'prepare_gotham.py', '--raw-dir', raw, '--output-dir', data / 'v1',
            '--tokenizer-dir', dense / 'tokenizer')
        add(scripts / 'verify_reproduction.py', '--work-dir', work, '--profile', 'v1')
    elif stage == 'train-v1':
        add(ROOT / 'historical/v1/scripts/evaluate_gotham.py', '--checkpoint', base,
            '--test-file', data / 'v1/test.jsonl', '--output-file', work / 'runs/v1/baseline_evaluation.json',
            '--fp16', runtime='v1')
        train(ROOT / 'historical/v1/scripts/finetune.py', base, data / 'v1', v1,
              'defense-praxis', 2, '1e-5', '2e-5', runtime='v1', save=True)
        add(ROOT / 'historical/v1/scripts/evaluate_gotham.py', '--checkpoint', v1,
            '--test-file', data / 'v1/test.jsonl', '--output-file', v1 / 'evaluation.json', '--fp16', runtime='v1')
    elif stage == 'prepare-v2':
        add(scripts / 'prepare_improvement.py', '--previous-dir', data / 'v1', '--raw-dir', raw,
            '--output-dir', data / 'v2', '--tokenizer-dir', dense / 'tokenizer')
        add(scripts / 'verify_reproduction.py', '--work-dir', work, '--profile', 'v2')
    elif stage == 'train-v2':
        for role, partition in [('historical', 'test'), ('confirmation', 'confirmation')]:
            evaluate(dense, partition, data / 'v2', 'dense_' + role, 'dense')
        for role, partition in [('development', 'validation'), ('historical', 'test'), ('confirmation', 'confirmation')]:
            evaluate(v1, partition, data / 'v2', 'v1_' + role)
        train(ROOT / 'historical/v2/scripts/finetune.py', v1, data / 'v1', v2 / 'control_a',
              'defense-praxis-control-a', 4, '5e-6', '1e-5')
        for role, partition in [('development', 'validation'), ('historical', 'test'), ('confirmation', 'confirmation')]:
            evaluate(v2 / 'control_a', partition, data / 'v2', 'control_' + role)
        train(ROOT / 'historical/v2/scripts/finetune.py', v1, data / 'v2', v2,
              'defense-praxis-v2', 3, '5e-6', '1e-5', save=True)
        for role, partition in [('development', 'validation'), ('historical', 'test'), ('confirmation', 'confirmation')]:
            evaluate(v2, partition, data / 'v2', 'v2_' + role)
    elif stage == 'committee':
        add(scripts / 'experiment_committee.py', '--predictions-dir', diag,
            '--output-dir', work / 'committee-refit', '--grid', 10)
    elif stage == 'prepare-v3':
        add(scripts / 'prepare_distillation.py', '--raw-dir', raw, '--v1-data', data / 'v1',
            '--v2-data', data / 'v2', '--tokenizer-dir', dense / 'tokenizer',
            '--policy-file', ROOT / 'reproducibility/archived_teacher_policy.json',
            '--teacher-root', work / 'teachers', '--output-dir', data / 'v3')
        add(scripts / 'verify_reproduction.py', '--work-dir', work, '--profile', 'v3')
    elif stage == 'train-v3':
        add(scripts / 'run_distillation.py', '--root', work / 'v3-bundle', '--working', work / 'runs/v3')
    elif stage == 'verify':
        add(scripts / 'verify_reproduction.py', '--work-dir', work, '--all', '--reports')
    else:
        raise ValueError(stage)
    return entries


def runtime_env(runtime):
    env = os.environ.copy()
    source = ROOT / ('historical/v1/src' if runtime == 'v1' else 'src')
    env['PYTHONPATH'] = str(source)
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env['OMP_NUM_THREADS'] = '2'
    env['MKL_NUM_THREADS'] = '2'
    return env


def link(source, destination):
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=True)


def prepare_stage(stage, work):
    """Guard completed/partial runs before any expensive operation."""
    guards = {
        'download': [work / 'julia-dense'], 'convert': [work / 'praxis-base'],
        'prepare-v1': [work / 'data/v1'], 'train-v1': [work / 'runs/v1'],
        'prepare-v2': [work / 'data/v2'], 'train-v2': [work / 'runs/v2'],
        'committee': [work / 'committee-refit'],
        'prepare-v3': [work / 'data/v3', work / 'teachers'],
        'train-v3': [work / 'runs/v3', work / 'v3-bundle'],
    }
    for path in guards.get(stage, []):
        if path.exists() or path.is_symlink():
            raise ValueError(f'Refusing to overwrite {path}; use a fresh work directory or inspect the interrupted stage')
    if stage.startswith('train-'):
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise ValueError('Reported recipe requires exactly two visible CUDA GPUs; set CUDA_VISIBLE_DEVICES to two GPUs')
        for path in [work / 'runs/v1', work / 'runs/v2/defense-praxis/diagnostics'] if stage != 'train-v3' else []:
            # Only create the output appropriate to this stage.
            if (stage == 'train-v1' and path == work / 'runs/v1') or (stage == 'train-v2' and path != work / 'runs/v1'):
                path.mkdir(parents=True)
    if stage == 'prepare-v3':
        for arm, source in [('v1', work / 'runs/v1/defense-praxis'),
                            ('control', work / 'runs/v2/defense-praxis/control_a'),
                            ('v2', work / 'runs/v2/defense-praxis')]:
            link(source, work / 'teachers' / arm)
    if stage == 'train-v3':
        bundle = work / 'v3-bundle'
        for source, destination in [(ROOT, bundle / 'code'), (work / 'data/v3', bundle / 'data'),
                                    (work / 'julia-dense', bundle / 'dense'),
                                    (work / 'runs/v1/defense-praxis', bundle / 'praxis'),
                                    (work / 'teachers/control', bundle / 'teachers/control'),
                                    (work / 'teachers/v2', bundle / 'teachers/v2')]:
            link(source, destination)
        recipe = json.loads((work / 'data/v3/data_manifest.json').read_text())['planned_recipe']
        (bundle / 'experiment.json').write_text(json.dumps(recipe, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['plan', 'run'])
    parser.add_argument('--stage', choices=['all', *STAGES], default='all')
    parser.add_argument('--work-dir', type=Path, default=ROOT / 'work/reproduction')
    args = parser.parse_args()
    work = args.work_dir.resolve()
    if work == ROOT or work in ROOT.parents:
        parser.error('work-dir must be a separate directory, not the repository or its ancestor')
    stages = STAGES if args.stage == 'all' else (args.stage,)
    for stage in stages:
        print(f'\n[{stage}]', flush=True)
        if args.mode == 'run':
            prepare_stage(stage, work)
            logs = work / 'logs'
            logs.mkdir(parents=True, exist_ok=True)
        for index, (command, runtime) in enumerate(commands(stage, work)):
            env = runtime_env(runtime)
            print('PYTHONPATH=' + shlex.quote(env['PYTHONPATH']) + ' ' + shlex.join(command), flush=True)
            if args.mode == 'run':
                # Combined stdout/stderr remains visible and is archived per command.
                logfile = logs / f'{stage}-{index:02d}.log'
                with logfile.open('w') as log:
                    with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, text=True) as process:
                        for line in process.stdout:
                            print(line, end='', flush=True)
                            log.write(line)
                        if process.wait():
                            raise subprocess.CalledProcessError(process.returncode, command)
        if args.mode == 'run':
            (logs / (stage + '.complete.json')).write_text(json.dumps({'stage': stage, 'status': 'complete'}) + '\n')


if __name__ == '__main__':
    main()
