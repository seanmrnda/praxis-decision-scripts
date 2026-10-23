"""Freeze the v3 training recipe, teacher policy, and unused development groups."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from transformers import AutoTokenizer
from decision_moe.data import marker_sequence
from prepare_gotham import choose, maven_rows, redocred_rows, feverous_pool, resolve_wiki, feverous_rows
from prepare_improvement import read_rows, assert_disjoint, validated_event
from download_gotham_sources import sha256

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / '.local'


def unused_groups(rows, occupied):
    """Keep only rows whose entire source/evidence group was never assigned."""
    return [row for row in rows if not occupied.intersection(row['split_groups'])]


def main():
    """Copy v2 training roles exactly and reserve new groups before any training."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['raw-dir', 'v1-data', 'v2-data', 'tokenizer-dir', 'policy-file', 'teacher-root', 'output-dir']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    target = args.output_dir
    if target.exists():
        raise ValueError('V3 inputs already prepared; do not overwrite provenance')
    occupied = set()
    previous = {}
    for profile, folder in [('v1', args.v1_data), ('v2', args.v2_data)]:
        for path in sorted(folder.glob('*.jsonl')):
            rows = read_rows(path)
            occupied.update(g for row in rows for g in row['split_groups'])
            previous[f'{profile}/{path.name}'] = sha256(path)
    raw = args.raw_dir
    sources = json.loads((raw / 'sources.json').read_text())
    for name in ['maven-ere.zip', 'dev_revised.json']:
        if sha256(raw / name) != sources[name]['sha256']:
            raise ValueError('Raw source checksum changed: ' + name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    relations = json.loads((raw / 'rel_info.json').read_text())
    generated = list(maven_rows(raw / 'maven-ere.zip'))
    generated += list(redocred_rows(json.loads((raw / 'dev_revised.json').read_text()), relations, 'dev'))
    pool = feverous_pool(raw)
    generated += list(feverous_rows(pool, resolve_wiki(raw, pool)))
    fresh, skipped, available = [], Counter(), {}
    for dataset in ['MAVEN-ERE', 'Re-DocRED', 'FEVEROUS']:
        candidates = [row for _, row in generated if row['source_partition'] == 'dev' and row['source_dataset'] == dataset]
        candidates = unused_groups(candidates, occupied)
        valid = []
        for row in candidates:
            try:
                valid.append(validated_event(row, tokenizer))
            except ValueError:
                skipped[dataset + ':over_budget'] += 1
        available[dataset] = len(valid)
        chosen = choose(valid, 200, False)
        if len(chosen) < 100:
            raise ValueError('Fewer than 100 unused fresh examples for ' + dataset)
        fresh += chosen
        occupied.update(g for row in chosen for g in row['split_groups'])
    roles = {role: read_rows(args.v2_data / f'{role}.jsonl')
             for role in ['train', 'validation', 'calibration', 'test', 'confirmation']}
    roles['fresh'] = fresh
    assert_disjoint(roles)
    target.mkdir(parents=True)
    files = {}
    for role, rows in roles.items():
        path = target / f'{role}.jsonl'
        if role != 'fresh':
            shutil.copyfile(args.v2_data / path.name, path)
        else:
            path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        files[path.name] = {'sha256': sha256(path), 'rows': len(rows),
                           'datasets': dict(Counter(r['source_dataset'] for r in rows)),
                           'labels': dict(Counter(r['task'] + ':' + r['label'] for r in rows))}
    policy_path = args.policy_file
    policy = json.loads(policy_path.read_text())
    expected = {'entity_relation': [.6, .4, 0.], 'evidence_verdict': [.3, .7, 0.],
                'temporal_relation': [0., 0., 1.], 'causal_relation': [0., 0., 1.]}
    if policy['weights'] != expected or policy['arms'] != ['v1', 'control', 'v2']:
        raise ValueError('Reviewed teacher policy changed')
    shutil.copyfile(policy_path, target / 'teacher_policy.json')
    teachers = {arm: args.teacher_root / arm for arm in ['v1', 'control', 'v2']}
    recipe = {'profile': 'v3', 'epochs': 1, 'lr': 5e-6, 'head_lr': 1e-5,
              'effective_batch': 16, 'planned_optimizer_windows': 500,
              'distill_weight': .5, 'distill_temperature': 2., 'seed': 42,
              'model_name': 'defense-praxis-v3', 'publication_revision': 'experimental-v3',
              'initialization': 'v1 for both candidate and hard-label control',
              'teacher_policy_sha256': sha256(policy_path),
              'teacher_model_sha256': {arm: sha256(folder / 'model.safetensors') for arm, folder in teachers.items()}}
    if files['train.jsonl']['rows'] != 8000:
        raise ValueError('500-window pilot expects exactly 8000 training rows')
    manifest = {'format_version': 3, 'sources': sources, 'files': files,
                'previous_partition_sha256': previous, 'available_unused_dev_rows': available,
                'skipped': dict(skipped), 'planned_recipe': recipe,
                'scope': 'Public candidate-conditioned relations and supplied-gold-evidence verdicts; not operational validation.',
                'split_policy': 'Exact v2 train/validation/calibration retained. Historical/confirmation are retrospective only. Fresh uses unused native dev groups excluded from every v1/v2 role and every other v3 role.',
                'selection': 'Gold validation NLL only; temperatures fit only on calibration. Fresh labels never select policy, sampling, checkpoint or calibration.',
                'release_gate': 'Experimental branch only; no automatic main promotion. Fresh source accuracy regression over 2 percentage points against v1 or v2 is flagged; report rare-class recall, macro F1, calibration and paired uncertainty.'}
    (target / 'data_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'files': files, 'available_unused_dev_rows': available, 'recipe': recipe}, indent=2))


if __name__ == '__main__':
    main()
