#!/usr/bin/env python3
"""Verify paper data fingerprints, source-group separation and training budgets."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_profile(folder, expected):
    """Do not trust a generated manifest: hash and inspect actual JSONL bytes."""
    manifest = json.loads((folder / 'data_manifest.json').read_text())
    occupied, ids, result = set(), set(), {}
    for name, receipt in expected['files'].items():
        path = folder / name
        contents = rows(path)
        actual = {'sha256': sha256(path), 'rows': len(contents)}
        if actual != receipt:
            raise ValueError(f'Paper partition differs: {path}: {actual}')
        if any(manifest['files'][name][key] != value for key, value in actual.items()):
            raise ValueError('Generated manifest does not match bytes: ' + str(path))
        group_set, row_ids = set(), set()
        dataset_counts = Counter()
        for row in contents:
            if row['id'] in ids or row['id'] in row_ids:
                raise ValueError('Repeated row ID: ' + row['id'])
            if not row['split_groups'] or row['label'] not in row['options']:
                raise ValueError('Missing groups or invalid gold: ' + row['id'])
            row_ids.add(row['id'])
            group_set.update(row['split_groups'])
            dataset_counts[row['source_dataset']] += 1
        if group_set & occupied:
            raise ValueError('Source/evidence group occurs in multiple roles: ' + str(path))
        occupied.update(group_set)
        ids.update(row_ids)
        result[name] = {**actual, 'groups': len(group_set), 'datasets': dict(dataset_counts)}
    return result


def verify_reports(work):
    """Require full planned windows; successful update counts can vary with AMP."""
    paths = {'v1': ('v1/defense-praxis', 750), 'control_a': ('v2/defense-praxis/control_a', 1500),
             'v2': ('v2/defense-praxis', 1500), 'control_c': ('v3/defense-praxis/control_c', 500),
             'v3': ('v3/defense-praxis', 500)}
    result = {}
    for arm, (suffix, windows) in paths.items():
        report = json.loads((work / 'runs' / suffix / 'training_report.json').read_text())
        updates, skipped = report['optimizer_updates'], report['skipped_updates']
        if report['status'] != 'complete' or updates < 1 or updates + skipped != windows:
            raise ValueError('Incomplete optimizer budget: ' + arm)
        if skipped > windows * .05:
            raise ValueError('More than 5% AMP windows skipped: ' + arm)
        result[arm] = {'updates': updates, 'skipped': skipped, 'planned': windows,
                       'temperatures': report['temperatures']}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--profile', choices=['v1', 'v2', 'v3'])
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--reports', action='store_true')
    args = parser.parse_args()
    if not args.profile and not args.all:
        parser.error('Choose --profile or --all')
    expected = json.loads((ROOT / 'reproducibility/expected_partitions.json').read_text())
    profiles = list(expected) if args.all else [args.profile]
    result = {p: verify_profile(args.work_dir / 'data' / p, expected[p]) for p in profiles}
    if args.all or args.profile == 'v3':
        for name in ['train', 'validation', 'calibration', 'test', 'confirmation']:
            if sha256(args.work_dir / f'data/v2/{name}.jsonl') != sha256(args.work_dir / f'data/v3/{name}.jsonl'):
                raise ValueError('v3 must retain exact v2 data: ' + name)
        occupied = set()
        for profile in ['v1', 'v2']:
            for path in (args.work_dir / 'data' / profile).glob('*.jsonl'):
                occupied.update(g for row in rows(path) for g in row['split_groups'])
        fresh = rows(args.work_dir / 'data/v3/fresh.jsonl')
        if occupied.intersection(g for row in fresh for g in row['split_groups']):
            raise ValueError('Fresh groups overlap a previous experimental role')
    if args.reports:
        result['training'] = verify_reports(args.work_dir)
    print(json.dumps({'status': 'verified', 'checks': result}, indent=2))


if __name__ == '__main__':
    main()
