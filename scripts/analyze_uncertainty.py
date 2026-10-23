#!/usr/bin/env python3
"""Compare source-ID and connected evidence-group accuracy bootstraps.

Post-hoc sensitivity analysis only. Never selects checkpoints or teacher policy.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from experiment_committee import load_predictions, paired_interval


def components(rows):
    """Join questions transitively whenever any supplied split_group is shared."""
    parent = {row['id']: row['id'] for row in rows}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    owners = {}
    for row in rows:
        if not row['split_groups']:
            raise ValueError('Missing split_groups: ' + row['id'])
        for group in row['split_groups']:
            if group in owners:
                parent[find(row['id'])] = find(owners[group])
            else:
                owners[group] = row['id']
    return {key: find(key) for key in parent}


def component_interval(candidate, baseline, component_ids, draws=500, seed=42):
    if set(candidate) != set(baseline) or set(candidate) != set(component_ids):
        raise ValueError('Paired prediction/data IDs differ')
    groups = defaultdict(list)
    for key, row in candidate.items():
        other = baseline[key]
        for field in ['gold', 'task', 'source_dataset', 'source_id']:
            if row[field] != other[field]:
                raise ValueError('Paired prediction metadata differ: ' + key)
        if set(row['probabilities']) != set(other['probabilities']):
            raise ValueError('Paired candidate identities differ: ' + key)
        correct = lambda item: max(item['probabilities'], key=item['probabilities'].get) == item['gold']
        groups[component_ids[key]].append(int(correct(row)) - int(correct(other)))
    arrays, samples, rng = list(groups.values()), [], random.Random(seed)
    for _ in range(draws):
        sample = [rng.choice(arrays) for _ in arrays]
        samples.append(sum(map(sum, sample)) / sum(map(len, sample)))
    samples.sort()
    return {'difference': sum(map(sum, arrays)) / len(candidate),
            'connected_evidence_group_bootstrap_95': [samples[int(.025 * draws)], samples[int(.975 * draws)]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-file', type=Path, required=True)
    parser.add_argument('--predictions-dir', type=Path, required=True)
    parser.add_argument('--output-file', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.draws < 40:
        parser.error('Use at least 40 draws')
    if args.output_file.exists():
        parser.error('Choose a fresh output file')
    rows = [json.loads(line) for line in args.data_file.read_text().splitlines() if line.strip()]
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate data IDs')
    component_ids = components(rows)
    candidate = load_predictions(args.predictions_dir / 'v3_fresh_predictions.jsonl')
    data = {row['id']: row for row in rows}
    if set(data) != set(candidate):
        raise ValueError('Predictions do not match the data rows')
    for key, row in candidate.items():
        if row['gold'] != data[key]['label'] or any(row[field] != data[key][field] for field in ['task', 'source_dataset', 'source_id']):
            raise ValueError('Prediction/data metadata differ: ' + key)
        if set(row['probabilities']) != set(data[key]['options']):
            raise ValueError('Prediction/data candidates differ: ' + key)
    counts = defaultdict(int)
    for group in component_ids.values():
        counts[group] += 1
    result = {'scope': 'Post-hoc sensitivity only; no recipe or selection changes',
              'clustering': 'Connected components of shared split_groups (article/evidence/claim groups)',
              'components': len(counts), 'largest_component_questions': max(counts.values()),
              'source_id_groups': len({(r['source_dataset'], r['source_id']) for r in rows}),
              'component_draws': args.draws, 'seed': args.seed, 'source_id_draws': 500,
              'comparisons': {}}
    for arm in ['v1', 'v2', 'control', 'committee']:
        baseline = load_predictions(args.predictions_dir / f'{arm}_fresh_predictions.jsonl')
        interval = component_interval(candidate, baseline, component_ids, args.draws, args.seed)
        interval['source_id_comparison'] = paired_interval(list(candidate.values()), baseline, args.seed)
        result['comparisons'][arm] = interval
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
