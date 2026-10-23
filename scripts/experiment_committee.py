"""Fit a task-conditioned probability committee from saved development predictions.

CPU-only retrospective experiment, not a checkpoint or inference service. Never
fits on evaluation labels. Candidate identities, row metadata, and source-group
separation are checked before reporting evaluation. Task is an application input;
source_dataset/source_id/gold must never be serving-time routing features.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

ARMS = ('v1', 'control', 'v2')


def load_predictions(path):
    """Read uniquely identified rows and normalize finite saved probabilities."""
    rows = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row['id'] in rows:
            raise ValueError('Duplicate prediction id: ' + row['id'])
        probs = row['probabilities']
        if not probs or row['gold'] not in probs:
            raise ValueError('Missing candidates or gold candidate')
        if any(not math.isfinite(p) or p < 0 for p in probs.values()):
            raise ValueError('Nonfinite or negative probability')
        total = sum(probs.values())
        if not math.isclose(total, 1., abs_tol=1e-5):
            raise ValueError('Candidate probabilities must sum to one')
        row['probabilities'] = {key: value / total for key, value in probs.items()}
        rows[row['id']] = row
    if not rows:
        raise ValueError('Empty prediction file')
    return rows


def align(arms):
    """Reject mixed partitions, inconsistent labels/tasks, or candidate sets."""
    first = arms[ARMS[0]]
    for arm in ARMS[1:]:
        if set(first) != set(arms[arm]):
            raise ValueError('Prediction row IDs differ across models')
        for key, row in first.items():
            other = arms[arm][key]
            if any(row[field] != other[field] for field in
                   ('gold', 'task', 'source_dataset', 'source_id')):
                raise ValueError('Prediction metadata mismatch: ' + key)
            if set(row['probabilities']) != set(other['probabilities']):
                raise ValueError('Candidate identities differ: ' + key)
    return list(first)


def mix(arms, weights):
    """Return blended rows; weights depend only on the supplied task tag."""
    ids = align(arms)
    result = []
    for key in ids:
        row = dict(arms['v1'][key])
        task_weights = weights[row['task']]
        if len(task_weights) != len(ARMS) or any(w < 0 for w in task_weights) or not math.isclose(sum(task_weights), 1.):
            raise ValueError('Weights must form a three-model convex mixture')
        row['probabilities'] = {
            label: sum(w * arms[arm][key]['probabilities'][label]
                       for arm, w in zip(ARMS, task_weights))
            for label in row['probabilities']}
        row['prediction'] = max(row['probabilities'], key=row['probabilities'].get)
        row.pop('scores', None)  # Raw scores from a single arm are not blend scores.
        result.append(row)
    return result


def summarize(rows):
    """Compute candidate accuracy, observed-label macro F1, NLL, and Brier."""
    slices = defaultdict(list)
    for row in rows:
        for key in ('all', 'dataset:' + row['source_dataset'], 'task:' + row['task']):
            slices[key].append(row)
    reports = {}
    for key, subset in slices.items():
        gold, pred, correct = Counter(), Counter(), Counter()
        nll = brier = 0.
        for row in subset:
            label = max(row['probabilities'], key=row['probabilities'].get)
            gold[row['gold']] += 1
            pred[label] += 1
            correct[row['gold']] += int(label == row['gold'])
            nll -= math.log(max(row['probabilities'][row['gold']], 1e-15))
            brier += sum((p - int(k == row['gold'])) ** 2 for k, p in row['probabilities'].items())
        labels = gold.keys() | pred.keys()
        reports[key] = {'examples': len(subset), 'correct': sum(correct.values()),
                        'accuracy': sum(correct.values()) / len(subset),
                        'macro_f1': sum(2 * correct[k] / (gold[k] + pred[k]) for k in labels) / len(labels),
                        'nll': nll / len(subset), 'brier': brier / len(subset)}
    return reports


def fit_policy(development, grid=10):
    """Select each task's lowest development NLL on a fixed simplex grid.

    Grid=10 searches 66 weight triples, including all single-arm endpoints.
    Ties use development accuracy, then stable grid order. No holdout argument
    is accepted. Development selection can still overfit; fresh data is needed.
    """
    align(development)
    if grid < 1:
        raise ValueError('Grid must be positive')
    weights, selection = {}, {}
    for task in sorted({r['task'] for r in development['v1'].values()}):
        subset = {arm: {key: row for key, row in rows.items() if row['task'] == task}
                  for arm, rows in development.items()}
        trials = []
        for a in range(grid + 1):
            for b in range(grid + 1 - a):
                ws = [a / grid, b / grid, (grid - a - b) / grid]
                stats = summarize(mix(subset, {task: ws}))['all']
                trials.append((stats['nll'], -stats['accuracy'], ws, stats))
        best = min(trials, key=lambda item: (item[0], item[1]))
        weights[task] = best[2]
        selection[task] = {'weights': dict(zip(ARMS, best[2])), 'development': best[3], 'trials': len(trials)}
    return weights, selection


def check_disjoint(development, evaluation):
    """Prevent development examples or document groups from entering holdout."""
    dev = development['v1']
    test = evaluation['v1']
    groups = lambda rows: {(r['source_dataset'], r['source_id']) for r in rows.values()}
    if set(dev) & set(test) or groups(dev) & groups(test):
        raise ValueError('Development and evaluation share row IDs or source groups')


def paired_interval(rows, baseline, seed=42):
    """Paired source-document bootstrap for accuracy difference (500 draws)."""
    grouped = defaultdict(list)
    for row in rows:
        other = baseline[row['id']]
        is_correct = lambda r: max(r['probabilities'], key=r['probabilities'].get) == r['gold']
        grouped[(row['source_dataset'], row['source_id'])].append(int(is_correct(row)) - int(is_correct(other)))
    groups, rng, samples = list(grouped.values()), random.Random(seed), []
    for _ in range(500):
        draws = [rng.choice(groups) for _ in groups]
        samples.append(sum(map(sum, draws)) / sum(map(len, draws)))
    samples.sort()
    return {'difference': sum(map(sum, groups)) / len(rows),
            'source_group_bootstrap_95': [samples[12], samples[487]], 'source_groups': len(groups)}


def main():
    """Freeze development policy before reading historical/confirmation files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--grid', type=int, default=10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    def load(partition):
        return {arm: load_predictions(args.predictions_dir / f'{arm}_{partition}_predictions.jsonl') for arm in ARMS}
    development = load('development')
    weights, selection = fit_policy(development, args.grid)
    policy = {'arms': ARMS, 'weights': weights, 'selection': selection, 'objective': 'development NLL',
              'grid': args.grid, 'routing_feature': 'task supplied by application',
              'development_sha256': {arm: hashlib.sha256((args.predictions_dir / f'{arm}_development_predictions.jsonl').read_bytes()).hexdigest() for arm in ARMS}}
    policy_path = args.output_dir / 'policy.json'
    policy_path.write_text(json.dumps(policy, indent=2) + '\n')
    report = {'policy_sha256': hashlib.sha256(policy_path.read_bytes()).hexdigest(),
              'scope': 'Retrospective diagnostics on previously inspected holdouts; not a fresh release validation.',
              'development': summarize(mix(development, weights)), 'evaluation': {}}
    for partition in ('historical', 'confirmation'):
        arms = load(partition)
        check_disjoint(development, arms)
        blended = mix(arms, weights)
        report['evaluation'][partition] = {
            'committee': summarize(blended),
            'single_models': {arm: summarize(list(rows.values())) for arm, rows in arms.items()},
            'paired_vs_v2': paired_interval(blended, arms['v2']),
            'input_sha256': {arm: hashlib.sha256((args.predictions_dir / f'{arm}_{partition}_predictions.jsonl').read_bytes()).hexdigest() for arm in ARMS}}
        (args.output_dir / f'{partition}_predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in blended))
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'weights': weights, 'evaluation': {p: r['committee']['all'] for p, r in report['evaluation'].items()}}, indent=2))


if __name__ == '__main__':
    main()
