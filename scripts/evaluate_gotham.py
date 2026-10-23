"""Evaluate untouched public proxy decisions, including per-source/task slices."""
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from decision_moe import DecisionMoE
from decision_moe.julia import JuliaDecisionModel
from decision_moe.data import DecisionCollator, candidates, read_rows
from decision_moe.model import model_inputs
from finetune import metrics


def evaluate_checkpoint(checkpoint, test_file, output_file, device='cuda', fp16=False,
                        batch_size=4, max_length=1024, head_length=256,
                        model_kind='moe', predictions_file=None):
    """Compute calibrated candidate metrics without changing model parameters.

    F1 is over the union of observed gold/predicted labels within a slice.
    Rows are positive-pair/candidate-conditioned proxies, not full extraction.
    Score/Noul and operational tasks absent from the test are not evaluated.
    """
    loader = JuliaDecisionModel if model_kind == 'dense' else DecisionMoE
    model = loader.from_pretrained(checkpoint).to(device).eval()
    if model_kind == 'moe':
        for block in model.moe_blocks:
            block.observe_routing = True
            block.routing_stats = None
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(Path(checkpoint) / 'tokenizer', local_files_only=True)
    rows = read_rows(test_file)
    collator = DecisionCollator(tokenizer, max_length, head_length)
    collected, truth, predictions, labels, details = [], [], [], [], []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            subset = rows[start:start + batch_size]
            batch = {key: value.to(device) for key, value in collator(subset).items()}
            context = torch.autocast('cuda', dtype=torch.float16) if fp16 else nullcontext()
            with context:
                result = model(**model_inputs(batch))
                logits = result[0] if isinstance(result, tuple) else result
            logits = logits / model.temperature[batch['qtype'], None]
            for row, score, target, mask in zip(subset, logits, batch['targets'], batch['marker_mask']):
                keys, _ = candidates(row)
                n = int(mask.sum())
                collected.append(score[:n].cpu())
                truth.append(target[:n].cpu())
                predictions.append(keys[int(score[:n].argmax())])
                labels.append(row['label'])
                details.append({'id': row['id'], 'source_dataset': row['source_dataset'],
                                'source_id': row['source_id'], 'task': row['task'],
                                'gold': row['label'], 'prediction': predictions[-1],
                                'scores': dict(zip(keys, score[:n].tolist())),
                                'probabilities': dict(zip(keys, score[:n].softmax(-1).tolist()))})
    logits = torch.nn.utils.rnn.pad_sequence(collected, batch_first=True, padding_value=float('-inf'))
    targets = torch.nn.utils.rnn.pad_sequence(truth, batch_first=True)
    slices = defaultdict(list)
    for i, row in enumerate(rows):
        for key in ['all', 'dataset:' + row['source_dataset'], 'task:' + row['task']]:
            slices[key].append(i)
    report = {'model_name': getattr(model, 'model_name', 'dense-julia'), 'model_kind': model_kind,
              'examples': len(rows), 'scope': 'Candidate-conditioned public proxy evaluation; split role documented in experiment manifest.', 'slices': {}}
    for key, indices in slices.items():
        gold = Counter(labels[i] for i in indices)
        predicted = Counter(predictions[i] for i in indices)
        correct = Counter(labels[i] for i in indices if labels[i] == predictions[i])
        scores = [2 * correct[label] / (gold[label] + predicted[label]) for label in gold.keys() | predicted.keys()]
        per_class = {label: {'support': gold[label], 'precision': correct[label] / predicted[label] if predicted[label] else 0.,
                            'recall': correct[label] / gold[label] if gold[label] else None,
                            'f1': 2 * correct[label] / (gold[label] + predicted[label])}
                     for label in sorted(gold.keys() | predicted.keys())}
        confusion = defaultdict(Counter)
        groups = defaultdict(list)
        for i in indices:
            confusion[labels[i]][predictions[i]] += 1
            groups[rows[i]['source_dataset'] + ':' + rows[i]['source_id']].append(int(labels[i] == predictions[i]))
        rng = random.Random(42)
        grouped = list(groups.values())
        sampled = []
        for _ in range(500):
            draws = [rng.choice(grouped) for _ in grouped]
            sampled.append(sum(sum(draw) for draw in draws) / sum(len(draw) for draw in draws))
        sampled.sort()
        ranked = sorted(indices, key=lambda i: max(details[i]['probabilities'].values()), reverse=True)
        coverage = {}
        for fraction in [.25,.5,.75,1.]:
            subset = ranked[:max(1, round(len(ranked)*fraction))]
            coverage[str(fraction)] = {'examples':len(subset), 'error_rate':1-sum(labels[i]==predictions[i] for i in subset)/len(subset)}
        report['slices'][key] = {'examples': len(indices), **metrics(logits[indices], targets[indices]),
                               'macro_f1': sum(scores) / len(scores), 'gold_counts': dict(gold),
                               'per_class': per_class, 'balanced_accuracy': sum(correct[label] / gold[label] for label in gold) / len(gold),
                               'majority_label': gold.most_common(1)[0][0],
                               'majority_accuracy': gold.most_common(1)[0][1] / len(indices),
                               'uniform_random_expected_accuracy': sum(1 / len(rows[i]['options']) for i in indices) / len(indices),
                               'confusion': {label: dict(counts) for label, counts in confusion.items()},
                               'accuracy_source_group_bootstrap_95': [sampled[12], sampled[487]], 'source_groups': len(grouped),
                               'risk_coverage_diagnostic': coverage}
    if device.startswith('cuda'): torch.cuda.synchronize()
    report['runtime'] = {'elapsed_seconds':time.monotonic()-started,
                         'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,
                         'note':'Includes diagnostic CPU aggregation; not a serving latency benchmark'}
    if model_kind == 'moe':
        report['routing'] = {str(layer): block.routing_stats for layer,block in zip(model.spec['moe_layers'], model.moe_blocks)}
    if predictions_file:
        Path(predictions_file).write_text(''.join(json.dumps(row) + '\n' for row in details))
    Path(output_file).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    """CLI for a single Praxis-format checkpoint and fixed test partition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--test-file', required=True)
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--model-kind', choices=['moe', 'dense'], default='moe')
    parser.add_argument('--predictions-file')
    args = parser.parse_args()
    if args.fp16 and not args.device.startswith('cuda'):
        parser.error('fp16 evaluation requires CUDA')
    evaluate_checkpoint(args.checkpoint, args.test_file, args.output_file, args.device, args.fp16,
                        model_kind=args.model_kind, predictions_file=args.predictions_file)


if __name__ == '__main__':
    main()
