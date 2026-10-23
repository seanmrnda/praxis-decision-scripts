"""Evaluate untouched public proxy decisions, including per-source/task slices."""
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from decision_moe import DecisionMoE
from decision_moe.data import DecisionCollator, candidates, read_rows
from decision_moe.model import model_inputs
from finetune import metrics


def evaluate_checkpoint(checkpoint, test_file, output_file, device='cuda', fp16=False,
                        batch_size=4, max_length=1024, head_length=256):
    """Compute calibrated candidate metrics without changing model parameters.

    F1 is over the union of observed gold/predicted labels within a slice.
    Rows are positive-pair/candidate-conditioned proxies, not full extraction.
    Score/Noul and operational tasks absent from the test are not evaluated.
    """
    model = DecisionMoE.from_pretrained(checkpoint).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(Path(checkpoint) / 'tokenizer', local_files_only=True)
    rows = read_rows(test_file)
    collator = DecisionCollator(tokenizer, max_length, head_length)
    collected, truth, predictions, labels = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            subset = rows[start:start + batch_size]
            batch = {key: value.to(device) for key, value in collator(subset).items()}
            context = torch.autocast('cuda', dtype=torch.float16) if fp16 else nullcontext()
            with context:
                logits, _ = model(**model_inputs(batch))
            logits = logits / model.temperature[batch['qtype'], None]
            for row, score, target, mask in zip(subset, logits, batch['targets'], batch['marker_mask']):
                keys, _ = candidates(row)
                n = int(mask.sum())
                collected.append(score[:n].cpu())
                truth.append(target[:n].cpu())
                predictions.append(keys[int(score[:n].argmax())])
                labels.append(row['label'])
    logits = torch.nn.utils.rnn.pad_sequence(collected, batch_first=True, padding_value=float('-inf'))
    targets = torch.nn.utils.rnn.pad_sequence(truth, batch_first=True)
    slices = defaultdict(list)
    for i, row in enumerate(rows):
        for key in ['all', 'dataset:' + row['source_dataset'], 'task:' + row['task']]:
            slices[key].append(i)
    report = {'model_name': model.model_name, 'examples': len(rows), 'scope': 'Untouched, candidate-conditioned public proxy test; no operational claims.', 'slices': {}}
    for key, indices in slices.items():
        gold = Counter(labels[i] for i in indices)
        predicted = Counter(predictions[i] for i in indices)
        correct = Counter(labels[i] for i in indices if labels[i] == predictions[i])
        scores = [2 * correct[label] / (gold[label] + predicted[label]) for label in gold.keys() | predicted.keys()]
        report['slices'][key] = {'examples': len(indices), **metrics(logits[indices], targets[indices]),
                               'macro_f1': sum(scores) / len(scores), 'gold_counts': dict(gold)}
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
    args = parser.parse_args()
    if args.fp16 and not args.device.startswith('cuda'):
        parser.error('fp16 evaluation requires CUDA')
    evaluate_checkpoint(args.checkpoint, args.test_file, args.output_file, args.device, args.fp16)


if __name__ == '__main__':
    main()
