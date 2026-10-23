"""Generate sequential offline teachers for training, retaining gold supervision."""
import argparse
import json
from pathlib import Path

from decision_moe.distillation import file_sha256
from evaluate_gotham import evaluate_checkpoint
from experiment_committee import ARMS, load_predictions, mix


def main():
    """Bind cached committee probabilities to exact train, policy and model bytes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-root', type=Path, required=True)
    parser.add_argument('--train-file', type=Path, required=True)
    parser.add_argument('--policy-file', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    policy = json.loads(args.policy_file.read_text())
    if policy['arms'] != list(ARMS):
        raise ValueError('Teacher policy arm order differs')
    models = {}
    for arm in ARMS:
        model = args.teacher_root / arm
        models[arm] = file_sha256(model / 'model.safetensors')
        evaluate_checkpoint(model, args.train_file, args.output_dir / f'{arm}_training_evaluation.json',
                            device=args.device, fp16=args.fp16, head_length=256,
                            predictions_file=args.output_dir / f'{arm}_predictions.jsonl')
        # The evaluator does not retain the loaded model after returning.
        import gc
        import torch
        gc.collect()
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    arms = {arm: load_predictions(args.output_dir / f'{arm}_predictions.jsonl') for arm in ARMS}
    predictions = args.output_dir / 'teacher_predictions.jsonl'
    predictions.write_text(''.join(json.dumps(row) + '\n' for row in mix(arms, policy['weights'])))
    receipt = {'role': 'train', 'examples': len(arms['v1']), 'train_sha256': file_sha256(args.train_file),
               'policy_sha256': file_sha256(args.policy_file), 'teacher_model_sha256': models,
               'predictions_sha256': file_sha256(predictions),
               'recipe': 'Fixed task mixture of calibrated probabilities; gold labels retained separately; no fitting on training labels.'}
    (args.output_dir / 'teacher_manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
