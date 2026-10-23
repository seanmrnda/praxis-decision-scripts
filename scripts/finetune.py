#!/usr/bin/env python3
"""Supervised decision tuning with proper scoring losses and optional calibration."""
import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, set_seed
from safetensors.torch import load_file
from decision_moe import DecisionMoE
from decision_moe.data import DecisionCollator, QTYPES, read_rows
from decision_moe.model import decision_loss, model_inputs
from decision_moe.distillation import attach_teachers, DistillationCollator, distillation_kl


def to_device(batch, device):
    """Move tensor fields to one device, preserving non-tensor fields.
    """
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def amp_context(args):
    """Return CUDA BF16/FP16 autocast or a no-op context for validated arguments.

    Parameter/optimizer storage stays FP32. main() verifies BF16 support.
    """
    if args.bf16 or args.fp16:
        return torch.autocast("cuda", dtype=torch.bfloat16 if args.bf16 else torch.float16)
    return nullcontext()


@torch.no_grad()
def evaluate(model, loader, args):
    """Collect raw candidate logits, targets, and type IDs without updates.

    Switches the model to eval mode. Returns CPU tensors [N,K], [N,K], [N],
    padding variable candidate counts with -infinity for logits and zero for
    targets. Does not apply temperatures or compute metrics. Requires labeled
    batches and args.device/args.bf16. Does not restore prior training mode.
    """
    model.eval()
    scores, targets, types = [], [], []
    for batch in loader:
        batch = to_device(batch, args.device)
        with amp_context(args):
            logits, _ = model(**model_inputs(batch))
        for logit, target, mask in zip(logits, batch["targets"], batch["marker_mask"]):
            count = int(mask.sum())
            scores.append(logit[:count].cpu())
            targets.append(target[:count].cpu())
        types.append(batch["qtype"].cpu())
    logits = torch.nn.utils.rnn.pad_sequence(scores, batch_first=True, padding_value=float("-inf"))
    truth = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)
    return logits, truth, torch.cat(types)


def metrics(logits, targets, temperature=1.0):
    """Summarize candidate probabilities against hard or soft targets.

    ``temperature`` can be one scalar or a broadcastable [N,1] tensor.
    Returns cross-entropy (named nll) and summed multiclass Brier. Accuracy
    and ten-bin ECE are included only for rows with exactly one-hot targets.
    Does not compute score MAE, per-type breakdowns, or threshold metrics.
    """
    scaled = logits / temperature
    probs = scaled.softmax(-1)
    result = {"nll": decision_loss(scaled, targets).item(),
              "brier": (probs - targets).square().sum(-1).mean().item()}
    # Accuracy/ECE are only meaningful here on one-hot categorical targets.
    hard = (targets.max(-1).values == 1) & (targets.sum(-1) == 1)
    if hard.any():
        p = probs[hard]
        correct = p.argmax(-1).eq(targets[hard].argmax(-1)).float()
        confidence = p.max(-1).values
        ece = 0.0
        for i in range(10):
            selected = (confidence > i / 10) & (confidence <= (i + 1) / 10)
            if selected.any():
                ece += selected.float().mean().item() * abs(confidence[selected].mean().item() - correct[selected].mean().item())
        result.update(accuracy=correct.mean().item(), ece=ece, hard_label_examples=int(hard.sum()))
    return result


def calibrate(logits, targets):
    # Finite padding prevents 0 * infinity in the temperature derivative.
    """Fit one temperature to a supplied CPU score/target partition.

    Bounded LBFGS minimizes cross-entropy over approximately [0.05,20].
    Replaces padding infinities with a finite floor to avoid undefined
    temperature gradients. Returns a Python float; does not modify the model.
    The caller separates question types and supplies independent held-out data.
    """
    logits = logits.masked_fill(~torch.isfinite(logits), -1e9)
    # Bounded scalar fit; optimize on a separate calibration partition.
    raw = torch.tensor(math.log((1 - 0.05) / (20 - 1)), requires_grad=True)
    optimizer = torch.optim.LBFGS([raw], max_iter=50, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        temperature = 0.05 + 19.95 * raw.sigmoid()
        loss = decision_loss(logits / temperature, targets)
        loss.backward()
        return loss
    optimizer.step(closure)
    return (0.05 + 19.95 * raw.sigmoid()).detach().item()


def main():
    """Train a local Julia MoE model, select by validation NLL, then calibrate.

    Parses CLI settings, validates separate data roles, builds AdamW groups,
    and weights accumulation windows by question count. The action head stays
    frozen. Saves only the best model/tokenizer and a training report, with
    optional per-type temperature fitting; exact resume is not implemented.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--calibration-file", help="Separate held-out rows for temperature fitting")
    parser.add_argument("--teacher-predictions", help="Offline training-only teacher probability JSONL")
    parser.add_argument("--teacher-manifest", help="Hashed receipt binding teacher cache to exact training file")
    parser.add_argument("--distill-weight", type=float, default=0.5, help="Soft KL weight; gold CE gets 1-weight when teachers are supplied")
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", help="Display name for the fine-tuned checkpoint")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4, help="Independent questions per batch")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--head-length", type=int, default=512, help="Question/options token budget")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--aux-weight", type=float, default=0.01)
    parser.add_argument("--brier-weight", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only routers and decision head")
    parser.add_argument("--bf16", action="store_true", help="CUDA BF16 autocast; parameters stay FP32")
    parser.add_argument("--fp16", action="store_true", help="CUDA FP16 autocast with gradient scaling (T4 compatible)")
    parser.add_argument("--save-each-epoch", action="store_true", help="Also retain complete weights after each epoch; no optimizer resume state")
    parser.add_argument("--data-parallel", action="store_true", help="Use all visible CUDA GPUs for question batches")
    parser.add_argument("--log-steps", type=int, default=25, help="Print progress every this many optimizer windows")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if bool(args.teacher_predictions) != bool(args.teacher_manifest):
        parser.error('teacher-predictions and teacher-manifest must be supplied together')
    if not 0 <= args.distill_weight < 1 or not math.isfinite(args.distill_temperature) or args.distill_temperature <= 0:
        parser.error('distill-weight must be in [0,1); distill-temperature must be positive and finite')
    if min(args.epochs, args.batch_size, args.grad_accum) < 1:
        parser.error("epochs, batch-size and grad-accum must be positive")
    if min(args.lr, args.head_lr, args.max_grad_norm) <= 0 or min(args.aux_weight, args.brier_weight, args.weight_decay) < 0:
        parser.error("Learning rates/gradient norm must be positive; loss weights/weight decay nonnegative")
    if not 0 <= args.warmup_ratio <= 1:
        parser.error("warmup-ratio must be in [0, 1]")
    if args.bf16 and (not args.device.startswith("cuda") or not torch.cuda.is_bf16_supported()):
        parser.error("--bf16 requires a CUDA device with BF16 support")
    if args.fp16 and not args.device.startswith("cuda"):
        parser.error("--fp16 requires CUDA")
    if args.fp16 and args.bf16:
        parser.error("Choose only one of fp16 and bf16")
    if args.log_steps < 1:
        parser.error("log-steps must be positive")
    if args.data_parallel and (not args.device.startswith('cuda') or torch.cuda.device_count() < 2):
        parser.error("--data-parallel requires at least two CUDA GPUs")
    if args.data_parallel and args.batch_size < torch.cuda.device_count():
        parser.error("batch-size must be at least the number of visible GPUs")
    if args.data_parallel and args.device not in ('cuda', 'cuda:0'):
        parser.error("DataParallel requires the primary device cuda:0")
    paths = [Path(p).resolve() for p in (args.train_file, args.validation_file, args.calibration_file) if p]
    if len(set(paths)) != len(paths):
        parser.error("Training, validation and calibration must use separate files")
    if Path(args.output_dir).resolve() == Path(args.model_dir).resolve():
        parser.error("output-dir must differ from model-dir")
    output = Path(args.output_dir)
    if (output / "model.safetensors").exists():
        parser.error("output-dir already contains a model; choose a fresh directory")
    set_seed(args.seed)
    model = DecisionMoE.from_pretrained(args.model_dir).to(args.device)
    model.model_name = args.model_name or model.model_name + "-finetuned"
    if args.max_length > model.encoder.config.max_position_embeddings:
        parser.error("max-length exceeds encoder context window")
    tokenizer = AutoTokenizer.from_pretrained(Path(args.model_dir) / "tokenizer", local_files_only=True)
    collator = DecisionCollator(tokenizer, args.max_length, args.head_length)
    train_rows = read_rows(args.train_file)
    val_rows = read_rows(args.validation_file)
    cal_rows = read_rows(args.calibration_file) if args.calibration_file else None
    # Detect exact state/question duplicates even if labels differ.
    def fingerprints(rows):
        return {json.dumps({k: v for k, v in row.items() if k not in ("label", "target_probs", "id")}, sort_keys=True) for row in rows}
    partitions = [fingerprints(train_rows), fingerprints(val_rows)]
    if cal_rows is not None:
        partitions.append(fingerprints(cal_rows))
    if any(partitions[i] & partitions[j] for i in range(len(partitions)) for j in range(i)):
        parser.error("Data partitions contain identical state/question examples")
    teacher_receipt = None
    train_collator = collator
    if args.teacher_predictions:
        train_rows, teacher_receipt = attach_teachers(train_rows, args.train_file, args.teacher_predictions, args.teacher_manifest)
        train_collator = DistillationCollator(tokenizer, args.max_length, args.head_length)
    train_loader = DataLoader(train_rows, batch_size=args.batch_size, shuffle=True, collate_fn=train_collator)
    val_loader = DataLoader(val_rows, batch_size=args.batch_size, collate_fn=collator)
    if args.freeze_backbone:
        model.encoder.requires_grad_(False)
        for block in model.moe_blocks:
            block.router.requires_grad_(True)
    # The preserved action head has no targets in this supervised decision objective.
    model.act_head.requires_grad_(False)
    forward_model = torch.nn.DataParallel(model) if args.data_parallel else model
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lr = args.head_lr if not name.startswith("encoder.") or ".router." in name else args.lr
        decay = args.weight_decay if parameter.ndim > 1 else 0.0
        groups.setdefault((lr, decay), []).append(parameter)
    optimizer = torch.optim.AdamW([dict(params=p, lr=lr, weight_decay=decay) for (lr, decay), p in groups.items()])
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    steps = updates_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.warmup_ratio * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)
    best_nll, history = float("inf"), []
    started = time.monotonic()
    updates, skipped_updates = 0, 0
    model.temperature.fill_(1.0)  # Never carry a previous calibration into training.
    for epoch in range(args.epochs):
        forward_model.train()
        optimizer.zero_grad(set_to_none=True)
        train_total = 0.0
        for i, batch in enumerate(train_loader):
            batch = to_device(batch, args.device)
            with amp_context(args):
                logits, aux = forward_model(**model_inputs(batch))
                aux = aux.mean()  # DataParallel gathers replica scalars as a vector.
            targets = batch["targets"]
            nll = decision_loss(logits, targets)
            brier = (logits.softmax(-1) - targets).square().sum(-1).mean()
            supervised = nll
            if teacher_receipt is not None:
                soft_loss = distillation_kl(logits, batch['teacher_targets'], batch['marker_mask'], args.distill_temperature)
                supervised = (1 - args.distill_weight) * nll + args.distill_weight * soft_loss
            loss = supervised + args.brier_weight * brier + args.aux_weight * aux
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            # Weight microbatches by question count, including the final short batch.
            start = (i // args.grad_accum) * args.grad_accum
            end = min(start + args.grad_accum, len(train_loader))
            window_questions = min(end * args.batch_size, len(train_rows)) - start * args.batch_size
            scaler.scale(loss * len(targets) / window_questions).backward()
            train_total += loss.detach().item() * len(targets)
            if (i + 1) % args.grad_accum == 0 or i + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=not args.fp16)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                    updates += 1
                else:
                    skipped_updates += 1
                optimizer.zero_grad(set_to_none=True)
                if (updates + skipped_updates) % args.log_steps == 0:
                    print(json.dumps({"epoch": epoch + 1, "microbatch": i + 1,
                                      "optimizer_updates": updates, "skipped_updates": skipped_updates,
                                      "train_loss_so_far": train_total / min((i + 1) * args.batch_size, len(train_rows)),
                                      "elapsed_seconds": round(time.monotonic() - started)}), flush=True)
        logits, targets, _ = evaluate(forward_model, val_loader, args)
        record = {"epoch": epoch + 1, "train_loss": train_total / len(train_rows), **metrics(logits, targets)}
        history.append(record)
        print(json.dumps(record), flush=True)
        if record["nll"] < best_nll:
            best_nll = record["nll"]
            model.save_pretrained(output)
            tokenizer.save_pretrained(output / "tokenizer")
        if args.save_each_epoch:
            checkpoint = output / f"epoch-{epoch + 1:03d}"
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint / "tokenizer")
            (checkpoint / "epoch_report.json").write_text(json.dumps(record, indent=2) + "\n")
        (output / "progress.json").write_text(json.dumps({"status": "training", "history": history,
            "best_nll": best_nll, "optimizer_updates": updates, "skipped_updates": skipped_updates,
            "args": vars(args)}, indent=2) + "\n")
    # Reload the selected checkpoint, then calibrate independently of model selection.
    del optimizer, scheduler, groups
    # Load into existing parameters instead of allocating a second GPU model.
    model.load_state_dict(load_file(str(output / "model.safetensors")), strict=True)
    report = {"status": "complete", "history": history, "args": vars(args),
              "optimizer_updates": updates, "skipped_updates": skipped_updates,
              "elapsed_seconds": round(time.monotonic() - started), "temperatures": dict.fromkeys(QTYPES, 1.0)}
    if teacher_receipt is not None:
        report['distillation'] = {'teacher_receipt': teacher_receipt, 'weight': args.distill_weight,
                                  'temperature': args.distill_temperature,
                                  'objective': '(1-alpha)*gold CE + alpha*T^2*KL(soft teacher || soft student) + gold Brier + router aux'}
    if cal_rows is not None:
        cal_loader = DataLoader(cal_rows, batch_size=args.batch_size, collate_fn=collator)
        logits, targets, types = evaluate(forward_model, cal_loader, args)
        report["calibration_before"] = metrics(logits, targets)
        report["calibration_counts"] = {}
        for name, index in QTYPES.items():
            subset = types.eq(index)
            report["calibration_counts"][name] = int(subset.sum())
            # Missing types retain unit temperature after fine-tuning.
            if subset.any():
                temperature = calibrate(logits[subset], targets[subset])
                model.temperature[index] = temperature
                report["temperatures"][name] = temperature
        row_temperatures = model.temperature.detach().cpu()[types, None]
        report["calibration_after"] = metrics(logits, targets, row_temperatures)
        model.save_pretrained(output)
    logits, targets, types = evaluate(forward_model, val_loader, args)
    report["validation"] = metrics(logits, targets, model.temperature.detach().cpu()[types, None])
    (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "progress.json").write_text(json.dumps({"status": "complete", "history": history,
        "best_nll": best_nll, "optimizer_updates": updates, "skipped_updates": skipped_updates}, indent=2) + "\n")
    print(json.dumps({"saved": str(output), "validation": report["validation"], "temperatures": report["temperatures"]}, indent=2))


if __name__ == "__main__":
    main()
