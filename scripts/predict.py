#!/usr/bin/env python3
"""Batch independent typed decisions; write JSONL without generating text."""
import argparse
import json

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from decision_moe import DecisionMoE
from decision_moe.data import DecisionCollator, QTYPES, candidates, read_rows
from decision_moe.model import model_inputs


def main():
    """Format typed JSONL decisions from a local Julia MoE checkpoint.

    Loads once, batches independent questions, and applies saved per-type
    temperatures. Choice/score expose probabilities and maximum-probability
    confidence; Noul exposes true probability only. Output order/IDs follow
    input. Opens output for replacement, and a late failure can leave partial
    output. This CLI does not enable BF16 autocast or execute actions.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--head-length", type=int, default=512, help="Question/options token budget")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    from pathlib import Path
    if Path(args.input_file).resolve() == Path(args.output_file).resolve():
        parser.error("output-file must differ from input-file")
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    model = DecisionMoE.from_pretrained(args.model_dir).to(args.device).eval()
    if args.max_length > model.encoder.config.max_position_embeddings:
        parser.error("max-length exceeds encoder context window")
    tokenizer = AutoTokenizer.from_pretrained(Path(args.model_dir) / "tokenizer", local_files_only=True)
    rows = read_rows(args.input_file, labeled=False)
    loader = DataLoader(rows, batch_size=args.batch_size,
                        collate_fn=DecisionCollator(tokenizer, args.max_length, args.head_length, labeled=False))
    offset = 0
    with open(args.output_file, "w") as out, torch.inference_mode():
        for batch in loader:
            inputs = {k: value.to(args.device) for k, value in model_inputs(batch).items()}
            logits, _ = model(**inputs)
            for row, scores in zip(rows[offset:offset + len(logits)], logits):
                keys, _ = candidates(row)
                kind = row.get("type", "choice")
                probs = (scores[:len(keys)] / model.temperature[QTYPES[kind]]).softmax(-1).cpu().tolist()
                result = {"id": row.get("id", offset), "type": kind}
                if kind == "noul":
                    result["noul"] = probs[1]
                else:
                    result["probabilities"] = dict(zip(map(str, keys), probs))
                    # Our definition: largest candidate probability, not Jev confidence.
                    result["confidence"] = max(probs)
                    if kind == "choice":
                        result["choice"] = keys[max(range(len(probs)), key=probs.__getitem__)]
                    else:
                        result["score"] = sum(value * p for value, p in zip(keys, probs))
                out.write(json.dumps(result) + "\n")
                offset += 1


if __name__ == "__main__":
    main()
