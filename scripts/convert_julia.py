#!/usr/bin/env python3
"""Upcycle Julia-1 encoder FFNs while preserving its trained decision components."""
import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, set_seed
from decision_moe import DecisionMoE, JuliaDecisionModel

DEFAULT_REVISION = "a85b127321d580d65176c89ced8273f305745d85"


def main():
    """Convert a dense local/Hub Julia checkpoint into a local sparse model.

    Downloads only data/config/tokenizer artifacts for Hub sources. Strictly
    loads the entire trained Julia model before FFN replacement, preserves
    all decision components, saves tokenizer files separately, and records
    conversion provenance. Conversion runs on CPU and rejects existing weights
    at its destination; source revision does not apply to local directories.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="SupersonicLabs/Julia-1", help="Hub repo or local Julia checkpoint")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Pinned upstream commit; use main to follow updates")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="praxis", help="Display name; checkpoint architecture stays JuliaDecisionMoE")
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--moe-layers", default="every-other", help="all, every-other, or comma-separated zero-based indices")
    parser.add_argument("--expert-noise-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.expert_noise_std < 0:
        parser.error("expert-noise-std must be nonnegative")
    output = Path(args.output_dir)
    if (output / "model.safetensors").exists():
        parser.error("output-dir already contains a model; choose a fresh directory")
    if not 1 <= args.top_k <= args.num_experts:
        parser.error("Require 1 <= top-k <= num-experts")
    set_seed(args.seed)
    local = Path(args.base_model).is_dir()
    source = Path(args.base_model) if local else Path(snapshot_download(
        args.base_model, revision=args.revision,
        allow_patterns=["config.json", "julia_config.json", "encoder/config.json",
                        "model.safetensors", "tokenizer/*"]))
    if source.resolve() == output.resolve():
        parser.error("output-dir must differ from the source checkpoint")
    dense = JuliaDecisionModel.from_pretrained(source)
    tokenizer = AutoTokenizer.from_pretrained(source / "tokenizer", local_files_only=True)
    n = len(dense.encoder.layers)
    if args.moe_layers == "all":
        layers = list(range(n))
    elif args.moe_layers == "every-other":
        layers = list(range(1, n, 2))
    else:
        layers = [int(x) for x in args.moe_layers.split(",")]
    model = DecisionMoE(dense, layers, args.num_experts, args.top_k, args.expert_noise_std)
    model.model_name = args.model_name
    model.save_pretrained(output)
    tokenizer.save_pretrained(output / "tokenizer")
    metadata = vars(args) | {"resolved_commit": None if local else source.name,
                            "parameters": sum(p.numel() for p in model.parameters()),
                            "julia_settings": model.settings,
                            "preserved_components": ["head", "type_emb", "scorer", "act_head", "temperature"]}
    (output / "conversion.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print("Julia decision weights preserved; routers are new. Fine-tune for expert specialization.")


if __name__ == "__main__":
    main()
