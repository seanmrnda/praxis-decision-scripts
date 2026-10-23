#!/usr/bin/env python3
"""Rebuild the matched comparison and source-group bootstrap from saved outputs."""
import argparse
import json
from pathlib import Path
from distillation_results import build_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    args = parser.parse_args()
    result = build_comparison(args.work_dir / 'runs/v3/defense-praxis', args.work_dir / 'data/v3/data_manifest.json')
    for arm, roles in result['arms'].items():
        print(f"{arm:12s} fresh accuracy: {100 * roles['fresh']['all']['accuracy']:.2f}%")
    print(json.dumps(result['fresh_paired_accuracy_vs'], indent=2))


if __name__ == '__main__':
    main()
