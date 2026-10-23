#!/usr/bin/env python3
"""Verify the shipped reproduction source against its assembly inventory."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    inventory = json.loads((ROOT / 'reproducibility/code_inventory.json').read_text())
    failures = []
    for name, expected in inventory['sha256'].items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(name)
    if failures:
        raise ValueError('Source inventory mismatch: ' + ', '.join(failures))
    print(json.dumps({'status': 'verified', 'files': len(inventory['sha256'])}))


if __name__ == '__main__':
    main()
