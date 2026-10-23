"""Download only the pinned Julia-1 checkpoint, configurations and tokenizer."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download

REVISION = 'a85b127321d580d65176c89ced8273f305745d85'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('Choose a fresh base checkpoint directory')
    snapshot_download('SupersonicLabs/Julia-1', revision=REVISION, local_dir=args.output_dir,
                      allow_patterns=['config.json', 'julia_config.json', 'encoder/config.json',
                                      'model.safetensors', 'tokenizer/*'], token=False)
    files = {}
    for path in sorted(args.output_dir.rglob('*')):
        if path.is_file() and '.cache' not in path.parts:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    digest.update(block)
            files[str(path.relative_to(args.output_dir))] = digest.hexdigest()
    (args.output_dir / 'download_provenance.json').write_text(json.dumps({
        'repo_id': 'SupersonicLabs/Julia-1', 'revision': REVISION, 'sha256': files}, indent=2) + '\n')


if __name__ == '__main__':
    main()
