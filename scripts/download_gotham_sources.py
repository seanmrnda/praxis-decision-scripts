"""Fetch official public corpora, pin provenance, and cache artifacts locally.

No credentials are required for dataset downloads. Large FEVEROUS pages are
downloaded on the CPU host before the GPU job. Partial downloads are never
accepted as complete; existing files must match their recorded SHA-256.
"""
import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download


def sha256(path):
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, path):
    """Download to a temporary file, then atomically record bytes and checksum."""
    path = Path(path)
    receipt = path.with_suffix(path.suffix + '.receipt.json')
    if path.exists() and receipt.exists():
        record = json.loads(receipt.read_text())
        if record['url'] == url and record['sha256'] == sha256(path) and (path.suffix != '.zip' or zipfile.is_zipfile(path)):
            return record
        raise ValueError(f'Cached artifact checksum mismatch: {path.name}')
    temporary = path.with_suffix(path.suffix + '.partial')
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        size, started, last = 0, time.monotonic(), 0
        with temporary.open('wb') as file:
            for chunk in response.iter_content(4 * 1024 * 1024):
                file.write(chunk)
                size += len(chunk)
                if time.monotonic() - last > 30:
                    print(json.dumps({'artifact': path.name, 'downloaded_mb': round(size / 1e6),
                                      'elapsed_seconds': round(time.monotonic() - started)}), flush=True)
                    last = time.monotonic()
        if response.headers.get('Content-Length') and size != int(response.headers['Content-Length']):
            raise ValueError(f'Incomplete download: {path.name}')
        record = {'url': url, 'bytes': size, 'etag': response.headers.get('ETag')}
    if path.suffix == '.zip' and not zipfile.is_zipfile(temporary):
        raise ValueError(f'Download is not a ZIP archive: {path.name}')
    temporary.replace(path)
    record['sha256'] = sha256(path)
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'downloaded': path.name, **record}), flush=True)
    return record


def main():
    """Fetch the recorded release bytes; fail if an upstream artifact changed."""
    import gzip
    import shutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('.local/raw'))
    parser.add_argument('--verify-only', action='store_true', help='Check an existing raw cache without network access')
    args = parser.parse_args()
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    records = json.loads((Path(__file__).resolve().parents[1] / 'reproducibility/sources.json').read_text())
    for name, record in records.items():
        destination = root / name
        if not destination.exists() and args.verify_only:
            raise FileNotFoundError(destination)
        if not destination.exists():
            if name.endswith('_revised.json'):
                cached = hf_hub_download('tonytan48/Re-DocRED', name, repo_type='dataset',
                                         revision=record['revision'], token=False)
                shutil.copyfile(cached, destination)
            elif name == 'rel_info.json':
                cached = hf_hub_download('thunlp/docred', 'data/rel_info.json.gz', repo_type='dataset',
                                         revision=record['revision'], token=False)
                if sha256(cached) != record['compressed_sha256']:
                    raise ValueError('Compressed relation map changed')
                with gzip.open(cached, 'rb') as file:
                    destination.write_bytes(file.read())
            else:
                download(record['url'], destination)
        if sha256(destination) != record['sha256']:
            raise ValueError('Pinned source checksum mismatch: ' + name)
        if name.endswith('.zip') and not zipfile.is_zipfile(destination):
            raise ValueError('Invalid archive: ' + name)
        if 'bytes' in record and destination.stat().st_size != record['bytes']:
            raise ValueError('Pinned source size mismatch: ' + name)
        # Preserve original metadata only after validating actual bytes. In
        # particular prepare_gotham reads this exact MAVEN receipt.
        destination.with_suffix(destination.suffix + '.receipt.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({'verified': name, 'sha256': record['sha256']}), flush=True)
    (root / 'sources.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
