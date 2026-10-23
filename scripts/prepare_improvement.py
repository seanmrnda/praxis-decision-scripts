"""Prepare the second experiment while protecting every historical holdout.

Retain Re-DocRED/FEVEROUS replay; expand MAVEN questions with tempered label
sampling and explicit A/B role summaries. Keep full documents or reject length.
Reserve previously unused native-development article groups for confirmation.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
from transformers import AutoTokenizer
from decision_moe.data import marker_sequence
from prepare_gotham import choose, digest, maven_rows, redocred_rows, feverous_rows, feverous_pool, resolve_wiki
from download_gotham_sources import sha256

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / '.local'


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def enhance_event(row):
    """Add role summaries without altering argument direction or document text."""
    row = deepcopy(row)
    if row['source_dataset'] == 'MAVEN-ERE':
        summaries = []
        for role in ['A', 'B']:
            values = []
            for mention in row['state'][role]:
                phrase = mention.get('trigger_word', mention.get('mention', ''))
                values.append(f"{phrase!r} (sentence {mention['sent_id']}, token span {mention['offset']})")
            summaries.append(role + ': ' + '; '.join(values))
        row['state']['pair_summary'] = ' | '.join(summaries)
        row['question'] = ('Use the document to determine the annotated ' + row['task'].replace('_', ' ')
                           + ' FROM A TO B. A and B are identified in pair_summary; do not reverse them.')
    return row


def tempered_sample(rows, limit, natural_fraction=.8):
    """Sample mostly natural frequencies, reserving coverage for rare labels."""
    natural = choose(rows, int(limit * natural_fraction), False)
    seen = {row['id'] for row in natural}
    tail = choose([r for r in rows if r['id'] not in seen], limit - len(natural), True)
    return natural + tail


def validated_event(row, tokenizer):
    """Keep full original context if adding the summary exceeds the budget."""
    candidate = enhance_event(row)
    try:
        marker_sequence(tokenizer, candidate, 1024, 256)
        return candidate
    except ValueError:
        marker_sequence(tokenizer, row, 1024, 256)
        return deepcopy(row)


def assert_disjoint(partitions):
    """Reject any shared article/claim group across separately assigned roles."""
    occupied = set()
    for name, rows in partitions.items():
        groups = {g for row in rows for g in row['split_groups']}
        if occupied & groups:
            raise ValueError('Cross-role source-group overlap in ' + name)
        occupied |= groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-dir', type=Path, required=True)
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--tokenizer-dir', type=Path, required=True)
    args = parser.parse_args()
    old, raw, out = args.previous_dir, args.raw_dir, args.output_dir
    if out.exists():
        raise ValueError('Choose a new output directory; prepared inputs are immutable')
    original = {role: read_rows(old / f'{role}.jsonl') for role in ['train','validation','calibration','test']}
    old_groups = {g for rows in original.values() for row in rows for g in row['split_groups']}
    reserved = {g for role, rows in original.items() if role != 'train' for row in rows for g in row['split_groups']}
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    sources = json.loads((raw / 'sources.json').read_text())
    for name, receipt in sources.items():
        if name in ['maven-ere.zip','dev_revised.json'] and receipt['sha256'] != sha256(raw / name):
            raise ValueError('Raw source checksum differs: ' + name)
    relations = json.loads((raw / 'rel_info.json').read_text())
    generated = list(maven_rows(raw / 'maven-ere.zip'))
    generated += list(redocred_rows(json.loads((raw / 'dev_revised.json').read_text()), relations, 'dev'))
    pool = feverous_pool(raw); pages = resolve_wiki(raw, pool)
    generated += list(feverous_rows(pool, pages))
    usable = []; skipped = Counter()
    for split, row in generated:
        try: candidate = validated_event(row, tokenizer)
        except ValueError: skipped[row['source_dataset'] + ':over_budget'] += 1; continue
        usable.append((split, candidate))
    confirmation = []
    for dataset in ['MAVEN-ERE','Re-DocRED','FEVEROUS']:
        pool = [row for _, row in usable if row['source_partition']=='dev' and row['source_dataset']==dataset
                and not old_groups.intersection(row['split_groups'])]
        confirmation += choose(pool, 120, False)
    counts = Counter(row['source_dataset'] for row in confirmation)
    if any(counts[d] < 20 for d in ['MAVEN-ERE','Re-DocRED','FEVEROUS']):
        raise ValueError('Insufficient unused confirmation groups: ' + str(counts))
    reserved |= {g for row in confirmation for g in row['split_groups']}
    maven = [row for split,row in usable if row['source_dataset']=='MAVEN-ERE'
             and row['source_partition']=='train' and not reserved.intersection(row['split_groups'])]
    training = [enhance_event(row) for row in original['train'] if row['source_dataset']!='MAVEN-ERE']
    training += tempered_sample(maven, 4000)
    roles = {'train':training, 'validation':[validated_event(r, tokenizer) for r in original['validation']],
             'calibration':[validated_event(r, tokenizer) for r in original['calibration']],
             'test':original['test'], 'confirmation':confirmation}
    assert_disjoint(roles)
    for rows in roles.values():
        for row in rows: marker_sequence(tokenizer,row,1024,256)
    out.mkdir(parents=True,exist_ok=True)
    files = {}
    for role,rows in roles.items():
        if role=='train': random.Random(42).shuffle(rows)
        path=out/f'{role}.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        files[path.name]={'sha256':sha256(path),'rows':len(rows),'datasets':dict(Counter(r['source_dataset'] for r in rows)),
                         'labels':dict(Counter(r['source_dataset']+':'+r['label'] for r in rows))}
    manifest={'format_version':2,'seed':42,'sources':sources,'files':files,'skipped':dict(skipped),
              'historical_parent_model_commit':'38b71af7ca7007726e71e252bf72b9c420d02610',
              'parent_manifest_sha256':sha256(old/'data_manifest.json'),
              'sampling':'4000 MAVEN (80% natural, 20% balanced residual), 2000 Re-DocRED, 2000 FEVEROUS replay',
              'serialization':'Explicit event role/mention summary; full original document retained; over-budget rejected',
              'scope':'Candidate-conditioned public relations and gold-evidence verdicts, not operational validation',
              'split_policy':'Historical validation/calibration roles retained. Historical test unchanged, regression only. Confirmation is unused native dev with no group shared with any historical role. All roles disjoint.',
              'selection':'Lowest validation NLL; no selection or calibration on either test role',
              'planned_recipe':{'epochs':3,'lr':5e-6,'head_lr':1e-5,'effective_batch':16}}
    (out/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'files':files,'skipped':dict(skipped)},indent=2),flush=True)


if __name__=='__main__':main()
