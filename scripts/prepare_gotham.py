"""Prepare a bounded, provenance-tracked public decision corpus for Kaggle.

Use positive annotated pairs for relation-type selection, preserving native
source partitions. This is candidate-conditioned classification, not complete
relation extraction. Unknown relations are never converted to binary false.
Whole documents/evidence bundles are kept losslessly or rejected for length.
"""
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import random
import re
import time
import zipfile

from transformers import AutoTokenizer
from decision_moe.data import candidates, marker_sequence, target_distribution
from download_gotham_sources import sha256


def digest(value):
    """Stable ordering independent of Python's randomized hash seed."""
    return hashlib.sha256(str(value).encode()).hexdigest()


def wiki_group(title):
    """Normalize article titles for conservative cross-corpus overlap checks."""
    return 'wiki:' + ' '.join(title.replace('_', ' ').casefold().split())


def holdout(source_id):
    """Partition a source development document into selection/calibration/test."""
    value = int(digest(source_id)[:8], 16) % 10
    return 'validation' if value < 4 else 'calibration' if value < 6 else 'test'


def read_jsonl(path):
    """Read nonempty JSON objects from a native source file."""
    with Path(path).open() as file:
        for line in file:
            row = json.loads(line)
            if isinstance(row, dict) and row.get('claim'):
                yield row


def wrap(row, dataset, source_id, task, groups, split):
    """Attach audit metadata outside model-visible state/question/options."""
    row.update(source_dataset=dataset, source_id=str(source_id), task=task,
               split_groups=sorted(set(groups)), source_partition=split)
    row['id'] = dataset + ':' + digest(json.dumps(row, sort_keys=True))[:20]
    return row


def redocred_rows(documents, relation_names, partition):
    """Generate positive-pair relation choices with other known golds excluded.

    Distractors are relation types, under the revised annotation coverage
    assumption. They are not human-reviewed negative entity-pair judgments.
    At most eight decisions per source document limit expansion dominance.
    """
    all_relations = sorted(relation_names)
    for doc in sorted(documents, key=lambda d: digest(d['title'])):
        split = holdout(doc['title']) if partition == 'dev' else partition
        rng = random.Random(digest('redocred:' + doc['title']))
        by_pair = defaultdict(set)
        for annotation in doc['labels']:
            by_pair[annotation['h'], annotation['t']].add(annotation['r'])
        pairs = list(by_pair.items())
        rng.shuffle(pairs)
        emitted = 0
        for (head, tail), golds in pairs:
            if emitted >= 8:
                break
            gold = rng.choice(sorted(golds))
            distractors = [key for key in all_relations if key not in golds]
            keys = [gold, *rng.sample(distractors, 3)]
            rng.shuffle(keys)
            entities = []
            for index in [head, tail]:
                entities.append([{'name': m['name'], 'sentence': m['sent_id'], 'token_span': m['pos']}
                                 for m in doc['vertexSet'][index]])
            state = {'document': '\n'.join(f'{i}: ' + ' '.join(s) for i, s in enumerate(doc['sents'])),
                     'subject_mentions': entities[0], 'object_mentions': entities[1]}
            row = {'type': 'choice', 'state': state,
                   'question': 'Which listed relation is annotated from the subject entity to the object entity?',
                   'options': {key: relation_names[key] for key in keys}, 'label': gold}
            yield split, wrap(row, 'Re-DocRED', doc['title'], 'entity_relation', [wiki_group(doc['title'])], partition)
            emitted += 1


TEMPORAL = {
    'BEFORE': 'A occurs before B.', 'OVERLAP': 'A and B overlap in time.',
    'CONTAINS': 'The time interval of A contains B.', 'SIMULTANEOUS': 'A and B are simultaneous.',
    'BEGINS-ON': 'A begins when B begins.', 'ENDS-ON': 'A ends when B ends.'}
CAUSAL = {'CAUSE': 'A causes B according to the text.',
          'PRECONDITION': 'A is a precondition for B according to the text.'}


def maven_rows(path):
    """Read official ZIP members without extracting arbitrary archive paths."""
    with zipfile.ZipFile(path) as archive:
        for partition, filename in [('train', 'train.jsonl'), ('dev', 'valid.jsonl')]:
            member = next(name for name in archive.namelist() if Path(name).name == filename)
            with archive.open(member) as file:
                documents = [json.loads(line) for line in file if line.strip()]
            for doc in sorted(documents, key=lambda d: digest(d['id'])):
                split = holdout(doc['id']) if partition == 'dev' else 'train'
                mentions = {event['id']: event['mention'] for event in doc['events']}
                mentions.update({event['id']: [{'mention': event['mention'], 'sent_id': event['sent_id'],
                                               'offset': event['offset']}] for event in doc.get('TIMEX', [])})
                rng = random.Random(digest('maven:' + doc['id']))
                pairs = []
                for task, field, options in [('temporal_relation', 'temporal_relations', TEMPORAL),
                                             ('causal_relation', 'causal_relations', CAUSAL)]:
                    by_pair = defaultdict(set)
                    for label, relations in doc[field].items():
                        for pair in relations:
                            by_pair[tuple(pair)].add(label)
                    for pair, labels in by_pair.items():
                        if len(labels) == 1:
                            pairs.append((task, pair, next(iter(labels)), options))
                rng.shuffle(pairs)
                for task, (left, right), label, options in pairs[:12]:
                    state = {'document': '\n'.join(f'{i}: {s}' for i, s in enumerate(doc['sentences'])),
                             'A': [{k: v for k, v in m.items() if k in ('trigger_word', 'mention', 'sent_id', 'offset')}
                                   for m in mentions[left]],
                             'B': [{k: v for k, v in m.items() if k in ('trigger_word', 'mention', 'sent_id', 'offset')}
                                   for m in mentions[right]]}
                    keys = list(options)
                    rng.shuffle(keys)
                    row = {'type': 'choice', 'state': state,
                           'question': f'Which {task.replace("_", " ")} is annotated from A to B?',
                           'options': {key: options[key] for key in keys}, 'label': label}
                    yield split, wrap(row, 'MAVEN-ERE', doc['id'], task, [wiki_group(doc['title'])], partition)


ELEMENT = re.compile(r'^(.*?)_(header_cell|table_caption|sentence|cell|item|section)_([0-9_]+)$')


def element_page(identifier):
    """Split a FEVEROUS element identifier without breaking title underscores."""
    if identifier.endswith('_title'):
        return identifier[:-6], 'title'
    match = ELEMENT.fullmatch(identifier)
    if not match:
        raise ValueError('Unrecognized FEVEROUS evidence identifier')
    return match.group(1), identifier[len(match.group(1)) + 1:]


def feverous_pool(raw, per_label=1500):
    """Select a deterministic claim pool; do not mix native training/dev files."""
    pool = []
    for partition in ['train', 'dev']:
        by_label = defaultdict(list)
        for claim in read_jsonl(raw / f'feverous_{partition}_challenges.jsonl'):
            if claim.get('label') in ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO'] and claim.get('evidence'):
                by_label[claim['label']].append(claim)
        for claims in by_label.values():
            ordered = sorted(claims, key=lambda c: digest(c['id']))[:per_label if partition == 'train' else 600]
            pool.extend((partition, claim) for claim in ordered)
    return pool


def needed_elements(pool):
    """Collect only elements referenced by selected claims and their context."""
    needed = defaultdict(set)
    for _, claim in pool:
        for evidence in claim['evidence']:
            identifiers = list(evidence['content'])
            for elements in evidence.get('context', {}).values():
                identifiers.extend(elements)
            for identifier in identifiers:
                page, element = element_page(identifier)
                needed[page].add(element)
    return needed


def page_values(page):
    """Index sentence, section, cell, item and caption text in an official page."""
    values = {'title': page['title']}
    def walk(value):
        if isinstance(value, dict):
            if 'id' in value and 'value' in value:
                values[value['id']] = str(value['value'])
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    for key, value in page.items():
        if key.startswith('sentence_'):
            values[key] = value
        elif key.startswith('section_') and isinstance(value, dict):
            values[key] = value.get('value', '')
        elif key.startswith('table_') and isinstance(value, dict) and 'caption' in value:
            values['table_caption_' + key[6:]] = value['caption']
        walk(value)
    return values


def resolve_wiki(raw, pool):
    """Stream compressed Wikipedia pages, retaining only requested elements.

    The 9.9 GB archive is not fully extracted. Materialized evidence is cached
    alongside a digest of the required-ID set; missing evidence is rejected.
    """
    needed = needed_elements(pool)
    request_hash = digest(json.dumps({k: sorted(v) for k, v in needed.items()}, sort_keys=True))
    cache = raw / f'feverous-elements-{request_hash[:16]}.json'
    if cache.exists():
        return json.loads(cache.read_text())
    found, scanned = {}, 0
    # In the official page schema, title follows order before content elements.
    # Read that string first, avoiding materializing millions of unused tables.
    title_field = re.compile(br'(?<!\\)"title"\s*:\s*("(?:\\.|[^"\\])*")')
    started = last = time.monotonic()
    with zipfile.ZipFile(raw / 'feverous-wiki-pages.zip') as archive:
        for member in archive.namelist():
            if member.endswith('/'):
                continue
            with archive.open(member) as file:
                for line in file:
                    if not line.strip():
                        continue
                    match = title_field.search(line)
                    title = json.loads(match.group(1)) if match else json.loads(line)['title']
                    scanned += 1
                    if title in needed:
                        page = json.loads(line)
                        if page['title'] != title:
                            raise ValueError('Unexpected Wikipedia page layout')
                        values = page_values(page)
                        found[title] = {key: values[key] for key in needed[title] if key in values}
                    if time.monotonic() - last > 30:
                        print(json.dumps({'wiki_pages_scanned': scanned, 'required_pages_found': len(found),
                                          'required_pages': len(needed), 'elapsed_seconds': round(time.monotonic() - started)}), flush=True)
                        last = time.monotonic()
    cache.write_text(json.dumps(found, ensure_ascii=False))
    print(json.dumps({'wiki_pages_scanned': scanned, 'required_pages_found': len(found), 'required_pages': len(needed)}), flush=True)
    return found


def feverous_rows(pool, pages):
    """Resolve complete evidence sets, skipping bundles with missing elements."""
    options = {'SUPPORTS': 'The evidence supports the claim.', 'REFUTES': 'The evidence refutes the claim.',
               'NOT ENOUGH INFO': 'The evidence is insufficient to decide the claim.'}
    for partition, claim in pool:
        split = holdout('feverous:' + str(claim['id'])) if partition == 'dev' else 'train'
        for evidence in claim['evidence']:
            identifiers = list(evidence['content'])
            for values in evidence.get('context', {}).values():
                identifiers.extend(values)
            identifiers = list(dict.fromkeys(identifiers))
            bundle, groups = [], []
            for identifier in identifiers:
                page, element = element_page(identifier)
                if not pages.get(page, {}).get(element):
                    break
                text = re.sub(r'\[\[([^]|]+)\|([^]]+)\]\]', r'\2', pages[page][element])
                text = text.replace('[[', '').replace(']]', '')
                bundle.append({'source': page, 'element': element, 'text': text})
                groups.append(wiki_group(page))
            else:
                if not bundle:
                    continue
                row = {'type': 'choice', 'state': {'claim': claim['claim'], 'evidence': bundle},
                       'question': 'Which verdict is supported by this evidence for the claim?',
                       'options': options, 'label': claim['label']}
                yield split, wrap(row, 'FEVEROUS', claim['id'], 'evidence_verdict',
                                  [*groups, 'claim:' + str(claim['id'])], partition)
                break  # One complete alternative evidence set per claim.


def choose(rows, limit, balanced):
    """Deterministic bounded sampling, optionally round-robin by gold label."""
    ordered = sorted(rows, key=lambda r: digest(r['id']))
    if not balanced:
        return ordered[:limit]
    buckets = defaultdict(deque)
    for row in ordered:
        buckets[row['task'] + ':' + str(row['label'])].append(row)
    selected = []
    while buckets and len(selected) < limit:
        for label in sorted(list(buckets)):
            selected.append(buckets[label].popleft())
            if not buckets[label]:
                del buckets[label]
            if len(selected) == limit:
                break
    return selected


def prepare(generated, tokenizer, output, sources, max_length=1024, head_length=256,
            limits=None):
    """Validate, select, and remove cross-partition article/content overlap.

    Training is class-balanced within each source. Evaluation is deterministic
    sampling of the surviving source pool. Splits are conservative: any shared
    article/evidence group with an earlier partition removes a later example.
    """
    limits = limits or {'train': 2000, 'validation': 240, 'calibration': 120, 'test': 240}
    pools, skipped = defaultdict(list), Counter()
    for split, row in generated:
        try:
            keys, _ = candidates(row)
            target_distribution(row, keys)
            marker_sequence(tokenizer, row, max_length, head_length)
        except ValueError as error:
            skipped[row['source_dataset'] + ':' + str(error)] += 1
            continue
        pools[split, row['source_dataset']].append(row)
    output.mkdir(parents=True, exist_ok=True)
    occupied_groups, occupied_inputs = set(), set()
    files, stats = {}, {}
    for split in ['train', 'validation', 'calibration', 'test']:
        selected = []
        this_groups, this_inputs = set(), set()
        for dataset in ['Re-DocRED', 'MAVEN-ERE', 'FEVEROUS']:
            available = []
            for row in pools[split, dataset]:
                fingerprint = digest(json.dumps({k: row[k] for k in ['type', 'state', 'question', 'options']}, sort_keys=True))
                if occupied_groups.intersection(row['split_groups']) or fingerprint in occupied_inputs:
                    skipped[dataset + ':' + split + ':cross_split_overlap'] += 1
                    continue
                available.append(row)
            sample = choose(available, limits[split], split == 'train')
            selected.extend(sample)
            for row in sample:
                this_groups.update(row['split_groups'])
                this_inputs.add(digest(json.dumps({k: row[k] for k in ['type', 'state', 'question', 'options']}, sort_keys=True)))
        if not selected:
            raise ValueError(f'No usable examples in {split}')
        random.Random(42).shuffle(selected)
        path = output / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected))
        occupied_groups.update(this_groups)
        occupied_inputs.update(this_inputs)
        files[path.name] = {'sha256': sha256(path), 'rows': len(selected)}
        stats[split] = {'datasets': dict(Counter(r['source_dataset'] for r in selected)),
                        'tasks': dict(Counter(r['task'] for r in selected)),
                        'labels': dict(Counter(r['source_dataset'] + ':' + str(r['label']) for r in selected)),
                        'source_groups': len(this_groups)}
    manifest = {'format_version': 1, 'seed': 42, 'max_length': max_length, 'head_length': head_length,
                'sources': sources, 'files': files, 'counts': stats, 'skipped': dict(skipped),
                'scope': 'Candidate-conditioned positive-pair relation classification and gold-evidence verification; not full extraction or operational validation.',
                'label_policy': 'Re-DocRED distractors assume revised annotation coverage; other gold relations excluded. No unannotated pairs become binary false. MAVEN only singly labeled positive temporal/causal pairs. FEVEROUS complete evidence bundles only.',
                'split_policy': 'Native train boundaries; source dev partitioned by document hash; Re-DocRED native test retained. Shared Wikipedia-title groups and exact model-input fingerprints excluded across partitions.',
                'sampling_policy': 'At most 8 Re-DocRED and 12 MAVEN questions per document; per-source class-balanced training cap; deterministic evaluation sampling; no truncation.'}
    (output / 'data_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'files': files, 'counts': stats, 'skipped': dict(skipped)}, indent=2), flush=True)
    return manifest


def main():
    """Create four ready-to-train JSONL partitions plus an auditable manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', default='.local/raw')
    parser.add_argument('--output-dir', default='.local/prepared')
    parser.add_argument('--tokenizer-dir', default='/tmp/julia-1-checkpoint/tokenizer')
    args = parser.parse_args()
    if Path(args.output_dir).exists():
        raise ValueError('Choose a new output directory; prepared inputs are immutable')
    raw = Path(args.raw_dir)
    sources = json.loads((raw / 'sources.json').read_text())
    sources['maven-ere.zip'] = json.loads((raw / 'maven-ere.zip.receipt.json').read_text())
    for name in ['maven-ere.zip', 'feverous-wiki-pages.zip', 'train_revised.json', 'dev_revised.json', 'test_revised.json']:
        if sources[name]['sha256'] != sha256(raw / name):
            raise ValueError(f'Source checksum mismatch: {name}')
    relations = json.loads((raw / 'rel_info.json').read_text())
    pool = feverous_pool(raw)
    pages = resolve_wiki(raw, pool)
    def generated():
        for split, filename in [('train', 'train_revised.json'), ('dev', 'dev_revised.json'), ('test', 'test_revised.json')]:
            yield from redocred_rows(json.loads((raw / filename).read_text()), relations, split)
        yield from maven_rows(raw / 'maven-ere.zip')
        yield from feverous_rows(pool, pages)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    prepare(generated(), tokenizer, Path(args.output_dir), sources)


if __name__ == '__main__':
    main()
