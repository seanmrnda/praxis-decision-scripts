"""Protect evidence resolution, annotation meaning and split isolation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_gotham import element_page, page_values, prepare, redocred_rows, maven_rows, wrap
from test_pipeline import tiny_tokenizer


class GothamDataTests(unittest.TestCase):
    def test_evidence_ids_preserve_title_underscores_and_table_headers(self):
        self.assertEqual(element_page('Some_Page_header_cell_0_1_2'), ('Some_Page', 'header_cell_0_1_2'))
        self.assertEqual(element_page('Some_Page_title'), ('Some_Page', 'title'))
        page = {'title': 'Some_Page', 'sentence_0': 'Fact.', 'section_0': {'value': 'History'},
                'table_0': {'caption': 'Fleet', 'table': [[{'id': 'header_cell_0_0_0', 'value': 'Year'},
                                                        {'id': 'cell_0_0_1', 'value': '2020'}]]}}
        values = page_values(page)
        self.assertEqual(values['header_cell_0_0_0'], 'Year')
        self.assertEqual(values['cell_0_0_1'], '2020')
        self.assertEqual(values['table_caption_0'], 'Fleet')

    def test_other_gold_relations_are_never_distractors(self):
        doc = {'title': 'Example', 'sents': [['Alice', 'works', 'here']],
               'vertexSet': [[{'name': 'Alice', 'sent_id': 0, 'pos': [0, 1]}],
                             [{'name': 'Here', 'sent_id': 0, 'pos': [2, 3]}]],
               'labels': [{'h': 0, 't': 1, 'r': 'P1'}, {'h': 0, 't': 1, 'r': 'P2'}]}
        relations = {f'P{i}': f'Relation {i}' for i in range(1, 7)}
        _, row = next(redocred_rows([doc], relations, 'train'))
        other_gold = ({'P1', 'P2'} - {row['label']}).pop()
        self.assertNotIn(other_gold, row['options'])
        self.assertNotIn('labels', row['state'])

    def test_maven_timex_strings_become_explicit_time_mentions(self):
        doc = {'id': 'x', 'title': 'Example', 'sentences': ['It opened in 2020.'],
               'events': [{'id': 'E', 'mention': [{'trigger_word': 'opened', 'sent_id': 0, 'offset': [1, 2]}]}],
               'TIMEX': [{'id': 'T', 'mention': '2020', 'sent_id': 0, 'offset': [3, 4]}],
               'temporal_relations': {'BEFORE': [['E', 'T']]}, 'causal_relations': {}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'maven.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                for name in ['train.jsonl', 'valid.jsonl']:
                    archive.writestr(name, json.dumps(doc) + '\n')
            _, row = next(maven_rows(path))
        self.assertEqual(row['state']['B'][0]['mention'], '2020')
        self.assertEqual(row['label'], 'BEFORE')

    def test_shared_article_removed_from_later_partition(self):
        def row(identifier, group, split):
            return split, wrap({'type': 'choice', 'state': identifier, 'question': 'route',
                                'options': {'a': 'A', 'b': 'B'}, 'label': 'a'},
                               'Re-DocRED', identifier, 'relation', [group], split)
        generated = [row('train', 'shared', 'train'), row('leak', 'shared', 'validation'),
                     row('valid', 'v', 'validation'), row('cal', 'c', 'calibration'), row('test', 't', 'test')]
        with tempfile.TemporaryDirectory() as tmp:
            manifest = prepare(generated, tiny_tokenizer(), Path(tmp), {})
            valid = [json.loads(line) for line in (Path(tmp) / 'validation.jsonl').read_text().splitlines()]
        self.assertEqual([r['source_id'] for r in valid], ['valid'])
        self.assertEqual(manifest['skipped']['Re-DocRED:validation:cross_split_overlap'], 1)


if __name__ == '__main__':
    unittest.main()
