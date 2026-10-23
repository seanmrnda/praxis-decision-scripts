"""Verify new experiment isolation and meaningful evaluation diagnostics."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_improvement import enhance_event, tempered_sample, assert_disjoint
from evaluate_gotham import evaluate_checkpoint
from test_model import tiny_julia
from test_pipeline import tiny_tokenizer
from decision_moe import DecisionMoE


class ImprovementTests(unittest.TestCase):
    def test_role_summary_preserves_document_direction_and_label(self):
        row={'source_dataset':'MAVEN-ERE','task':'temporal_relation','label':'BEFORE',
             'state':{'document':'Original complete document.',
                      'A':[{'trigger_word':'opened','sent_id':0,'offset':[1,2]}],
                      'B':[{'mention':'2020','sent_id':0,'offset':[3,4]}]}}
        improved=enhance_event(row)
        self.assertEqual(improved['state']['document'],row['state']['document'])
        self.assertIn('A:',improved['state']['pair_summary'])
        self.assertIn('FROM A TO B',improved['question'])
        self.assertEqual(improved['label'],'BEFORE')
        self.assertNotIn('pair_summary',row['state'])

    def test_tempered_sampling_is_unique_and_split_overlap_rejected(self):
        rows=[{'id':str(i),'task':'temporal','label':'BEFORE' if i<80 else 'OVERLAP'} for i in range(100)]
        sample=tempered_sample(rows,40)
        self.assertEqual(len(sample),40)
        self.assertEqual(len({r['id'] for r in sample}),40)
        with self.assertRaises(ValueError):
            assert_disjoint({'train':[{'split_groups':['x']}], 'confirmation':[{'split_groups':['x']}]})

    def test_dense_and_moe_diagnostics_with_real_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); dense=tiny_julia();dense.save_pretrained(root/'dense')
            tiny_tokenizer().save_pretrained(root/'dense/tokenizer')
            moe=DecisionMoE(dense,[0],num_experts=3,top_k=2);moe.save_pretrained(root/'moe')
            tiny_tokenizer().save_pretrained(root/'moe/tokenizer')
            rows=[{'type':'choice','id':str(i),'source_dataset':'test','source_id':str(i//2),
                   'task':'relation','state':'state','question':'route','options':{'a':'A','b':'B'},
                   'label':'a' if i<3 else 'b'} for i in range(4)]
            (root/'test.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            for kind in ['dense','moe']:
                report=evaluate_checkpoint(root/kind,root/'test.jsonl',root/(kind+'.json'),device='cpu',
                                           model_kind=kind,predictions_file=root/(kind+'.jsonl'))
                self.assertEqual(report['slices']['all']['majority_accuracy'],.75)
                self.assertEqual(report['slices']['all']['source_groups'],2)
                self.assertEqual(len((root/(kind+'.jsonl')).read_text().splitlines()),4)
                self.assertEqual(sum(sum(c.values()) for c in report['slices']['all']['confusion'].values()),4)
                if kind=='moe':
                    stats=report['routing']['0'];self.assertEqual(sum(stats['selection_counts']),2*stats['valid_tokens'])
