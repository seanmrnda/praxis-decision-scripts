import copy
import tempfile
import unittest

import torch
from transformers import ModernBertConfig, ModernBertModel
from decision_moe import DecisionMoE, JuliaDecisionModel, SparseMLP
from decision_moe.data import candidates, target_distribution
from decision_moe.model import decision_loss


def tiny_encoder():
    config = ModernBertConfig(vocab_size=64, hidden_size=16, intermediate_size=24,
                             num_hidden_layers=2, num_attention_heads=2,
                             max_position_embeddings=128, local_attention=16,
                             pad_token_id=0, cls_token_id=1, sep_token_id=2,
                             bos_token_id=1, eos_token_id=2, mask_token_id=4,
                             reference_compile=False, mlp_dropout=0.0,
                             attention_dropout=0.0, embedding_dropout=0.0)
    config._attn_implementation = "sdpa"
    return ModernBertModel(config)


def tiny_julia():
    return JuliaDecisionModel(tiny_encoder(), head_layers=2, n_act=2, dropout=0.1)


def marker_batch():
    ids = torch.tensor([[1, 6, 2, 4, 7, 4, 8, 2, 9, 2, 0, 0],
                        [1, 6, 2, 4, 7, 4, 8, 4, 10, 2, 9, 2]])
    return dict(input_ids=ids, attention_mask=ids.ne(0),
                marker_pos=torch.tensor([[3, 5, 0], [3, 5, 7]]),
                marker_mask=torch.tensor([[True, True, False], [True, True, True]]),
                qtype=torch.tensor([0, 1]))


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_upcycling_preserves_julia_scores_components_and_checkpoint(self):
        dense = tiny_julia().eval()
        dense.temperature.copy_(torch.tensor([1.2, 1.7, 0.8]))
        moe = DecisionMoE(copy.deepcopy(dense), [0, 1], 3, 2).eval()
        batch = marker_batch()
        with torch.no_grad():
            expected = dense(**batch)
            actual, _ = moe(**batch)
            valid = batch["marker_mask"]
            torch.testing.assert_close(actual[valid], expected[valid], atol=2e-6, rtol=2e-5)
            for component in ("head", "type_emb", "scorer", "act_head"):
                for key, value in getattr(dense, component).state_dict().items():
                    torch.testing.assert_close(value, getattr(moe, component).state_dict()[key], rtol=0, atol=0)
            torch.testing.assert_close(moe.temperature, dense.temperature)
            with tempfile.TemporaryDirectory() as tmp:
                moe.save_pretrained(tmp)
                restored = DecisionMoE.from_pretrained(tmp).eval()
                torch.testing.assert_close(restored(**batch)[0], actual)
                torch.testing.assert_close(restored.temperature, dense.temperature)

    def test_sparse_routing_gradients_and_padding(self):
        model = DecisionMoE(tiny_julia(), [1], 3, 2)
        batch = marker_batch()
        logits, aux = model(**batch)
        targets = torch.tensor([[0., 1., 0.], [0., 0., 1.]])
        loss = decision_loss(logits, targets) + 0.01 * aux
        loss.backward()
        block = model.moe_blocks[0]
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(block.router.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.scorer[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.type_emb.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        model.eval()
        with torch.no_grad():
            row = {key: value[:1] for key, value in batch.items()}
            padded, _ = model(**row)
            row["input_ids"] = row["input_ids"][:, :10]
            row["attention_mask"] = row["attention_mask"][:, :10]
            unpadded, _ = model(**row)
            torch.testing.assert_close(unpadded, padded, atol=2e-6, rtol=2e-5)

    def test_sparse_dispatch_counts_only_valid_tokens(self):
        block = SparseMLP(tiny_encoder().layers[0].mlp, 3, 2).eval()
        seen = [0, 0, 0]
        handles = []
        for i, expert in enumerate(block.experts):
            def hook(module, inputs, i=i):
                seen[i] += inputs[0].shape[0]
            handles.append(expert.register_forward_pre_hook(hook))
        block.valid_mask = torch.tensor([True, False, True, False, True, True])
        block(torch.randn(2, 3, 16))
        self.assertEqual(sum(seen), 4 * 2)
        for handle in handles:
            handle.remove()

    def test_typed_targets_and_native_limits(self):
        choice = {"state": "x", "question": "route", "options": {"a": "A", "b": "B"}, "label": "b"}
        torch.testing.assert_close(target_distribution(choice, candidates(choice)[0]), torch.tensor([0., 1.]))
        noul = {"state": "x", "question": "true?", "type": "noul", "label": .3}
        self.assertEqual(candidates(noul)[1], ["false", "true"])
        torch.testing.assert_close(target_distribution(noul, candidates(noul)[0]), torch.tensor([.7, .3]))
        score = {"state": "x", "question": "rate", "type": "score",
                 "levels": [{"value": 0, "description": "low"}, {"value": 2, "description": "mid"}, {"value": 5, "description": "high"}], "label": 3.5}
        torch.testing.assert_close(target_distribution(score, candidates(score)[0]), torch.tensor([0., .5, .5]))
        for malformed in (dict(choice, label="c"), dict(noul, label=1.1), dict(score, label=9), dict(choice, target_probs=[.1, .1])):
            with self.assertRaises(ValueError):
                target_distribution(malformed, candidates(malformed)[0])
        with self.assertRaises(ValueError):
            candidates(dict(choice, options={str(i): "option" for i in range(21)}))


if __name__ == "__main__":
    unittest.main()
