"""Julia-1 FFN upcycling, sparse routing, raw decision scoring, and checkpoints."""
import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from transformers import ModernBertConfig, ModernBertModel
from .julia import JuliaDecisionModel


class SparseMLP(nn.Module):
    """Sparse top-k replacement for one pretrained encoder FFN.

    Experts are independent copies of the dense FFN. ``valid_mask`` and
    ``aux_loss`` are temporary forward state managed by ``DecisionMoE``.
    This module is not safe for overlapping forwards on the same instance.
    """
    def __init__(self, dense, num_experts, top_k, noise_std=0.0):
        """Clone ``dense`` into ``num_experts`` FFNs and initialize a router.

        ``dense`` must expose ModernBERT's ``Wi`` input projection. ``top_k``
        must be in [1, num_experts]. Nonzero ``noise_std`` perturbs expert
        parameters and removes initial equivalence to the dense FFN.
        """
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("Require 1 <= top_k <= num_experts")
        self.experts = nn.ModuleList([copy.deepcopy(dense) for _ in range(num_experts)])
        self.router = nn.Linear(dense.Wi.in_features, num_experts, bias=False)
        self.observe_routing = False
        self.routing_stats = None
        nn.init.normal_(self.router.weight, std=0.02)
        self.top_k = top_k
        self.valid_mask = None
        self.aux_loss = None
        # Zero noise preserves the original FFN in eval mode, independently of routing.
        if noise_std:
            with torch.no_grad():
                for expert in self.experts:
                    for p in expert.parameters():
                        p.add_(torch.randn_like(p) * noise_std)

    def forward(self, hidden_states):
        """Return routed FFN outputs with the same shape as ``hidden_states``.

        Inputs are normally [batch, sequence, hidden]. A flattened Boolean
        ``valid_mask`` excludes padding; if absent, all tokens are routed.
        Only selected expert/token pairs execute. The differentiable balance
        plus z-loss is stored in ``aux_loss`` for the enclosing model to read.
        Padding outputs are zero, and no valid tokens are capacity-dropped.
        """
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        valid = self.valid_mask
        if valid is None:
            valid = torch.ones(flat.shape[0], dtype=torch.bool, device=flat.device)
        if valid.numel() != flat.shape[0]:
            raise ValueError("MoE mask must match padded encoder layout")
        indices = valid.reshape(-1).nonzero(as_tuple=True)[0]
        x = flat[indices]
        if not len(x):
            self.aux_loss = flat.sum() * 0
            return torch.zeros_like(hidden_states)
        with torch.autocast(device_type=x.device.type, enabled=False):
            router_logits = F.linear(x.float(), self.router.weight.float())
            probs = router_logits.softmax(-1)
        selected_probs, selected = probs.topk(self.top_k, dim=-1)
        weights = selected_probs / selected_probs.sum(-1, keepdim=True)
        if self.observe_routing:
            counts = F.one_hot(selected, len(self.experts)).sum((0, 1)).detach().cpu()
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1).sum().detach().item()
            if self.routing_stats is None:
                self.routing_stats = {'valid_tokens': 0, 'selection_counts': [0] * len(self.experts), 'entropy_sum': 0.}
            self.routing_stats['valid_tokens'] += len(x)
            self.routing_stats['selection_counts'] = [a + int(b) for a, b in zip(self.routing_stats['selection_counts'], counts)]
            self.routing_stats['entropy_sum'] += entropy
        mixed = torch.zeros_like(x)
        for expert_id, expert in enumerate(self.experts):
            tokens, slots = (selected == expert_id).nonzero(as_tuple=True)
            if tokens.numel():
                values = expert(x[tokens]) * weights[tokens, slots, None].to(x.dtype)
                mixed.index_add_(0, tokens, values.to(mixed.dtype))
        # Switch-style balancing plus router z-loss; padding never contributes.
        load = F.one_hot(selected, len(self.experts)).float().mean(dim=(0, 1))
        balance = len(self.experts) * (load * probs.mean(0)).sum()
        z_loss = router_logits.logsumexp(-1).square().mean()
        self.aux_loss = balance + 0.001 * z_loss
        output = torch.zeros_like(flat).index_copy(0, indices, mixed)
        return output.reshape_as(hidden_states)


MODEL_INPUTS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")


def model_inputs(batch):
    """Select the five forward-input tensors from a collator batch.

    Training batches may also contain ``targets``; those are loss inputs,
    not arguments to ``DecisionMoE.forward``. Missing required fields raise
    KeyError, and tensors are returned unchanged on their existing device.
    """
    return {k: batch[k] for k in MODEL_INPUTS}


class DecisionMoE(JuliaDecisionModel):
    """Julia's pretrained decision modules with sparse encoder FFNs.

    The constructor reuses the supplied dense model's modules and mutates
    its encoder by replacing selected FFNs. Pass a deep copy if the original
    dense model must remain independently usable. Temperature is a copied
    three-element buffer; forward returns raw scores, not probabilities.
    """
    def __init__(self, dense, moe_layers, num_experts=4, top_k=2, noise_std=0.0):
        """Upcycle distinct zero-based encoder layers of ``dense``.

        Preserve Julia head/scorer/type/action weights and temperatures.
        Expert weights are cloned; router weights are new. ``dense`` must
        use a ModernBERT-compatible encoder. No checkpoint is read here.
        """
        nn.Module.__init__(self)
        encoder = dense.encoder
        if encoder.config.model_type != "modernbert":
            raise ValueError("Julia MoE currently requires its ModernBERT/mmBERT encoder")
        if not moe_layers or len(set(moe_layers)) != len(moe_layers):
            raise ValueError("Select distinct, nonempty MoE layer indices")
        if any(i < 0 or i >= len(encoder.layers) for i in moe_layers):
            raise ValueError("MoE layer index outside encoder")
        self.encoder = encoder
        # Reuse all trained Julia modules instead of adding a random decision head.
        self.head, self.type_emb = dense.head, dense.type_emb
        self.scorer, self.act_head = dense.scorer, dense.act_head
        self.settings = dict(dense.settings)
        self.head_checkpointing = False
        self.marker_only_head = False
        self.register_buffer("temperature", dense.temperature.detach().clone())
        self.spec = dict(moe_layers=list(moe_layers), num_experts=num_experts, top_k=top_k)
        self.model_name = "praxis"
        for i in moe_layers:
            encoder.layers[i].mlp = SparseMLP(encoder.layers[i].mlp, num_experts, top_k, noise_std)

    @property
    def moe_blocks(self):
        """Return the sparse blocks in configured layer order.
        """
        return [self.encoder.layers[i].mlp for i in self.spec["moe_layers"]]

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        """Return ``(logits, router_aux)`` for one sequence per question.

        Shapes: input_ids/attention_mask [B,L], marker_pos/marker_mask [B,K],
        qtype [B]. Type IDs are choice=0, score=1, noul=2. Logits are FP32
        [B,K], with invalid candidates masked to negative infinity. Auxiliary
        loss is a scalar averaged across converted layers.

        This method sets per-block masks, reads auxiliary graphs, and clears
        temporary state in a finally block. Overlapping forwards on this
        object are unsupported. Gradient checkpointing is rejected. Apply
        the saved per-type temperature and softmax outside this method.
        """
        if self.encoder.is_gradient_checkpointing or self.head_checkpointing:
            raise ValueError("Gradient checkpointing is unsupported with router auxiliary losses")
        try:
            for block in self.moe_blocks:
                block.valid_mask = attention_mask.bool().reshape(-1)
            logits = super().forward(input_ids, attention_mask, marker_pos, marker_mask, qtype)
            # Julia's native forward uses -1e4; use exact masking for decision losses.
            logits = logits.masked_fill(~marker_mask, float("-inf"))
            aux = torch.stack([block.aux_loss for block in self.moe_blocks]).mean()
            return logits, aux
        finally:
            for block in self.moe_blocks:
                block.valid_mask = None
                block.aux_loss = None

    def save_pretrained(self, directory):
        """Write complete model weights and Julia MoE format-2 configuration.

        Creates the destination directory and overwrites existing model/config
        files. Tokenizer and provenance/report files are saved separately by
        CLI callers. This saves no optimizer or exact-resume state.
        """
        super().save_pretrained(directory)
        path = Path(directory)
        julia_config = json.loads((path / "julia_config.json").read_text())
        julia_config["architecture"] = "JuliaDecisionMoE"
        (path / "julia_config.json").write_text(json.dumps(julia_config, indent=2) + "\n")
        (path / "decision_config.json").write_text(json.dumps(
            dict(format_version=2, architecture="JuliaDecisionMoE", **self.spec), indent=2) + "\n")
        manifest = dict(format_version=2, architecture="JuliaDecisionMoE",
                        model_name=self.model_name,
                        julia_config_file="julia_config.json", encoder_config_file="encoder/config.json",
                        moe_config_file="decision_config.json", weights_file="model.safetensors",
                        tokenizer_directory="tokenizer")
        (path / "config.json").write_text(json.dumps(manifest, indent=2) + "\n")

    @classmethod
    def from_pretrained(cls, directory):
        """Build a local Julia MoE checkpoint and load all tensors strictly.

        Reads encoder/head/expert settings and reconstructs their shapes.
        Dense Julia and earlier ModernBERT formats are rejected. Temperatures
        must be finite and positive. The returned model is on CPU in training
        mode; callers choose device and call eval() for inference.
        """
        path = Path(directory)
        spec = json.loads((path / "decision_config.json").read_text())
        if spec.pop("format_version", None) != 2 or spec.pop("architecture", None) != "JuliaDecisionMoE":
            raise ValueError("Expected a Julia MoE checkpoint; reconvert older ModernBERT checkpoints")
        config = ModernBertConfig.from_pretrained(path / "encoder", local_files_only=True)
        config._attn_implementation = "sdpa"
        config.reference_compile = False
        julia_config = json.loads((path / "julia_config.json").read_text())
        if julia_config.get("architecture") != "JuliaDecisionMoE":
            raise ValueError("Invalid Julia MoE architecture")
        settings = {k: julia_config[k] for k in ("head_layers", "n_act", "dropout")}
        from transformers.initialization import no_init_weights
        with no_init_weights():
            dense = JuliaDecisionModel(ModernBertModel(config), **settings)
            model = cls(dense, **spec)
        model.model_name = json.loads((path / "config.json").read_text()).get("model_name", "praxis")
        model.load_state_dict(load_file(str(path / "model.safetensors")), strict=True)
        if not torch.isfinite(model.temperature).all() or (model.temperature <= 0).any():
            raise ValueError("Checkpoint temperatures must be finite and positive")
        return model


def decision_loss(logits, targets):
    """Mean candidate cross-entropy for padded logits and target distributions.

    ``logits`` and ``targets`` are [B,K]; targets must already be validated
    nonnegative distributions with zero padding. Zero-target log-probabilities
    are masked before multiplication to avoid 0 * -infinity. Temperatures
    and router/Brier penalties are applied by the caller, not this function.
    """
    log_probs = logits.log_softmax(-1)
    # Avoid 0 * -inf for padded candidates.
    return -(targets * log_probs.masked_fill(targets == 0, 0)).sum(-1).mean()
