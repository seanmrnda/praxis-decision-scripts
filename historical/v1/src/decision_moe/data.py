"""One JSONL row = state + one independent typed question + optional target."""
import json
import math
from pathlib import Path

import torch

QTYPES = {"choice": 0, "score": 1, "noul": 2}


def candidates(row):
    """Validate a typed question and return ordered candidate keys/descriptions.

    Choice keys follow map insertion order, score keys are increasing floats,
    and Noul keys are [false, true]. This checks schema, not tokenizer budgets.
    IDs and labels do not enter candidate description text.
    """
    if not isinstance(row, dict):
        raise ValueError("Each request must be a JSON object")
    kind = row.get("type", "choice")
    if not isinstance(row.get("question"), str) or not row["question"].strip():
        raise ValueError("question must be a nonempty string")
    if "state" not in row:
        raise ValueError("state is required")
    if kind == "choice":
        options = row.get("options")
        if not isinstance(options, dict) or not 2 <= len(options) <= 20:
            raise ValueError("choice options must map 2..20 keys to descriptions")
        if any(not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip() for k, v in options.items()):
            raise ValueError("option keys and descriptions must be nonempty strings")
        return list(options), list(options.values())
    if kind == "noul":
        descriptions = [row.get("false_criteria", "false"), row.get("true_criteria", "true")]
        if any(not isinstance(s, str) or not s.strip() for s in descriptions):
            raise ValueError("noul criteria must be nonempty strings")
        return ["false", "true"], descriptions
    if kind == "score":
        levels = row.get("levels")
        if not isinstance(levels, list) or not 2 <= len(levels) <= 20:
            raise ValueError("score levels must contain 2..20 {value, description} objects")
        values = [float(level["value"]) for level in levels]
        if any(not math.isfinite(v) for v in values) or any(a >= b for a, b in zip(values, values[1:])):
            raise ValueError("score values must be finite and strictly increasing")
        descriptions = [level["description"] for level in levels]
        if any(not isinstance(s, str) or not s.strip() for s in descriptions):
            raise ValueError("score descriptions must be nonempty strings")
        return values, descriptions
    raise ValueError(f"Unsupported type: {kind}")


def target_distribution(row, keys):
    """Convert a validated row's supervision into a float target vector.

    target_probs takes precedence over label. Choice labels become one-hot;
    Noul probabilities become [1-p,p]; fractional scores interpolate mass
    between adjacent numeric levels. ``keys`` must be from candidates(row).
    This function validates target values but does not normalize bad inputs.
    """
    n = len(keys)
    if "target_probs" in row:
        p = torch.tensor(row["target_probs"], dtype=torch.float32)
        if p.shape != (n,) or not torch.isfinite(p).all() or (p < 0).any() or not torch.isclose(p.sum(), torch.tensor(1.), atol=1e-5):
            raise ValueError("target_probs must be a finite nonnegative vector summing to 1 in candidate order")
        return p
    label = row["label"]
    kind = row.get("type", "choice")
    if kind == "choice":
        if label not in keys:
            raise ValueError("choice label must be an option key")
        return torch.nn.functional.one_hot(torch.tensor(keys.index(label)), n).float()
    label = float(label)
    if not math.isfinite(label):
        raise ValueError("label must be finite")
    if kind == "noul":
        if not 0 <= label <= 1:
            raise ValueError("noul label must be in [0, 1]")
        return torch.tensor([1 - label, label])
    if not keys[0] <= label <= keys[-1]:
        raise ValueError("score label outside level range")
    p = torch.zeros(n)
    for i, value in enumerate(keys):
        if label == value:
            p[i] = 1
            return p
        if label < value:
            weight = (label - keys[i - 1]) / (value - keys[i - 1])
            p[i - 1], p[i] = 1 - weight, weight
            return p
    raise ValueError("Invalid score label")


def read_rows(path, labeled=True):
    """Read nonblank JSONL rows, validating schema and optional supervision.

    Returns all rows as a list; this is not a streaming loader. With labeled
    True, every row needs label or target_probs. Parse/schema errors include
    the physical file line number. Token-budget checks occur in the collator.
    Empty datasets are rejected.
    """
    rows = []
    for line_num, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            keys, _ = candidates(row)
            if labeled:
                target_distribution(row, keys)
            rows.append(row)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}:{line_num}: {exc}") from exc
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    return rows


def marker_sequence(tokenizer, row, max_length=1024, head_length=512):
    """Encode one question using Julia's marker layout without truncation.

    Returns ids, marker positions, and integer qtype. State strings are used
    directly; other JSON values are serialized. Option descriptions are the
    candidate text; choice IDs and numeric score values are not encoded.
    Enforces the 48-token option limit, reserved MASK exclusion, a lossless
    head budget with at least 16 question slots, and the total state budget.
    The caller must also enforce the encoder's maximum positional length.
    """
    if head_length < 1 or head_length + 4 >= max_length:
        raise ValueError("max_length must leave room beyond the question/options head")
    if any(x is None for x in (tokenizer.mask_token_id, tokenizer.cls_token_id,
                               tokenizer.sep_token_id, tokenizer.pad_token_id)):
        raise ValueError("Julia tokenizer must define MASK, CLS, SEP and PAD tokens")
    _, descriptions = candidates(row)
    state = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"], ensure_ascii=False)
    texts = [state, row["question"], *descriptions]
    if any(tokenizer.mask_token in text for text in texts):
        raise ValueError("Reserved model marker in request")
    encode = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]
    head = encode(f"{row.get('type', 'choice')} question: {row['question']}")
    option_ids = [encode(" " + description) for description in descriptions]
    if any(len(ids) > 48 for ids in option_ids):
        raise ValueError("Option exceeds Julia's 48-token contract")
    options = [[tokenizer.mask_token_id] + ids for ids in option_ids]
    budget = head_length - sum(map(len, options))
    # Julia's strict runtime requires lossless options and at least 16 head slots.
    if budget < 16 or len(head) > budget:
        raise ValueError("Question/options exceed lossless head budget; increase --head-length")
    ids = [tokenizer.cls_token_id] + head + [tokenizer.sep_token_id]
    markers = []
    for option in options:
        markers.append(len(ids))
        ids.extend(option)
    ids.append(tokenizer.sep_token_id)
    state_ids = encode(state)
    room = max_length - len(ids) - 1
    if room < 1 or len(state_ids) > room:
        raise ValueError("State exceeds lossless context budget; increase --max-length")
    return dict(ids=ids + state_ids + [tokenizer.sep_token_id], markers=markers,
                qtype=QTYPES[row.get("type", "choice")])


class DecisionCollator:
    """Build padded Julia tensors from typed JSONL rows.

    One row becomes one sequence, with all candidates in that sequence.
    Padding and candidate masks keep batch members independent. The collator
    is suitable as torch DataLoader's collate_fn.
    """
    def __init__(self, tokenizer, max_length=1024, head_length=512, labeled=True):
        """Configure tokenizer, total/head budgets, and whether to build targets.

        ``max_length`` must leave space beyond ``head_length`` and special
        tokens. Actual option/state checks happen when rows are encoded.
        """
        self.tokenizer, self.max_length = tokenizer, max_length
        self.head_length, self.labeled = head_length, labeled
        if head_length < 1 or head_length + 4 >= max_length:
            raise ValueError("max_length must leave room beyond the question/options head")

    def __call__(self, rows):
        """Return CPU model-input tensors and optional padded target vectors.

        input_ids/attention_mask: [B,L]; marker_pos/marker_mask: [B,K];
        qtype: [B]. L rounds up to eight within max_length and K is the largest
        candidate count. When labeled, targets [B,K] have zero padding.
        """
        encoded = [marker_sequence(self.tokenizer, row, self.max_length, self.head_length) for row in rows]
        length = min(self.max_length, ((max(len(item["ids"]) for item in encoded) + 7) // 8) * 8)
        count = max(len(item["markers"]) for item in encoded)
        ids = torch.full((len(rows), length), self.tokenizer.pad_token_id, dtype=torch.long)
        attention = torch.zeros_like(ids)
        positions = torch.zeros((len(rows), count), dtype=torch.long)
        mask = torch.zeros_like(positions, dtype=torch.bool)
        for i, item in enumerate(encoded):
            n, k = len(item["ids"]), len(item["markers"])
            ids[i, :n] = torch.tensor(item["ids"])
            attention[i, :n] = 1
            positions[i, :k] = torch.tensor(item["markers"])
            mask[i, :k] = True
        batch = dict(input_ids=ids, attention_mask=attention, marker_pos=positions,
                     marker_mask=mask, qtype=torch.tensor([item["qtype"] for item in encoded]))
        if self.labeled:
            targets = [target_distribution(row, candidates(row)[0]) for row in rows]
            batch["targets"] = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)
        return batch
