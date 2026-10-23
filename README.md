# Praxis Decision: experiment reproduction

Code and instructions for **Praxis Decision: Sparse Upcycling and Committee Distillation of Julia-1 for Typed Decisions**, by **Sean Lester Miranda** ([contact@seanmiranda.com](mailto:contact@seanmiranda.com)).

This repository reproduces the original Julia-1 → sparse MoE conversion and the v1, v2, and v3 experiments completed on September 28, 2026. Historical checkpoints and reports use `defense-praxis` names; the model family is **Praxis Decision**. Later experiments that also used “Praxis-v3,” and the Laya, v4, v5, and H/P runs, are outside this repository's scope.

The pipeline starts with public Julia-1 weights and downloads the public training sources. Access to the author's private checkpoints is **not required**. Model weights, dataset copies, credentials, and cloud-account automation are excluded from this source repository.

## What is reproduced

```text
Julia-1 (dense, pinned commit)
  └─ clone FFNs into four experts; route top two; eleven alternating layers
      └─ untuned sparse checkpoint
          └─ v1: 6,000 training questions × 2 epochs
              ├─ control A: original 6,000 × 4 epochs
              ├─ v2 (B): revised 8,000 × 3 epochs
              ├─ control C: revised 8,000 × 1 epoch, hard labels
              └─ v3 student: same 8,000 × 1 epoch, committee distillation
                   teachers = v1 + control A + v2, fixed task-conditioned weights
```

**v2, control A, control C, and the v3 student all initialize from v1.** v3 does not continue from v2. C and v3 share training, validation, calibration, initialization, and planned optimization budget. The distillation treatment changes both gold-CE weighting and the KL term.

Core settings:

| Setting | Reported value |
| --- | --- |
| Base | `SupersonicLabs/Julia-1`, revision `a85b127321d580d65176c89ced8273f305745d85` |
| Encoder | 22 layers, hidden 384, FFN 1,152, 6 attention heads |
| MoE layers | Zero-based `1,3,5,7,9,11,13,15,17,19,21` |
| Experts / active experts | 4 / 2; full pretrained FFN clones, no expert noise |
| Parameters | Dense 144,292,867; sparse 188,104,195 |
| Preserved Julia components | Encoder, decision head, type embeddings, scorer, action head, temperature |
| Serialization | One sequence per question; 1,024 total tokens, 256 head tokens, 48 per option; reject overflow |
| Training | Seed 42; microbatch 4; accumulation 4; effective batch 16 |
| Optimizer | AdamW; decay 0.01 on matrix parameters; cosine schedule; 10% warmup; clipping 1.0 |
| v1 LR | Encoder `1e-5`; head/router `2e-5` |
| Other arms' LR | Encoder `5e-6`; head/router `1e-5` |
| Hard-label loss | Gold CE + `0.1 × Brier + 0.01 × router auxiliary` |
| v3 loss | `0.5 × gold CE + 0.5 × T² KL(teacher || student)` + same Brier/auxiliary; `T=2` |
| Selection / calibration | Minimum gold validation NLL; then temperature fitting on separate calibration rows |
| Precision / devices | FP32 parameters and optimizer state, FP16 autocast + GradScaler; question-level DataParallel on 2 Tesla T4 GPUs |

The action head is retained and frozen. This study trains choice questions; it does not establish score/no-utility task performance. See the archived implementations for the exact loss, parameter groups, routing, calibration, and batching behavior.

## Environment and installation

Run commands from this repository's root on Linux. The original environment was **Python 3.12.13, PyTorch 2.10.0+cu128, CUDA 12.8, Transformers 5.17.0**, with two 16 GB Tesla T4 GPUs. The orchestrator requires exactly two visible CUDA GPUs for training. Other hardware or numerical libraries can change gradients, AMP skips, selected epochs, fitted temperatures, and final metrics.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-reported.txt
python -m pip install --no-deps -e .
```

`requirements-reported.txt` pins the recorded Torch/Transformers versions. Historical records do not provide a complete transitive dependency lock; other entries retain compatible lower bounds. `requirements.txt`/`pyproject.toml` provide the broader supported installation. Check the NVIDIA driver supports the chosen CUDA wheel.

Allow approximately **60 GB of free disk** for sources, Hub cache, checkpoints, epoch exports, and predictions; this is a planning estimate. The FEVEROUS archive alone is 9.9 GB. Preparation streams its members and caches only needed evidence elements, avoiding a full wiki extraction. A host with around 32 GB RAM is a practical starting point; peak host-memory requirements were not benchmarked. Downloads and preparation run on CPU. Original training times were approximately 18/37/37/13/13 minutes for v1/A/B/C/v3, excluding preparation, teacher caching, and evaluation.

Check the installation without downloading models or corpora:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

These tests use tiny local checkpoints and fixtures. They exercise conversion preservation, routing gradients, serialization limits, the historical hard-label trainers, teacher provenance, distillation, calibration, source-group separation, and run-plan safeguards. They do not run the full GPU experiment.

## Plan the complete run

Planning is read-only and works without Torch installed:

```bash
python scripts/reproduce.py plan --work-dir work/reproduction
```

It prints the concrete subprocess commands and the source tree used by each command. `historical/v1` supplies the original v1 runtime; `historical/v2/scripts/finetune.py` supplies the original v2 hard-label trainer; `src/` and the main trainer supply the archived v3 runtime. This preserves the original global-RNG dataloader shuffling, which differs from later development code.

## Reproduce step by step

Use a **new** work directory. Keep the same `--work-dir` for every stage. To restrict a larger machine to the reported GPU count, set `CUDA_VISIBLE_DEVICES` to the two selected GPU indices before running training.

```bash
python scripts/reproduce.py run --stage download --work-dir work/reproduction
python scripts/reproduce.py run --stage convert --work-dir work/reproduction
python scripts/reproduce.py run --stage prepare-v1 --work-dir work/reproduction
python scripts/reproduce.py run --stage train-v1 --work-dir work/reproduction
python scripts/reproduce.py run --stage prepare-v2 --work-dir work/reproduction
python scripts/reproduce.py run --stage train-v2 --work-dir work/reproduction
python scripts/reproduce.py run --stage committee --work-dir work/reproduction
python scripts/reproduce.py run --stage prepare-v3 --work-dir work/reproduction
python scripts/reproduce.py run --stage train-v3 --work-dir work/reproduction
python scripts/reproduce.py run --stage verify --work-dir work/reproduction
```

Or execute those stages consecutively on a fresh workspace:

```bash
python scripts/reproduce.py run --stage all --work-dir work/reproduction
```

| Stage | Operations and safeguards |
| --- | --- |
| `download` | Download pinned dense Julia configurations, weights, tokenizer, and the exact recorded Re-DocRED, DocRED relation-map, MAVEN-ERE, and FEVEROUS bytes. Validate sizes/hashes. No remote Python execution. |
| `convert` | Run the original v1 converter with seed 42; preserve Julia decision components; clone experts without noise. |
| `prepare-v1` | Build 6,000 train / 720 validation / 360 calibration / 720 historical-test questions, then verify exact JSONL fingerprints and disjoint source/evidence groups. |
| `train-v1` | Evaluate untuned MoE; train two epochs; select by validation NLL, calibrate, and evaluate historical test. Save epoch weights and training report. |
| `prepare-v2` | Expand MAVEN to 4,000 questions (80% natural + 20% balanced residual); retain 2,000 Re-DocRED and 2,000 FEVEROUS replay questions; revise event-role serialization; reserve 360 confirmation questions. |
| `train-v2` | Run untuned dense and v1 diagnostics; train A for four epochs on original data and B for three on revised data, both from v1; save development/historical/confirmation predictions. |
| `committee` | Fit task-conditioned probability mixtures on development predictions only, over 66 simplex points/task; write the newly fitted policy and retrospective evaluations. |
| `prepare-v3` | Copy v2 train/validation/calibration/test/confirmation byte-for-byte; reserve 600 unused development questions; freeze the paper's teacher policy and hash the actual reproduction teacher checkpoints. |
| `train-v3` | Cache teachers on training rows only; finish C and student training/calibration before opening fresh evaluations; evaluate all arms and create the matched comparison. |
| `verify` | Recheck all partition hashes, source-group separation, unchanged v2→v3 files, and complete optimization windows. |

Stages refuse existing output directories so completed or interrupted experiments are not silently overwritten. Logs appear under `work/reproduction/logs/`. After interruption, inspect the logs and partial outputs; use a fresh workspace or explicitly relocate the affected stage's partial output before retrying. There is **no exact optimizer-resume facility**: epoch exports contain weights, not optimizer, scheduler, scaler, and RNG state. To resume the sequence after a completed stage, invoke the next individual stage; rerunning `all` on an existing workspace is intentionally refused.

## Data provenance and split roles

Pinned URLs, release revisions, byte counts, and SHA-256 checksums are in [`reproducibility/sources.json`](reproducibility/sources.json). The downloader resolves recorded commits instead of querying the latest dataset revision. If a public URL becomes unavailable, recover the same bytes through the original provider and verify their recorded hashes; do not silently substitute newer releases.

| Profile | Train | Validation | Calibration | Historical test | Confirmation | Fresh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 | 6,000 | 720 | 360 | 720 | — | — |
| v2 | 8,000 | 720 | 360 | 720 | 360 | — |
| v3 / C | 8,000 | 720 | 360 | 720 | 360 | 600 |

v1 training contains 2,000 questions from each source. v2/v3 retain 2,000 Re-DocRED and 2,000 FEVEROUS questions and use 4,000 MAVEN questions. Fresh contains 200 per source and 438 `(source_dataset, source_id)` groups. Shared article/evidence/claim groups are excluded across assigned roles. Fresh also excludes groups in **every** prior v1/v2 role. Its labels were unused in the original selection; it is now a published historical evaluation, so a new study should not treat it as an unseen holdout after inspecting it.

Re-DocRED uses positive relation pairs with four supplied candidates; MAVEN uses positive annotated temporal/causal pairs with supplied arguments and candidates; FEVEROUS receives complete supplied gold evidence alternatives. Full documents are retained or rejected for length. These are candidate-conditioned proxy tasks, **not official extraction/retrieval benchmark protocols**.

Validation selects checkpoints; calibration fits temperature. The committee uses development NLL only. Historical test and confirmation are retrospective diagnostics. Fresh labels must not select a new policy, checkpoint, data recipe, or temperature.

[`reproducibility/expected_partitions.json`](reproducibility/expected_partitions.json) records the exact prepared-file fingerprints. Manifest paths and provenance fields can differ in a new workspace; the verifier compares actual JSONL bytes and counts, not just manifest claims.

```bash
python scripts/verify_reproduction.py --work-dir work/reproduction --all
# Validate an existing raw cache, without downloads:
python scripts/download_gotham_sources.py --output-dir work/reproduction/raw --verify-only
```

## Frozen committee versus a newly fitted committee

For the **paper-recipe replay**, `prepare-v3` deliberately uses [`reproducibility/archived_teacher_policy.json`](reproducibility/archived_teacher_policy.json). Its weights, in `[v1, control A, v2]` order, are:

| Task | v1 | A | v2 |
| --- | ---: | ---: | ---: |
| Entity relation | 0.6 | 0.4 | 0.0 |
| Evidence verdict | 0.3 | 0.7 | 0.0 |
| Temporal relation | 0.0 | 0.0 | 1.0 |
| Causal relation | 0.0 | 0.0 | 1.0 |

The archived policy also contains the **original** development scores and prediction hashes. Those are historical provenance, not scores or predictions generated by your new checkpoints. v3's new manifest and teacher receipt bind the fixed policy bytes to the actual newly trained teachers and exact training file.

The `committee` stage separately saves a new fit in `committee-refit/policy.json`. GPU variation may select different weights. Compare and report this outcome; the replay does not replace the paper's policy with the new fit. An end-to-end replication that uses a different newly fitted policy is a separate experiment. The preparation script rejects weights outside the reviewed paper recipe; changing that recipe requires a new protocol and should not be described as an exact paper replay.

Teacher caching runs the three calibrated teachers sequentially and checks candidate identities and row metadata. It caches training probabilities only. The student objective uses `T²`-scaled KL at temperature 2 while retaining gold labels; the hard-label control has no teacher cache.

## Outputs and results

```text
work/reproduction/
  julia-dense/                     # pinned base + download_provenance.json
  praxis-base/                     # original sparse conversion
  raw/                            # sources, receipts, evidence cache
  data/{v1,v2,v3}/                 # JSONL roles + data_manifest.json
  runs/v1/defense-praxis/          # calibrated selected v1
  runs/v2/defense-praxis/          # calibrated selected B/v2
    control_a/                    # calibrated selected A
    diagnostics/                  # aligned saved predictions
  committee-refit/                # newly fitted development policy
  teachers/                       # local checkpoint symlinks
  v3-bundle/                      # local staging links for archived runner
  runs/v3/defense-praxis/          # calibrated selected student
    control_c/                    # matched hard-label continuation
    teacher_cache/                # cached probabilities + hashed receipt
    diagnostics/                  # each arm/partition's predictions
    experiment_comparison.json    # metrics + paired bootstrap + checks
  logs/                           # each command's stdout/stderr
```

Rebuild the matched comparison and its 500-draw, seed-42 source-group paired bootstrap from saved outputs:

```bash
PYTHONPATH=src python scripts/rebuild_results.py --work-dir work/reproduction
```

The comparison checks completed matched training budgets, teacher/data/policy hashes, evaluation counts, finite NLL, and candidate alignment. Metrics include accuracy, macro F1, per-class precision/recall, NLL, Brier, calibration diagnostics, and source-group uncertainty. Connected evidence-component uncertainty is provided by the additional analysis command in [the provenance guide](docs/PROVENANCE.md#uncertainty-analysis).

These are **reported original-run results**, not guaranteed outcomes of retraining:

| Arm | Fresh accuracy (600 questions) | Successful / skipped optimizer windows | Selected epoch |
| --- | ---: | ---: | ---: |
| Dense Julia-1, untuned | 29.50% | — | — |
| v1 | 56.17% | 745 / 5 | 2 of 2 |
| Control A | 59.83% | 1,495 / 5 | 2 of 4 |
| v2 / B | 71.83% | 1,496 / 4 | 1 of 3 |
| Control C | 73.17% | 497 / 3 | 1 of 1 |
| Frozen committee | 76.33% | — | — |
| v3 student | 76.50% | 498 / 2 | 1 of 1 |

v3 scores 459/600 versus C's 439/600: +3.33 percentage points, paired source-group 95% interval [1.24, 5.66]. The v3–committee difference is only +0.17 points, interval [−1.85, 2.19]; this does not establish superiority over the committee. Complete archived metrics, temperatures, configuration, and hashes are in [`reproducibility/reported_results.json`](reproducibility/reported_results.json).

Exact data bytes are checked. Exact retrained weight hashes or identical numerical results are **not** assumed. There is one reported seed, no matched dense fine-tuning arm, and no architecture-speedup claim. The distillation comparison evaluates an objective package: half-weight gold CE plus teacher KL, versus full-weight gold CE for C.

## Run a checkpoint on new questions

Each question provides state, question, and candidate options. See [`examples/requests.jsonl`](examples/requests.jsonl) for the schema; toy examples are not research data.

```bash
PYTHONPATH=src python scripts/predict.py \
  --model-dir work/reproduction/runs/v3/defense-praxis \
  --input-file examples/requests.jsonl \
  --output-file work/reproduction/predictions.jsonl \
  --device cpu
```

## Source provenance, licenses, and contact

[`docs/PROVENANCE.md`](docs/PROVENANCE.md) explains archived source selection, portability changes, validation, and uncertainty analysis. [`reproducibility/code_origins.json`](reproducibility/code_origins.json) records original source hashes; [`reproducibility/code_inventory.json`](reproducibility/code_inventory.json) records shipped source hashes. The original v1/v2 hard-label code is preserved rather than reconstructed from later trainers.

Repository code is provided under the included [Apache 2.0 license](LICENSE). Julia-1 runtime attribution and dataset terms are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), with upstream notices in [`licenses/`](licenses/). Re-DocRED, MAVEN-ERE, FEVEROUS, and underlying Wikipedia material retain their applicable terms; the repository's code license does not relicense those sources. Downloading and transforming data is separate from redistributing it.

Author: **Sean Lester Miranda** · [contact@seanmiranda.com](mailto:contact@seanmiranda.com)
