# Source provenance and replication boundaries

## Archived implementations

The reproduction code was recovered from the **code directories staged with the original v1, v2, and v3 Kaggle runs**, rather than the parent project's later development checkout. Original runs completed September 28, 2026. Historical `Defense Praxis` artifact identifiers are retained; the paper/model-family name is Praxis Decision.

- `historical/v1/src/decision_moe/` contains the original dense/sparse runtime. `historical/v1/scripts/` contains its converter, hard-label trainer, and evaluator.
- `historical/v2/scripts/finetune.py` is the original v2 hard-label trainer. Its source is identical to the v1 trainer; it uses the main runtime through `PYTHONPATH`.
- `src/decision_moe/` is the archived v3 runtime. Its model source is identical to the v2 runtime. The v2/v3 model adds routing observation support relative to v1.
- Main `scripts/finetune.py`, `evaluate_gotham.py`, `experiment_committee.py`, `cache_teachers.py`, `run_distillation.py`, `distillation_results.py`, `convert_julia.py`, and `predict.py` are copied from the v3 code snapshot.

[`code_origins.json`](../reproducibility/code_origins.json) records original SHA-256 hashes and archive-relative origins. The original snapshots were filesystem artifacts; this receipt does not invent a source Git commit. [`code_inventory.json`](../reproducibility/code_inventory.json) records hashes of the shipped files. The v1/v2 trainers deliberately retain `DataLoader(..., shuffle=True)` using the global RNG, rather than the explicit loader generator added during later development.

## Portability changes

Only the following archived components were adapted:

1. `prepare_improvement.py` accepts explicit raw, prior-data, tokenizer, and output paths. Existing outputs are rejected. The historical parent-checkpoint commit is explicitly marked historical, rather than attributed to a newly trained reproduction checkpoint. Sampling, serialization, seed, and row-writing logic are retained.
2. `prepare_distillation.py` accepts explicit raw, v1/v2 data, tokenizer, teacher-root, policy, and output paths. Sampling and exclusion logic are retained. It hashes the actual teacher models used for the new run. Its recipe retains the original historical model identifiers.
3. `prepare_gotham.py` rejects an existing output directory at the CLI boundary. Core generation, filtering, and partitioning are retained.
4. `download_gotham_sources.py` replaces dynamic latest-revision discovery with recorded source revisions and checksums. It verifies cached/downloaded bytes before writing provenance. Original receipt metadata is restored only after that check, so downstream preparation sees the recorded source descriptors. No expiring download credentials are bundled.
5. `pyproject.toml` and `requirements.txt` add the `requests` dependency used by the downloader. The third-party notice points to the new guide.

The orchestrator, base downloader, verifier, result-rebuild entry point, uncertainty analysis, documentation, and orchestration tests are new. Cloud credential helpers, owner-specific publishing code, raw datasets, weights, run-account state, and later experiments are excluded. Running the new orchestrator performs local downloads/training/evaluation and does not publish models or submit cloud jobs.

## Experiment replay and new selection

The original committee policy is a historical artifact. Its selection scores and development prediction hashes refer to the original teachers. Replaying that policy with newly trained teachers reproduces the frozen v3 treatment, while introducing the normal variation of retraining those teachers.

The `committee` stage fits a separate policy on the newly generated development predictions. A differing optimum is a replication finding. It does not automatically replace the frozen policy, and no fresh labels influence fitting. A new-policy distillation study needs a separately recorded protocol and should report its distinction from the paper replay.

The per-file expected hashes apply to prepared JSONL files. New manifests contain new teacher hashes and local path/provenance information, so their bytes need not equal the archived manifests. Checkpoints are selected by validation NLL; observing a different selected epoch is possible and must not be “corrected” by choosing the historical epoch from test performance.

## Uncertainty analysis

The archived comparison script reproduces the primary paired accuracy bootstrap: seed 42, 500 draws, resampling `(source_dataset, source_id)` groups with replacement and retaining all questions in each drawn group. Sorted endpoints use indices 12 and 487.

The additional command reconstructs the paper's post-hoc evidence-group sensitivity analysis:

```bash
python scripts/analyze_uncertainty.py \
  --data-file work/reproduction/data/v3/fresh.jsonl \
  --predictions-dir work/reproduction/runs/v3/defense-praxis/diagnostics \
  --output-file work/reproduction/evidence_group_bootstrap.json
```

It forms connected components by shared `split_groups`, including transitive shared article/evidence/claim connections. It checks prediction/data candidate identities, labels, IDs, and metadata before comparisons. Default component resampling also uses seed 42 and 500 draws, with sorted endpoints 12/487. Components follow first occurrence in the saved prediction order. On the archived data and predictions, it exactly recovers all four reported connected-component point estimates and confidence intervals, with **431 components and a largest component of five questions**. The same invocation also returns the primary source-ID intervals.

Optional `--draws` changes only the number of component draws; the original source-ID routine remains at 500. Alternative Monte Carlo settings are additional sensitivity analyses. These intervals do not quantify variability across training seeds and do not establish operational generalization.

## Validation performed while assembling this repository

See [`VALIDATION.md`](../VALIDATION.md) for the checks actually run and the environment used. Small CPU training tests validate mechanics; they do not constitute a rerun of the full experiment. Full retraining is an explicit user-run GPU stage. No new accuracy or exact weight-reproduction claim follows from passing unit tests.

## Upstream attribution

The Julia runtime derives from the pinned [Julia-1 source](https://huggingface.co/SupersonicLabs/Julia-1/blob/a85b127321d580d65176c89ced8273f305745d85/julia/model.py). Source repositories include [Re-DocRED](https://github.com/tonytan48/Re-DocRED), [MAVEN-ERE](https://github.com/THU-KEG/MAVEN-ERE), and [FEVEROUS](https://fever.ai/dataset/feverous.html). Dataset transformations are detailed in the README and generated manifests. Applicable notices are retained in `licenses/` and `THIRD_PARTY_NOTICES.md`.
