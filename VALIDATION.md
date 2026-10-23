# Validation record

Repository assembly: October 24, 2026 (Asia/Manila).

## Checks performed

- 27 CPU tests passed across conversion, model routing and gradients, marker serialization, dataset transformations, committee fitting and leakage checks, teacher-cache provenance, actual tiny distillation/calibration, both archived hard-label trainers, and reproduction planning/data-tamper safeguards.
- The two additional uncertainty tests passed: transitive shared-evidence components and rejection of mismatched paired metadata, including zero intervals for identical predictions. Total: **29 passing tests**.
- Python compilation succeeded for all runtime, script, historical, and test modules.
- The full stage plan generated successfully without downloading or training. The v1 converter and the main converter are byte-identical; v1 and v2 historical hard-label trainers are byte-identical.
- The verifier accepted every archived v1/v2/v3 partition, exact v2→v3 retained roles, exclusion of fresh groups from prior roles, and all five archived complete training budgets.
- Regenerating v1, v2, and v3 from the recorded cached raw sources and tokenizer reproduced **all 15 prepared JSONL fingerprints exactly**, including the 600-question fresh set. The combined verifier accepted all regenerated profiles and confirmed byte-identical v2→v3 retained roles and fresh-group exclusion. The v3 preparation check used archived v2 inputs while v2 regenerated independently; their fingerprints were subsequently verified identical.
- The new uncertainty-analysis entry point, applied to the archived fresh data and saved predictions, exactly reproduced all four connected-component point estimates and intervals, and recovered 431 components, a largest component of five questions, and 438 source-ID groups.

## Environment and limits

These assembly checks used the available host environment: Python 3.14, PyTorch 2.14.0+cu130, Transformers 5.17.0; training fixtures ran on CPU. This differs from the reported original two-T4 CUDA 12.8 / Python 3.12.13 / Torch 2.10.0 environment. Follow the README for the reported-environment setup.

No full-size GPU retraining was launched while assembling this repository. No corpus/model download, cloud training submission, model publication, Git push, or arXiv submission was performed. Dataset preparation used existing local cached source bytes; tiny training tests use temporary local checkpoints. Passing these checks verifies source/data mechanics, not identical retrained weights or new benchmark results.
