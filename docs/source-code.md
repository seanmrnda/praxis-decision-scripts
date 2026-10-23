# Runtime source map

- `julia.py`: complete Julia decision model and strict checkpoint loading.
- `model.py`: cloned sparse FFNs, top-k dispatch, auxiliary loss, and MoE checkpoint format.
- `data.py`: typed targets, marker serialization, length validation, and batch collation.
- `distillation.py`: teacher-cache provenance, candidate alignment, and masked KL.

For archived versions, portability changes, and reproduction boundaries, see [PROVENANCE.md](PROVENANCE.md) and the root [README](../README.md).
