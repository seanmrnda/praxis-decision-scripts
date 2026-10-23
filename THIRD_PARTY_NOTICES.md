# Julia-1 runtime attribution

`src/decision_moe/julia.py` is adapted from Supersonic Labs' Julia-1
`julia/model.py` at revision `a85b127321d580d65176c89ced8273f305745d85`:
https://huggingface.co/SupersonicLabs/Julia-1/blob/a85b127321d580d65176c89ced8273f305745d85/julia/model.py

Julia-1's model card licenses its model artifacts and runtime under Apache 2.0.
The upstream model definition and state-dictionary names are retained. Local
changes add attribution, enforce the dense checkpoint format and offline config
loading, remove the legacy RL-agent checkpoint fallback, and document the
retained APIs with docstrings. Marker serialization
in `src/decision_moe/data.py` follows upstream `julia/data.py`, with strict lossless
encoding and this repository's typed JSONL labels.

The Apache 2.0 license is included in `licenses/Julia-1-Apache-2.0.txt`.

## Training sources

Re-DocRED annotations are provided by Tan et al. at
https://github.com/tonytan48/Re-DocRED; their MIT notice is reproduced in
`licenses/Re-DocRED-MIT.txt`. Underlying Wikipedia text retains its source terms.

MAVEN-ERE is provided by Wang et al. at https://github.com/THU-KEG/MAVEN-ERE.
The author's GPL-3.0 repository notice is reproduced in
`licenses/MAVEN-ERE-GPL-3.0.txt`; underlying Wikipedia text retains its source terms.

FEVEROUS is provided by Aly et al. at https://fever.ai/dataset/feverous.html.
Its Wikipedia evidence follows the publisher's applicable source licenses at
https://fever.ai/download/feverous/license.html, including CC BY-SA 3.0 where
specified. Adaptations select complete evidence alternatives, preserve titles
and element IDs, and serialize text/table evidence into candidate questions.
Wikipedia article attribution can be resolved from each retained title at
`https://en.wikipedia.org/wiki/<title>`. These mixed source terms are not
replaced by Julia-1's Apache license. See `README.md` and the
generated `data_manifest.json` for source revisions, hashes, adaptations and scope.
