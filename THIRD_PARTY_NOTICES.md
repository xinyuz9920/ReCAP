# Source attribution and release rights

The public `recap/frontend.py`, `recap/graph.py`, `recap/greedy.py`, and `recap/backend.py` files are byte-preserving extractions from the author's existing `ResidualCover` project snapshot. SHA256 and exact project-relative source names are recorded in `SOURCE_PROVENANCE.json`.

The earlier backend provenance identifies mathematical functions inherited from the same project's `OnceCover/oncecover/graph.py`, `greedy.py`, and `coverage_floor.py`. This is provenance for local project code, not a claim that the upstream OnceSelect paper authors supplied those files. The partition and nested residual-candidate frontend is a separate part of the recorded implementation. `tests/test_core.py` adapts the original core tests only by package import changes and omission of deployment-specific pipeline tests.

External runtime libraries are NumPy, SciPy and PyTorch, installed separately under their respective licenses. Optional training and feature extraction refer to the official LLaVA and OpenAI CLIP repositories; no source code, checkpoints or data from those projects are included. Any dataset, benchmark, base model or encoder obtained separately remains subject to its original terms.

No standalone source license was identified in the extracted project directory. This extraction therefore does not assign an MIT, Apache, BSD or other license, and no `LICENSE` file is created. The repository author must confirm authorship and any third-party obligations before choosing a license. Public availability is not equivalent to an open-source license grant.
