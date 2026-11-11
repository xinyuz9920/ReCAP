# ReCAP

**Residual Coverage with Anchored Protection for Visual Instruction Tuning**

This repository contains the smallest executable **selection experiment** extracted from the project's `ResidualCover` implementation. It runs on CPU using supplied frozen image and visible-QA text features. The four selection-core modules are preserved byte for byte; their source hashes and the earlier coverage-backend provenance are in [`SOURCE_PROVENANCE.json`](SOURCE_PROVENANCE.json).

## Quick start: an independent CPU experiment

Use Python 3.10 or newer with NumPy, SciPy and PyTorch. Install a CPU PyTorch build appropriate for your platform if desired; CUDA is not required. The entry point always uses CPU and never downloads data or model weights.

```bash
python -m pip install -e .
python -m recap demo --output outputs/demo
python -m recap validate --run outputs/demo/run
python -m unittest discover -s tests -v
```

The demo creates 32 artificial records with 63 artificial QA demands, selects 8 records from a 16-record nested candidate pool, and checks the independently recomputed protection floors. It also demonstrates the deterministic record export rule. These artificial vectors test the algorithm and are unrelated to benchmark accuracy.

## Run selection on another dataset

Prepare one NPZ file using the following schema. Array rows, not external IDs, define the record order.

| Array | Shape | Meaning |
|---|---|---|
| `image` | `[N, D_image]` | Finite nonzero frozen image feature for each original record |
| `text` | `[M, D_text]` | Finite nonzero text feature for each training-visible QA demand |
| `owner` | `[M]`, integer | Parent row of each demand, in `[0, N)` |
| `direction` | `[M]`, integer | Nonnegative semantic-direction label; only equal labels can match |

Every record must own at least one demand. Features are normalized by the core; image and text feature dimensions may differ. Do not include held-out evaluation examples. Visibility, truncation and direction labels must be prepared using your training protocol. With a single task direction, use all-zero labels. Direction labels are compatibility constraints, not validated capability classes.

```python
import numpy as np
np.savez_compressed("features.npz", image=image, text=text,
                    owner=owner.astype(np.int64), direction=direction.astype(np.int64))
```

```bash
python -m recap select --features features.npz --output outputs/my_run --budget 9369 --depth 7
python -m recap validate --run outputs/my_run
```

Default settings preserve the recorded experiment: 128 balanced PCA leaves (`depth=7`), 24 power iterations, seed 42, Hamilton integer quotas, one greedy sequence with reference prefix `q` and candidate prefix `2q`, record-max **before** top-32 distinct records, and global protection `tau=1`. The bounded 1-for-1 / 2-for-2 backend settings are unchanged. Small datasets require a lower tree depth. The program stops when any leaf has `q=0` or cannot fit `2q`; it also stops if a direction has demands but no candidate representation.

Outputs include parent-row indices `selected.npy`, `reference.npy`, `candidate.npy`; strata and quotas; the rebuilt global CSR graph; greedy and exchange traces; input and implementation hashes; and independent floor validation. The output directory must be empty. Given fixed inputs and numerical environment, ties and membership are deterministic. The validator checks exported artifacts; it does not establish that the supplied features represent the intended source records.

For an unchanged source JSON list in the same row order:

```bash
python -m recap export --pool pool.json --run outputs/my_run --output outputs/my_subset
```

The exporter preserves complete source objects, shuffles sorted selected row indices with Python `Random(42)`, and applies prefix padding to a multiple of 128. It writes both the unique `subset.json` and padded `train.json`. This ordering contract must be maintained by the caller; unrelated JSON data cannot be matched to vectors automatically.

## What the selector optimizes

Each original record has total demand weight one, distributed equally across its visible QA demands. A demand matches a candidate record through same-direction positive image cosine times positive text cosine, maximized across that candidate's QA demands. A balanced PCA tree supports local greedy candidate construction. The final graph is rebuilt across all parent demands and all candidates; reference floors are recomputed on this graph. The backend accepts only strict total-coverage improvements that preserve every group's reference coverage.

See [`docs/METHOD.md`](docs/METHOD.md). Semantic coverage is a proxy; it is not target-model confidence, training loss or benchmark accuracy. The fixed-graph protection constraint does not guarantee that every benchmark improves. The solver stops after bounded local search and makes no global-optimality claim.

## Recorded benchmark values

The following are **observed values from the existing experiment**, not outputs of the artificial demo. Full, OnceSelect and ReCAP were evaluated with the same local protocol. The metric JSON includes scorer limitations and the record hash: [`results/observed_metrics.json`](results/observed_metrics.json). No generated paper-table values are included.

| Method | MMB-EN | MMB-CN | MME-P | MME-C | AI2D | POPE-A | POPE-P | SQA-IMG | OCRBench | Mean relative to Full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 59.00 | 52.11 | 1317.53 | 282.14 | 50.91 | 85.54 | 92.30 | 63.66 | 28.60 | 100.00% |
| OnceSelect | 54.05 | 47.54 | 1162.95 | 282.50 | 48.41 | 83.18 | 87.71 | 61.97 | 28.90 | 95.22% |
| ReCAP | 51.65 | 47.42 | 1323.35 | 298.21 | 48.58 | 84.24 | 88.38 | 63.46 | 28.60 | 97.11% |

Mean relative effect is `100 * mean(metric_method / metric_full)` over these nine metrics. This is one recorded 15% subset setting and training seed, with no confidence intervals. MMB is labeled-dev circular accuracy; POPE-A/P are split-macro accuracy/precision; SQA is image-bearing accuracy. They are not interchangeable with other benchmark variants or hidden-test leaderboards.

The recorded selection used 62,461 parent records, 335,369 visible demands, 18,738 candidates and 9,369 final unique records. The final set differs from its own frontend reference by five records. Its cached-feature core time was approximately 35.72 seconds on the original GPU-assisted graph implementation; this excludes encoding, import, I/O verification, training and evaluation. **This CPU-only portable release does not reproduce that speed claim.**

## Training and evaluation scope

This release reproduces selection and record export from precomputed features. It does not bundle the project's LLaVA training engine, custom unified evaluation runner, original datasets, CLIP feature encoder/cache, checkpoints, benchmark predictions or model weights. An end-to-end cold-start accuracy reproduction requires those external inputs and an independently verified environment. Install the official [LLaVA](https://github.com/haotian-liu/LLaVA) stack and obtain the datasets and base checkpoints under their own terms. CLIP representations refer to [OpenAI CLIP](https://github.com/openai/CLIP); neither dependency's source or weights are redistributed here.

[`examples/recorded_protocol.json`](examples/recorded_protocol.json) records the observed training/evaluation parameters. It is a protocol record, not a drop-in launcher or a complete benchmark runner. Full and subset use the same recipe, while optimizer updates follow the padded data size (488 versus 74 recorded steps). The local benchmark scorer distinctions are documented rather than claimed to be official hidden-test results.

## Attribution and licensing

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The graph, greedy and protected-exchange backend originate in this project's earlier OnceCover implementation; the residual frontend replaces the earlier imported-selection frontend. This release does not claim that all components are newly introduced. No license has been assigned in this extraction. The author must confirm redistribution rights and choose a repository license; publication on GitHub alone does not grant an open-source license.

[`PUBLIC_FILES.json`](PUBLIC_FILES.json) enumerates the reviewed release files. The public scan report records path, credential-pattern and large-file checks; no private data, cached vectors or model artifacts belong in this repository.
