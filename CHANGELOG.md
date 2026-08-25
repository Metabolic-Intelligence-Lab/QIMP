# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The autonomous-ratiometric line of work: a reversible-arithmetic layer on
NEQR intensity registers, three end-to-end operator pipelines built on it,
a QAE oracle composed from them, and the IBM Heron r2 runtime used to put
the smallest of them on real hardware. This is the library behind the
`autonomous_qimp_paper` manuscript.

### Added — processing / arithmetic
- **Inverses and controlled variants** for every adder: `qc_add_1_inv`,
  `qc_add_1_ctrl`, `qc_add_1_ctrl_inv`, `q_add_inv`, `q_add_ctrl`,
  `q_add_ctrl_inv`, `q_sub_inv`, `q_sub_ctrl`, `q_sub_ctrl_inv`. Exact
  inverses are the prerequisite for using any of this as a QAE state-prep.
- **`q_mul_const` / `q_mul_const_inv`** — Booth-free shift-add multiply by
  a classical constant.
- **Dividers.** `q_div_restoring` (square, q ÷ q) and `q_div_general`
  (non-square, n ÷ m) shift-subtract dividers with divide-by-zero flags,
  plus `q_div_restoring_inv`. `q_div` aliases the current default.
- **`q_div_nonrestoring` / `q_div_nonrestoring_inv`** and the
  `q_add_sub_ctrl` / `q_add_sub_ctrl_inv` they are built on — an
  algebraically-equivalent divider costing 48-61 % fewer two-qubit gates
  after transpile. This is the compression that carries the Class-B
  pipeline across the Heron r2 gate-noise floor.

### Added — processing / ratiometric_circuit (new module)
- **Class B**, integer ratio: `class_b_ratio` / `class_b_ratio_inv` /
  `decode_class_b_ratio`, on a `dual_neqr_load` / `dual_neqr_load_inv`
  pair of co-registered NEQR images. Computes `R = I_a // I_b` per pixel
  in-circuit with a divide-by-zero flag, and round-trips to |0...0>.
- **Class A**, Laurdan generalized polarization: `class_a_gp_prefix` /
  `decode_class_a_prefix` for the numerator/denominator stage, and
  `class_a_gp_full` / `decode_class_a_full` for the whole signed
  fractional pipeline (subtract, conditional negate, bit-shift,
  non-square divide) recovering GP in [-1, +1].
- **Class C**, roGFP calibrated redox: `class_c_rogfp_full` /
  `decode_class_c_rogfp`, a fractional ratio plus an in-circuit
  `affine_subtract_constant`, recovering R_C in [0, 1]. The fractional
  ratio is what lifts the integer-quotient degeneracy that makes Class B
  uninformative at roGFP's sub-unit F405/F488 operating point.
- **QAE oracle**: `mark_good_oracle` / `mark_good_oracle_inv`, a
  threshold predicate over the quotient register, self-inverse and
  divzero-corrected, so Class B composes into an amplitude-estimation
  state-prep.

### Added — runtime
- **`qimp.runtime.ibm`** — `get_service`, `list_backends`, `pick_backend`
  (named or least-busy), `hw_run` on SamplerV2 with TREX / dynamical
  decoupling / ZNE mitigation modes, `aer_noisy_run` via
  `AerSimulator.from_backend`, and `persist_run` / `is_run_complete` for
  QPY + JSON run archival.
- **`qimp.runtime.circuits`** — `CircuitRecipe` and `build_recipes`, one
  row of the sweep matrix (circuit + decoder + classical reference) per
  encoder across FRQI, FRQI-multi, NEQR, QPIE, MCRQI, NCQI and GP.

### Fixed
- `decode_class_a_full` read only the low `q_frac` bits of the quotient.
  The magnitude reaches `2**q_frac` at the saturating GP = +/-1 endpoints,
  so exactly those pixels decoded as 0.0. It now reads the full register.
- `gp_ratio`: closed-form analytical parameters and corrected H placement.
- `decode_class_b_ratio` redefined its `bit_at` helper inside the counts
  loop, closing over the loop variable (ruff B023). Hoisted out.
- `q_add_sub_ctrl`, `q_add_sub_ctrl_inv`, `q_div_nonrestoring` and
  `q_div_nonrestoring_inv` were public but missing from `__all__`, so
  they were invisible to `import *` and to the generated API docs — while
  being named directly in the manuscript's *Code availability*.

### Changed — tooling
- `ruff` and `mypy` are now pinned in the dev extra and kept in sync with
  the pre-commit `rev:`s. A floating `ruff>=0.4` against a pinned
  `v0.4.10` hook meant CI failed on rules the pre-commit hook never ran.
- mypy's analysis target moved to 3.12 (supported runtime stays >= 3.10):
  numpy's PEP 695 stubs cannot be parsed below it, which was aborting the
  type-check before it reached any project code.
- The IBM hardware smoke test now skips, rather than fails, without the
  `[ibm]` extra or saved credentials.

## [0.2.0] — 2026-05-18

Implements every previously-stubbed library module: RGB encodings, NEQR
arithmetic, FRQI/NEQR compression, the variational image classifier, and
folder-level dataset iteration. All thesis-spec features are now live.

### Added — encoding
- **`qimp.encoding.mcrqi`** — FRQI extension to RGB. ``2n + 3`` qubits,
  one color qubit per channel, separate controlled-RY per (pixel, channel).
  `McrqiEncoder` mirrors `FrqiEncoder`.
- **`qimp.encoding.ncqi`** — NEQR extension to RGB. ``2n + 3q`` qubits with
  per-channel intensity registers. Exact retrieval.
- **`qimp.encoding.compression`** — Quine–McCluskey prime-implicant
  generation plus greedy disjoint-cover selection. `FrqiCompressor` and
  `NeqrCompressor` emit reduced multi-controlled gate sequences. Disjoint
  cover required for correctness (XOR / angle-sum semantics).

### Added — processing
- **`qimp.processing.arithmetic`** — `qc_add_1`, `q_add` (ripple-carry on
  arbitrary widths), `q_sub` (two's-complement), `neqr_comparator`
  (sets a gt-qubit iff a > b).

### Added — qml
- **`qimp.qml.classifier`** — `FrqiClassifier`, a binary variational
  classifier with a `RealAmplitudes` ansatz on top of FRQI features,
  trained with `scipy.optimize.minimize` (COBYLA) against ±1 labels.

### Added — io
- **`qimp.io.datasets`** — `ImageDataset` (folder iterator with
  square / pow2 / blank filtering) and `batch_process_images` helper.

### App
- `apps/qimp_explorer/pages/5_System_Info.py` updated: the previous
  "stubs not yet implemented" list has been replaced with the current
  implementation status (everything green except `neqr_sort` and the
  ESPRESSO compression heuristic, deferred to v0.3).

### Tests
- 49 new tests across the new modules.
- ruff / mypy strict / mkdocs `--strict` all clean.

## [0.1.1] — 2026-05-14

Patch release addressing all findings from the v0.1.0 code review.

### Fixed (bugs)
- **`encoding/frqi.py`**: float images with intensity > 1 were silently saturated
  because the default normalization was 1.0 for any float dtype. A new
  `_infer_normalization` helper now uses `image.max()` for floats and the dtype
  max for integers.
- **`runtime/memory_pool.py`**: `get_image_buffer` no longer returns the same
  buffer on consecutive calls. The cursor advances on every call now,
  matching the documented round-robin behaviour and `get_angle_buffer`.
- **`processing/geometric.py`**: `pos_shift` used to use `axis` before
  validating it; ill-typed `axis` values silently took the X branch. The
  validation is now first. A new `_validate_position_register` helper is
  called by every public geometric operation, surfacing
  "`pos_offset + 2n` doesn't fit in circuit" errors instead of producing
  silently-wrong gates.
- **`processing/filters.py`**: stopped reaching into the private
  `qpie._validate_and_normalize`. The helper is now public as
  `qimp.encoding.qpie.normalize_amplitudes`.
- **`encoding/qpie.py`**: an all-zero image is still encoded as a uniform
  superposition (so the circuit construction doesn't blow up) but now emits a
  `UserWarning` explaining that decoders with `rms ≠ 0` will return spurious
  intensities.
- **`metrics.psnr`**: emits a `UserWarning` if called on float images whose
  observed max exceeds 1.0 without an explicit `max_intensity`. Previously
  this returned absurdly low dB values silently.

### Changed
- **`processing/filters.py`**: `qhed_filter` docstring rewritten with an
  explicit warning that the cyclic decrement produces a *row-major flattened*
  gradient, not a pure horizontal one. The wrap-around at row boundaries is
  documented as a feature of the algorithm, not noise.
- **`metrics.total_variation`**: docstring corrected — the implementation is
  isotropic (``√(∇x² + ∇y²)``), the previous wording said "anisotropic".
- **`pyproject.toml`**: pytest now deselects `slow` tests by default. CI has a
  separate `test-slow` job that runs everything.
- **Geometric private helpers `_increment`/`_decrement`** consolidated into the
  new public `qimp.runtime.arithmetic_gates` module
  (`cyclic_increment`, `cyclic_decrement`). Both `pos_shift` and `qhed_filter`
  use the shared implementation.

### Added (tests)
- Regression tests for every bug above (float normalization, memory-pool
  cursor, pos_offset validation, QPIE all-zero warn, PSNR float warn).
- Semantic round-trip tests for `restr_flip` and `restr_coord_swap`
  (previously only their input-validation was covered).
- Tests for `cyclic_increment` / `cyclic_decrement` (round-trip identity).
- `transpile_summary` test with a custom basis.

### Added (docs)
- `docs/encoding.md`: explicit qubit-layout convention table.
- `docs/processing.md`: QHED wrap-around warning admonition.

## [0.1.0] — 2026-05-14

First tagged release. Repository remains **private** while we iterate;
publication on PyPI is deferred. See README for the private-install workflow.

### Security
- Redacted an IBM Quantum API token that was hard-coded in eight legacy
  exploratory scripts under `legacy/Old/`. The token is replaced by
  `<REDACTED-IBM-Q-TOKEN>`; the original was committed to the repository in
  earlier history (during the legacy migration in Fase 2) so it has been
  **revoked** on the IBM Quantum side. If you cloned the repo prior to v0.1.0,
  treat that token as compromised.

### Added — Fase 5
- `examples/` directory with four self-contained runnable scripts:
  `01_frqi_fidelity.py`, `02_neqr_round_trip.py`, `03_qhed_edges.py`,
  `04_qpie_round_trip.py`.
- `mkdocs.yml` with Material theme and `mkdocstrings[python]` for auto-generated
  API reference.
- `docs/index.md`, `docs/getting-started.md`, `docs/encoding.md`,
  `docs/processing.md`, `docs/examples.md` plus four API-reference pages.
- `.github/workflows/docs.yml` for building and deploying docs to GitHub Pages
  on every push to `main`.
- `CONTRIBUTING.md` explaining setup, design invariants, and commit conventions.

### Added — Fase 4
- Full implementations of the three thesis encodings:
  - `qimp.encoding.qpie` — amplitude encoding via `qc.initialize`.
  - `qimp.encoding.neqr` — basis-state encoding with exact retrieval.
  - (`qimp.encoding.frqi` was migrated in Fase 3.)
- `qimp.processing.geometric` — `axis_flip`, `coord_swap`, `ort_rotation`,
  `pos_shift`, `restr_flip`, `restr_coord_swap`. Encoding-agnostic via
  `pos_offset`.
- `qimp.processing.chromatic` — color complement / change for FRQI and NEQR;
  half-intensity and classify operations for NEQR.
- `qimp.processing.filters` — Quantum Hadamard Edge Detection (QHED) for QPIE
  with `qhed_filter`, `qhed_decode`, and `qhed_full_edges`.

### Added — Fase 3
- Migration of `legacy/scripts/*.py` into typed, tested modules under
  `qimp.metrics`, `qimp.runtime`, `qimp.io.image`, `qimp.qft`, `qimp.testing`,
  `qimp.encoding.frqi`, and `qimp.processing.gp_ratio`.
- Tests with parametric fixtures over `n` (spatial qubits) and `q` (intensity
  qubits); 159 tests overall.

### Added — Fase 2
- Package scaffolding: `src/qimp/` with sub-packages `encoding`, `processing`,
  `qml`, `io`, `runtime`, plus modules `qft`, `testing`, `metrics`, `config`,
  `cli`.
- `pyproject.toml` (hatchling, PEP 621), ruff, mypy, pytest, pre-commit configs.
- GitHub Actions CI: lint + tests on Python 3.10/3.11/3.12.
- MIT license, README, CHANGELOG.

### Changed — Fase 1
- Reorganized repository layout: `data/` is now under the repo root, gitignored.
- Legacy scripts moved from `src/*.py` and `archive/Old/` into `legacy/scripts/`
  and `legacy/Old/`.

### Removed — Fase 1
- Empty placeholder files at repo root (`App`, `Impostazioni`, `alias`).
- Duplicate working copies previously nested under `2024_QIMP/QIMP/` and
  `2024_QIMP/2024_QIMP/QIMP/`.

### Notes

Stubs remaining for v0.2:
- `qimp.encoding.mcrqi` / `qimp.encoding.ncqi` (RGB extensions)
- `qimp.encoding.compression` (Quine–McCluskey boolean minimisation)
- `qimp.processing.arithmetic` (`q_ADD/SUB`, comparator, sort)
- `qimp.io.datasets`, `qimp.qml.classifier`
