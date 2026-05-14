# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
