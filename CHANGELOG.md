# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
