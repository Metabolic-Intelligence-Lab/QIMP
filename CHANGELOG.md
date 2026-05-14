# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Package scaffolding: `src/qimp/` with sub-packages `encoding`, `processing`, `qml`,
  `io`, `runtime`, plus modules `qft`, `testing`, `metrics`, `config`, `cli`.
- `pyproject.toml` (hatchling, PEP 621), ruff, mypy, pytest, pre-commit configs.
- GitHub Actions CI: lint + tests on Python 3.10/3.11/3.12.
- MIT license, README, CHANGELOG.
- Smoke tests + parametric fixtures for `n` (spatial qubits) and `q` (intensity qubits).

### Changed
- Reorganized repository layout: `data/` is now under the repo root, gitignored.
- Legacy scripts moved from `src/*.py` and `archive/Old/` into `legacy/scripts/` and
  `legacy/Old/`. These are kept for reference and will be migrated into the
  `qimp` package across Fase 3 (existing code) and Fase 4 (NEQR / QPIE / processing
  modules from the thesis).

### Removed
- Empty placeholder files at repo root (`App`, `Impostazioni`, `alias`).
- Duplicate working copies of the repository previously nested under
  `2024_QIMP/QIMP/` and `2024_QIMP/2024_QIMP/QIMP/`.
