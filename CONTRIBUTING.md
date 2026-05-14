# Contributing to QIMP

Thanks for your interest in QIMP! This guide covers the development workflow.

## Setup

```bash
git clone https://github.com/Metabolic-Intelligence-Lab/QIMP
cd QIMP
python -m venv .venv
source .venv/bin/activate            # On Windows: .venv\Scripts\activate
pip install -e ".[dev,docs]"
pre-commit install
```

## Test, lint, type-check

```bash
ruff check src tests
ruff format --check src tests
mypy src/qimp
pytest -v --cov=qimp
```

All four commands run in CI on every PR; PRs failing any of them will not be
merged.

The slow tests (round-trip simulations that need ≥ 40 000 shots) are skipped
by default. Run them explicitly with:

```bash
pytest -v -m slow
```

## Design constraints

When adding new code, please respect the library-wide invariants:

1. **No hard-coded qubit counts.** Every encoder/processor accepts `n`, `q`,
   `m` as parameters. Tests must be parametrized via the `n_qubits` and
   `q_qubits` fixtures defined in `tests/conftest.py`.
2. **Encoding-agnostic processing where possible.** Position-qubit operations
   (geometric transforms) take a `pos_offset` argument so the same function
   works on FRQI, NEQR, and QPIE.
3. **Pure functions over hidden state.** Encoders expose both a stateless
   functional API (`frqi_circuit`, `frqi_decode`) and a thin class wrapper
   (`FrqiEncoder`) that carries `n`, `q`, normalisation across calls.
4. **No mutable module globals.** Singletons (e.g. `SimulatorManager`) carry a
   `reset()` classmethod for tests.

## Style

- Python ≥ 3.10 syntax. Use `|` union types, `dict[…]`, `list[…]`, etc.
- `from __future__ import annotations` at the top of every module.
- Numpy-style docstrings.
- Type annotations on every public symbol. `Any` is allowed only for Qiskit
  return types that mypy can't infer.
- Public symbols listed in `__all__`, sorted alphabetically (RUF022 enforces).

## Commit messages

Conventional Commits prefix: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`,
`test:`, `ci:`. Imperative mood; capitalize the first word; no trailing period.

```
feat: add NCQI encoder for RGB images

…body explaining what and why…
```

## Documentation

Docstrings are picked up by `mkdocstrings` and rendered in the API reference
section. Run `mkdocs serve` locally to preview.

## License

By contributing you agree to license your contributions under the MIT license,
the same as the project itself.
