# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] - 2026-10-20

Version described in the SoftwareX software paper. Completes the parts of
the library that were documented as planned.

### Added
- `qimp.encoding.compression`: `espresso_minimize`, a pure-Python
  implementation of the ESPRESSO heuristic (expand, irredundant, reduce),
  and `make_disjoint_cover`. `FrqiCompressor`, `NeqrCompressor` and
  `position_strings_to_implicants` take `method="auto" | "qm" | "espresso"`;
  `"auto"` keeps Quine-McCluskey up to `n = 4` and uses ESPRESSO above, so
  existing results for small images are unchanged.
- `qimp.runtime.zne` (new module): `fold_gates` (global unitary folding
  after transpilation, with barriers that survive optimisation),
  `zne_probabilities` and `zne_extrapolate` (Richardson, linear or
  exponential extrapolation of bitstring probabilities to zero noise).
  `hw_run(mitigation="zne")` submits the folded circuits as one job on top
  of TREX and dynamical decoupling and records the per-scale counts in its
  summary; `aer_noisy_run` takes `zne_scales` and `seed` to run the same
  path on a noise model.
- `qimp.processing.arithmetic`: `neqr_compare_swap` (comparator followed
  by Fredkin gates, with one flag qubit that records the swap) and
  `neqr_sort`, an odd-even transposition network that sorts `k` intensity
  registers with `k(k-1)/2` compare-and-swaps, in ascending or descending
  order, plus their exact inverses `neqr_compare_swap_inv` and
  `neqr_sort_inv`, which restore the inputs and clear the flags.
- `qimp.processing.ratiometric_circuit`: the Class A builders
  (`class_a_gp_prefix`, `class_a_gp_full`, `class_a_gp_lookup`) accept any
  positive G-factor through the keywords `G` and `g_frac_bits`. G is
  represented in fixed point as `K / 2^f` (`gp_g_fixed_point`), `I_b` is
  multiplied by `K` with `q_mul_const` and `I_a` is shifted by `f` bits, so
  the ratio is unchanged. The default `G = 1.0` produces the same circuits
  as before; values that are not dyadic need an explicit `g_frac_bits`.
  `class_a_gp_lookup`, `class_c_rogfp_lookup` and `gp_g_fixed_point` are
  now listed in `__all__`.
- `qimp.testing.ideal_simulation` takes an optional keyword `seed` that
  fixes the transpiler and simulator seeds. The default (`None`) keeps the
  previous behaviour.
- API reference pages for every module (MCRQI, NCQI, compression,
  arithmetic, ratiometric circuits, classifier, datasets, IBM runtime, ZNE
  and the runtime helpers were missing), and a documentation page on
  simulation, hardware execution and error mitigation.

### Changed
- Documentation rewritten to match the code. The previous pages still
  described arithmetic, MCRQI and NCQI as unimplemented and listed a
  roadmap of features that had already been released; the FRQI examples
  used `decode(counts)[0]`, which returns the first row of the image.
- Module docstrings cite the original publications instead of the thesis
  file `docs/tesi.pdf`, which is not part of the repository.
- README and this changelog in a plain style.

### Fixed
- `scripts/softwarex_figures.py`: the QHED panel drops the wrap-around row
  of the vertical pass as well as the wrap-around column of the horizontal
  pass. The axis-flip check compares against the row inversion only. The
  shot-sampled FRQI run uses `ideal_simulation(seed=...)`. Panel titles
  follow `n` and `q`. The NEQR circuit and the frame are built once.
- `examples/05_real_image_pipeline.py` falls back to the bundled Laurdan
  frame also when `Train_QML_16/` holds only blank tiles. Its messages name
  the data actually used.
- `frqi_color_complement`: the docstring gave the angle map as
  `theta -> pi - theta`; the gate implements `theta -> pi/2 - theta`.
- Docstrings of `q_div_restoring` and of `qimp.processing.ratiometric_circuit`
  no longer describe inverses and operators that exist as missing.
- `q_mul_const` documented the accumulator as "not assumed to be initially
  zero"; the shifted adds do not propagate a carry out of their window, so
  the result is `b * k` only from a zeroed accumulator. The docstring now
  states the precondition. The circuit is unchanged.
- Stale comments in `qimp.processing.ratiometric_circuit` that described
  the Class C composition as not packaged, although `class_c_rogfp_full`
  exists.
- The 0.3.0 entry of this changelog had been split by a misplaced heading.

## [0.3.0] - 2026-10-19

Reversible arithmetic on NEQR intensity registers, the ratiometric circuits
built on it, an amplitude-estimation oracle and an IBM Quantum runtime layer.
Also prepares the repository for the SoftwareX software paper.

### Added
- `scripts/softwarex_figures.py`, which regenerates the figures and numbers of
  the software paper from the bundled Laurdan frame on a simulator.
- `CITATION.cff`.
- Arithmetic: inverse and controlled variants of every adder (`qc_add_1_inv`,
  `qc_add_1_ctrl`, `qc_add_1_ctrl_inv`, `q_add_inv`, `q_add_ctrl`,
  `q_add_ctrl_inv`, `q_sub_inv`, `q_sub_ctrl`, `q_sub_ctrl_inv`). Exact
  inverses are needed to use these blocks inside an amplitude-estimation
  state preparation.
- Arithmetic: `q_mul_const` and `q_mul_const_inv`, shift-add multiplication by
  a classical constant.
- Arithmetic: `q_div_restoring` (q by q) and `q_div_general` (n by m)
  shift-subtract dividers with divide-by-zero flags, and `q_div_restoring_inv`.
  `q_div` is an alias of the default divider.
- Arithmetic: `q_div_nonrestoring` and `q_div_nonrestoring_inv`, and the
  `q_add_sub_ctrl` and `q_add_sub_ctrl_inv` blocks they use. The non-restoring
  divider is equivalent to the restoring one and needs 48 to 61 % fewer
  two-qubit gates after transpilation.
- New module `qimp.processing.ratiometric_circuit`:
  - `class_b_ratio`, `class_b_ratio_inv` and `decode_class_b_ratio` compute the
    integer ratio `I_a // I_b` per pixel, with a divide-by-zero flag, on two
    co-registered NEQR images loaded by `dual_neqr_load` and
    `dual_neqr_load_inv`.
  - `class_a_gp_prefix` and `decode_class_a_prefix` compute the numerator and
    denominator of the Laurdan generalised polarisation; `class_a_gp_full` and
    `decode_class_a_full` compute the signed fractional GP in [-1, 1].
  - `class_c_rogfp_full` and `decode_class_c_rogfp` compute a fractional ratio
    followed by `affine_subtract_constant`, giving the calibrated roGFP redox
    ratio in [0, 1].
  - `mark_good_oracle` and `mark_good_oracle_inv`, a self-inverse threshold
    predicate on the quotient register, corrected for the divide-by-zero flag,
    for use in amplitude estimation.
- `qimp.runtime.ibm`: `get_service`, `list_backends`, `pick_backend` (by name
  or least busy), `hw_run` on SamplerV2 with TREX, dynamical decoupling or ZNE
  mitigation, `aer_noisy_run` through `AerSimulator.from_backend`, and
  `persist_run` and `is_run_complete` for QPY and JSON archival of runs.
- `qimp.runtime.circuits`: `CircuitRecipe` and `build_recipes`, one row of the
  sweep matrix (circuit, decoder, classical reference) per encoder for FRQI,
  multi-image FRQI, NEQR, QPIE, MCRQI, NCQI and GP.

### Fixed
- README quick start: `FrqiEncoder.decode` takes the counts only. The example
  passed an `n=` keyword that the method does not accept.
- Lint, formatting and `mypy` errors that made the CI `lint` job fail.
- `decode_class_a_full` read only the low `q_frac` bits of the quotient, so
  pixels at GP = -1 or +1 decoded as 0.0. It now reads the full register.
- `gp_ratio`: closed-form parameters and corrected placement of the Hadamard
  gates.
- `decode_class_b_ratio` defined its `bit_at` helper inside the counts loop
  (ruff B023). The helper is now defined once.
- `q_add_sub_ctrl`, `q_add_sub_ctrl_inv`, `q_div_nonrestoring` and
  `q_div_nonrestoring_inv` were missing from `__all__`, so they did not appear
  in `import *` or in the API documentation.

### Changed
- `ruff` and `mypy` are pinned in the dev extra and match the pre-commit
  hooks. The unpinned `ruff>=0.4` had made CI fail on rules the hook did not
  run.
- The mypy target is Python 3.12 (the supported runtime is still 3.10 or
  later), because the numpy stubs cannot be parsed for earlier versions.
- The IBM hardware smoke test is skipped, rather than failed, when the `[ibm]`
  extra or saved credentials are missing.

## [0.2.0] - 2026-05-18

Implements the library modules that were still stubs: RGB encodings, NEQR
arithmetic, FRQI and NEQR compression, the variational image classifier and
dataset iteration.

### Added
- `qimp.encoding.mcrqi`: FRQI extension to RGB, with `2n + 3` qubits, one
  colour qubit per channel and a controlled RY per pixel and channel.
  `McrqiEncoder` has the same interface as `FrqiEncoder`.
- `qimp.encoding.ncqi`: NEQR extension to RGB, with `2n + 3q` qubits and one
  intensity register per channel. Retrieval is exact.
- `qimp.encoding.compression`: Quine-McCluskey prime implicants with a greedy
  disjoint cover. `FrqiCompressor` and `NeqrCompressor` emit reduced sequences
  of multi-controlled gates. The cover must be disjoint for the result to be
  correct.
- `qimp.processing.arithmetic`: `qc_add_1`, `q_add` (ripple-carry adder of any
  width), `q_sub` (two's complement) and `neqr_comparator` (sets a flag qubit
  when a > b).
- `qimp.qml.classifier`: `FrqiClassifier`, a binary variational classifier
  with a `RealAmplitudes` ansatz on FRQI features, trained with COBYLA
  (`scipy.optimize.minimize`) on labels of -1 and +1.
- `qimp.io.datasets`: `ImageDataset`, a folder iterator with square,
  power-of-two and blank-image filters, and `batch_process_images`.
- 49 tests for the new modules.

### Changed
- `apps/qimp_explorer/pages/5_System_Info.py` lists the current
  implementation status. Only `neqr_sort` and the ESPRESSO compression
  heuristic remain unimplemented.

## [0.1.1] - 2026-05-14

Fixes the findings of the v0.1.0 code review.

### Fixed
- `encoding/frqi.py`: float images with values above 1 were saturated, because
  the default normalisation was 1.0 for every float dtype. The new helper
  `_infer_normalization` uses `image.max()` for floats and the dtype maximum
  for integers.
- `runtime/memory_pool.py`: `get_image_buffer` returned the same buffer on
  consecutive calls. The cursor now advances on every call, as documented and
  as in `get_angle_buffer`.
- `processing/geometric.py`: `pos_shift` used `axis` before validating it, so
  invalid values fell through to the X branch. Validation now comes first. The
  new helper `_validate_position_register`, called by every public geometric
  operation, raises an error when `pos_offset + 2n` does not fit in the
  circuit.
- `processing/filters.py` no longer calls the private
  `qpie._validate_and_normalize`. The helper is public as
  `qimp.encoding.qpie.normalize_amplitudes`.
- `encoding/qpie.py`: an all-zero image is still encoded as a uniform
  superposition, and now raises a `UserWarning`, because decoders with
  `rms != 0` return spurious intensities for it.
- `metrics.psnr` raises a `UserWarning` for float images with values above 1.0
  when `max_intensity` is not given. It used to return very low values without
  notice.

### Changed
- `qhed_filter` documents that the cyclic decrement gives a gradient of the
  row-major flattened image, not a pure horizontal gradient, and that the
  wrap-around at row boundaries is part of the algorithm.
- `metrics.total_variation`: the docstring now says isotropic
  (`sqrt(dx^2 + dy^2)`), as implemented, instead of anisotropic.
- `pyproject.toml`: pytest deselects `slow` tests by default. A separate CI job,
  `test-slow`, runs all tests.
- The private helpers `_increment` and `_decrement` of the geometric module are
  replaced by `cyclic_increment` and `cyclic_decrement` in the new public module
  `qimp.runtime.arithmetic_gates`, used by `pos_shift` and `qhed_filter`.

### Added
- Regression tests for each fix above.
- Round-trip tests for `restr_flip` and `restr_coord_swap`, which previously
  had input-validation tests only.
- Round-trip tests for `cyclic_increment` and `cyclic_decrement`.
- A `transpile_summary` test with a custom basis.
- `docs/encoding.md`: table of the qubit-layout conventions.
- `docs/processing.md`: note on the QHED wrap-around.

## [0.1.0] - 2026-05-14

First tagged release. The repository was private at the time and the package
was not published on PyPI.

### Security
- An IBM Quantum API token was hard-coded in eight legacy scripts under
  `legacy/Old/`. It is replaced by `<REDACTED-IBM-Q-TOKEN>`. The token is still
  in the earlier history of the repository and has been revoked on IBM
  Quantum; clones made before v0.1.0 contain it.

### Added
- `examples/` with four runnable scripts: `01_frqi_fidelity.py`,
  `02_neqr_round_trip.py`, `03_qhed_edges.py` and `04_qpie_round_trip.py`.
- Documentation with mkdocs (Material theme) and `mkdocstrings[python]` for
  the API reference: `docs/index.md`, `docs/getting-started.md`,
  `docs/encoding.md`, `docs/processing.md`, `docs/examples.md` and four API
  pages. `.github/workflows/docs.yml` builds and deploys them to GitHub Pages
  on every push to `main`.
- `CONTRIBUTING.md` with setup, design invariants and commit conventions.
- `qimp.encoding.qpie` (amplitude encoding with `qc.initialize`) and
  `qimp.encoding.neqr` (basis encoding with exact retrieval).
- `qimp.processing.geometric`: `axis_flip`, `coord_swap`, `ort_rotation`,
  `pos_shift`, `restr_flip` and `restr_coord_swap`, independent of the encoding
  through `pos_offset`.
- `qimp.processing.chromatic`: colour complement and colour change for FRQI and
  NEQR, half intensity and classification for NEQR.
- `qimp.processing.filters`: quantum Hadamard edge detection for QPIE, with
  `qhed_filter`, `qhed_decode` and `qhed_full_edges`.
- The scripts in `legacy/scripts/` migrated to typed, tested modules:
  `qimp.metrics`, `qimp.runtime`, `qimp.io.image`, `qimp.qft`, `qimp.testing`,
  `qimp.encoding.frqi` and `qimp.processing.gp_ratio`.
- 159 tests, parametrised over `n` (spatial qubits) and `q` (intensity qubits).
- Package layout `src/qimp/` with the subpackages `encoding`, `processing`,
  `qml`, `io` and `runtime`, and the modules `qft`, `testing`, `metrics`,
  `config` and `cli`.
- `pyproject.toml` (hatchling, PEP 621) and configuration for ruff, mypy,
  pytest and pre-commit.
- GitHub Actions CI with lint and tests on Python 3.10, 3.11 and 3.12.
- MIT licence, README and CHANGELOG.

### Changed
- `data/` is at the repository root and gitignored.
- The legacy scripts moved from `src/*.py` and `archive/Old/` to
  `legacy/scripts/` and `legacy/Old/`.

### Removed
- Empty placeholder files at the repository root (`App`, `Impostazioni`,
  `alias`).
- Duplicate working copies under `2024_QIMP/QIMP/` and
  `2024_QIMP/2024_QIMP/QIMP/`.

### Not yet implemented
- `qimp.encoding.mcrqi` and `qimp.encoding.ncqi` (RGB extensions).
- `qimp.encoding.compression` (Quine-McCluskey minimisation).
- `qimp.processing.arithmetic` (addition, subtraction, comparator, sort).
- `qimp.io.datasets` and `qimp.qml.classifier`.
