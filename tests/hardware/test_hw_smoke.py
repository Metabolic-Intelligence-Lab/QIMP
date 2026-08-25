"""End-to-end smoke test against a real IBM Quantum backend.

Excluded from CI by the `hardware` marker (see pyproject.toml's pytest
addopts which selects `not slow and not hardware`, and the explicit
`and not hardware` in the CI test-slow job, whose `-m` overrides addopts).
It also skips itself if the `[ibm]` extra or the saved credentials are
missing, so it degrades to a skip rather than a failure wherever it is
selected without a device to talk to.

Run manually with:

    pytest -m hardware tests/hardware/test_hw_smoke.py -v

Burns approximately 10 seconds of real QPU time. Requires saved IBM
Quantum credentials at ~/.qiskit/qiskit-ibm.json. Asserts the pipeline
end-to-end (submission, transpile, counts, decode), NOT the device
quality — the PSNR floor is set deliberately low.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CREDENTIALS = Path.home() / ".qiskit" / "qiskit-ibm.json"


@pytest.mark.hardware
def test_gp_n1_runs_on_real_backend() -> None:
    """Submit the smallest GP recipe (n=1, 4 qubits) to a real backend
    via Sampler v2 + TREX + DD, decode, and assert a generous PSNR floor."""
    pytest.importorskip(
        "qiskit_ibm_runtime",
        reason="needs the [ibm] extra: pip install -e '.[dev,ibm]'",
    )
    if not CREDENTIALS.exists():
        pytest.skip(f"no saved IBM Quantum credentials at {CREDENTIALS}")

    from qimp.metrics import psnr
    from qimp.runtime import ibm
    from qimp.runtime.circuits import build_recipes

    rgb = np.random.default_rng(0).integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    recipes = build_recipes(rgb, n=1, alpha=0.5)
    gp = next(r for r in recipes if r.encoder == "gp")

    service = ibm.get_service()
    backend = ibm.pick_backend(service, min_qubits=gp.qc.num_qubits)

    counts, transpiled, job_id, summary = ibm.hw_run(
        gp.qc,
        backend=backend,
        shots=1024,
        mitigation="trex+dd",
    )

    assert job_id, "hw_run should return a non-empty job_id"
    assert isinstance(job_id, str)
    assert transpiled.num_qubits >= 2
    assert "depth" in summary

    decoded = gp.decoder(counts)
    assert decoded.shape == (2, 2)  # 2^n x 2^n for n=1

    snr = float(psnr(gp.reference, decoded, max_intensity=2.0))  # GP range [-1,1]
    assert snr >= 5.0, (
        f"smoke-test PSNR too low: {snr:.1f} dB — device behaviour anomaly? "
        f"job_id={job_id}, backend={backend.name}"
    )
