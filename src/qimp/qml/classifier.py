"""Variational Quantum Classifier on FRQI-encoded images.

The model:

    image  ──►  FrqiEncoder.encode  ──►  RealAmplitudes(reps)  ──►  Z⊗I⊗…⊗I  ──►  ŷ
                                                                                ▲
                                                                                │
                                            scipy.optimize.minimize(COBYLA)  ───┘

A binary classifier: the expectation value of ``Z`` on the FRQI color qubit
predicts the label. Train by minimising the mean squared loss against ``±1``
labels.

This is a minimal-but-functional implementation. For production use, see
``qiskit-machine-learning``'s ``VQC`` / ``NeuralNetworkClassifier``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

# Qiskit ≥ 2.1: the functional ``real_amplitudes`` is the replacement for the
# now-deprecated ``RealAmplitudes`` class (removed in 3.x). Fall back to the
# class form on older Qiskit so users on 1.x still work.
try:
    from qiskit.circuit.library import real_amplitudes as _real_amplitudes
except ImportError:  # pragma: no cover - older Qiskit
    from qiskit.circuit.library import RealAmplitudes as _RealAmplitudesClass

    def _real_amplitudes(num_qubits: int, *, reps: int) -> QuantumCircuit:
        return _RealAmplitudesClass(num_qubits=num_qubits, reps=reps)


from qimp.encoding.frqi import FrqiEncoder

__all__ = ["FrqiClassifier", "FrqiClassifierResult"]


class FrqiClassifierResult:
    """Container for the outcome of `FrqiClassifier.fit`."""

    def __init__(
        self,
        params: np.ndarray,
        loss_history: list[float],
        n_iter: int,
    ) -> None:
        self.params = params
        self.loss_history = loss_history
        self.n_iter = n_iter

    def __repr__(self) -> str:
        final = f"{self.loss_history[-1]:.4f}" if self.loss_history else "nan"
        return f"FrqiClassifierResult(n_iter={self.n_iter}, final_loss={final})"


class FrqiClassifier:
    """Binary classifier on FRQI-encoded grayscale images.

    Parameters
    ----------
    reps
        Depth of the variational ansatz (``RealAmplitudes``).
    seed
        Optional seed for parameter initialisation.

    Examples
    --------
    >>> images, labels = …
    >>> clf = FrqiClassifier(reps=2, seed=0)
    >>> result = clf.fit(images, labels, max_iter=50)
    >>> predictions = clf.predict(images)
    """

    def __init__(self, *, reps: int = 2, seed: int | None = None) -> None:
        if reps < 1:
            raise ValueError("reps must be >= 1")
        self.reps = reps
        self.seed = seed
        self._encoder = FrqiEncoder()
        self._params: np.ndarray | None = None
        self._n: int | None = None
        self._ansatz: QuantumCircuit | None = None

    # ----------------------------- public API ----------------------------- #

    def predict_expectation(self, image: np.ndarray) -> float:
        """Return ``⟨Z⟩`` on the FRQI color qubit for the given image."""
        if self._params is None:
            raise RuntimeError("call fit() before predict_expectation()")
        return self._expectation(image, self._params)

    def predict(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Return ``+1`` / ``-1`` predictions for a batch of images."""
        if self._params is None:
            raise RuntimeError("call fit() before predict()")
        return np.array([1 if self._expectation(img, self._params) >= 0 else -1 for img in images])

    def fit(
        self,
        images: Sequence[np.ndarray],
        labels: Sequence[int],
        *,
        max_iter: int = 50,
    ) -> FrqiClassifierResult:
        """Train the variational ansatz against the (image, label) pairs.

        Labels must be ``+1`` / ``-1``. Uses ``scipy.optimize.minimize`` with COBYLA.
        """
        from scipy.optimize import minimize

        labels_arr = np.asarray(labels, dtype=np.float64)
        if labels_arr.shape != (len(images),):
            raise ValueError(f"labels length {labels_arr.shape} != number of images {len(images)}")
        if not np.all(np.isin(labels_arr, (-1.0, 1.0))):
            raise ValueError("labels must be ±1")
        if len(images) == 0:
            raise ValueError("at least one (image, label) pair is required")

        # All images must share the same dimensions for a single ansatz.
        first = np.asarray(images[0])
        for img in images[1:]:
            if np.asarray(img).shape != first.shape:
                raise ValueError("all images must have the same shape")
        n = int(math.log2(first.shape[0]))
        if n < 1 or first.shape[0] != first.shape[1]:
            raise ValueError(f"images must be square 2^n × 2^n, got shape {first.shape}")

        self._n = n
        self._ansatz = _real_amplitudes(num_qubits=2 * n + 1, reps=self.reps)
        num_params = len(self._ansatz.parameters)

        rng = np.random.default_rng(self.seed)
        x0 = rng.uniform(-math.pi, math.pi, size=num_params)

        history: list[float] = []

        def loss(theta: np.ndarray) -> float:
            preds = np.array([self._expectation(img, theta) for img in images])
            mse = float(np.mean((preds - labels_arr) ** 2))
            history.append(mse)
            return mse

        result = minimize(loss, x0, method="COBYLA", options={"maxiter": max_iter, "disp": False})
        self._params = np.asarray(result.x, dtype=np.float64)
        return FrqiClassifierResult(
            params=self._params,
            loss_history=history,
            n_iter=int(result.nfev),
        )

    # ----------------------------- internals ------------------------------ #

    def _expectation(self, image: np.ndarray, params: np.ndarray) -> float:
        """Return ``⟨ψ| Z₀ |ψ⟩`` where ψ is FRQI(image) followed by the ansatz."""
        qc = self._build_full_circuit(image, params)
        sv = Statevector.from_instruction(qc)
        # Color qubit is the highest-indexed one (FRQI layout).
        observable_str = "Z" + "I" * (qc.num_qubits - 1)
        observable = SparsePauliOp.from_list([(observable_str, 1.0)])
        return float(np.real(sv.expectation_value(observable)))

    def _build_full_circuit(self, image: np.ndarray, params: np.ndarray) -> QuantumCircuit:
        if self._n is None or self._ansatz is None:
            raise RuntimeError("internal state not initialised; call fit() first")
        # FRQI handles its own dtype/normalisation via _infer_normalization; don't
        # force-cast to uint8 here or 16-bit / float inputs get silently truncated.
        encoded = self._encoder.encode(image)
        # Positional binding by parameter order — same ansatz object that holds the
        # parameter symbols, so binding always finds them.
        bound = self._ansatz.assign_parameters(list(params))
        full = encoded.compose(bound)
        return full
