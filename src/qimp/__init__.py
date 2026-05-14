"""QIMP — Quantum Image Processing library.

Sub-packages
------------
encoding   Image encodings (FRQI, NEQR, QPIE, MCRQI, NCQI) + compression.
processing Geometric, chromatic, arithmetic, and filtering operations on encoded images.
qml        Variational quantum classifiers on encoded images.
io         Image and dataset loaders.
runtime    Performance utilities (memory pool, simulator manager, caching).
testing    Wrappers around Qiskit Aer simulators and IBM Quantum hardware.
metrics    Figures of merit (PSNR, MSE, TV, transpile summary).

Scalability
-----------
All encoders accept arbitrary `n` (spatial qubits per axis, image size = 2^n × 2^n),
arbitrary `q` (intensity qubits for NEQR), and arbitrary `m` (number of images for
multi-image FRQI). No hard-coded qubit counts.

Specification: see docs/tesi.pdf, Chapter 3.
"""

from qimp._version import __version__

__all__ = ["__version__"]
