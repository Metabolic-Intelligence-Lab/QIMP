"""Preview of the JOSS paper as a docx.

JOSS itself builds the published PDF from paper.md via pandoc; this script
exists only to give the authors a quick, readable preview before
submission.
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.shared import Cm, Inches, Pt, RGBColor

REPO = Path(
    "/mnt/c/Users/Giuseppe/OneDrive - Università Cattolica del Sacro Cuore/"
    "Metabolic Intelligence - Projects-MI/2024_QIMP/repo"
)
OUT = REPO / "paper" / "joss_paper_preview.docx"
FIG = REPO / "paper" / "figures"

SERIF = "Times New Roman"
DARK = RGBColor(0x1F, 0x3A, 0x5F)
GREY = RGBColor(0x55, 0x55, 0x55)


def setup(doc: Document) -> None:
    n = doc.styles["Normal"]
    n.font.name = SERIF
    n.font.size = Pt(11)
    n.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    n.paragraph_format.space_after = Pt(6)
    for level, size in [(1, 16), (2, 13), (3, 12)]:
        h = doc.styles[f"Heading {level}"]
        h.font.name = SERIF
        h.font.size = Pt(size)
        h.font.color.rgb = DARK
        h.font.bold = True
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(4)


def add_para(doc: Document, text: str) -> None:
    doc.add_paragraph(text)


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(item, style="List Bullet")
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        p.paragraph_format.space_after = Pt(2)


def build() -> Document:
    doc = Document()
    setup(doc)

    sec = doc.sections[0]
    sec.top_margin = Cm(2.5)
    sec.bottom_margin = Cm(2.5)
    sec.left_margin = Cm(2.5)
    sec.right_margin = Cm(2.5)

    # ---- header --------------------------------------------------------
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("qimp-mi: a scalable Python library for Quantum Image Processing on Qiskit")
    r.bold = True
    r.font.size = Pt(16)
    r.font.color.rgb = DARK

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sr = sub.add_run("Software paper, prepared for submission to The Journal of Open Source Software (JOSS)")
    sr.italic = True
    sr.font.size = Pt(10)
    sr.font.color.rgb = GREY

    doc.add_paragraph()

    authors = doc.add_paragraph()
    authors.alignment = WD_ALIGN_PARAGRAPH.CENTER
    authors.add_run("Author 1¹*, Author 2¹, Giulio Dolciami²").font.size = Pt(11)

    aff = doc.add_paragraph()
    aff.alignment = WD_ALIGN_PARAGRAPH.CENTER
    aff.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    aff.add_run(
        "¹ Metabolic Intelligence Lab, Università Cattolica del Sacro Cuore, Milan, Italy\n"
        "² Department of Electronics and Telecommunications, Politecnico di Torino, Turin, Italy\n"
        "* Corresponding author: email@unicatt.it"
    ).font.size = Pt(10)

    doc.add_paragraph()

    # ---- Summary -------------------------------------------------------
    doc.add_heading("Summary", level=1)
    add_para(
        doc,
        "qimp-mi is a Python library that implements the canonical encodings, "
        "processing operations and benchmarking tools of the Quantum Image "
        "Processing (QIMP) field, on top of Qiskit 2.x. The library covers five "
        "image encodings — FRQI (Le et al. 2011), NEQR (Zhang et al. 2013), "
        "QPIE, MCRQI (Sun et al. 2013) and NCQI (Sang et al. 2017) — together "
        "with geometric transformations (axis flip, coordinate swap, rotation, "
        "cyclic shift, region-restricted variants), chromatic operations (color "
        "complement, color change, half-intensity, threshold classify), quantum "
        "arithmetic (ripple-carry adder, two's-complement subtractor, NEQR "
        "comparator), the QHED edge-detection filter, a Quine–McCluskey based "
        "circuit compressor, a variational FRQI classifier, and a ratiometric "
        "Green–Purple (GP) sub-circuit with a recently-derived closed-form "
        "parameter solution. All operations are parametrised over the spatial "
        "qubit count n and the intensity qubit count q; no fixed image size or "
        "bit depth is assumed.",
    )
    add_para(
        doc,
        "The library is shipped with a Streamlit-based interactive explorer "
        "that walks an image end-to-end through the load → preprocess → encode "
        "→ process → execute pipeline, including direct submission to IBM "
        "Quantum hardware via the Sampler primitive. A unit suite of 338 tests "
        "covers every public function in a grid over n and q and verifies exact "
        "encoder/processor behaviour against statevector simulation. Type "
        "checking (mypy --strict), linting (ruff), documentation (mkdocs-"
        "material with API auto-generation via mkdocstrings), and packaging "
        "(PEP 621 with hatchling) are all in place; CI runs the suite on three "
        "operating systems and three Python versions.",
    )

    # ---- Statement of need --------------------------------------------
    doc.add_heading("Statement of need", level=1)
    add_para(
        doc,
        "Quantum image processing has accumulated more than twenty years of "
        "algorithm-level work — five competing encodings, dozens of processing "
        "operations, and several application-specific extensions (multi-channel, "
        "compression, classification) (Yan & Venegas-Andraca 2025; Lisnichenko "
        "& Protasov 2023; Farooq et al. 2025). Despite this, the field lacks a "
        "maintained, tested, scaling open-source library. Most published QIMP "
        "work re-implements the FRQI or NEQR encoder from scratch, with "
        "hard-coded qubit counts and ad-hoc validation, slowing comparison "
        "across papers and making reproducibility difficult.",
    )
    add_para(
        doc,
        "qimp-mi is intended to be the missing library. It provides the "
        "complete encoder/processor catalog of the Dolciami thesis (Dolciami "
        "2022), factored into composable, type-annotated primitives with "
        "parametric n/q everywhere. By treating QIMP as a software-engineering "
        "target rather than a per-paper scratchpad we hope to (a) lower the "
        "entry barrier for new researchers (install-and-run rather than "
        "translate-from-pseudocode), (b) enable side-by-side benchmarking "
        "across encodings (the bundled Benchmark page in the explorer runs "
        "FRQI, NEQR and QPIE on the same image and emits a comparison table of "
        "qubit count, depth, transpiled depth, runtime and PSNR), and (c) "
        "provide a reference implementation against which new methodological "
        "contributions can be compared.",
    )
    add_para(
        doc,
        "The development of the library directly produced a new methodological "
        "result: a closed-form analytical solution for the parameters of a "
        "variational FRQI-based Green–Purple ratio circuit, derived from a "
        "careful re-examination of the canonical ratio ansatz. This result is "
        "reported separately (companion manuscript in preparation); the "
        "library ships the analytical_gp_params solver, the corrected "
        "per-pixel apply_gp_function circuit, and a reproducible Streamlit "
        "validation page so that the derivation can be re-checked end-to-end "
        "against the classical GP reference on any RGB tile.",
    )

    # ---- Functionality ------------------------------------------------
    doc.add_heading("Functionality", level=1)

    if (FIG / "fig_qimp_architecture.png").exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(
            str(FIG / "fig_qimp_architecture.png"), width=Inches(6.0)
        )
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.LEFT
        cap.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        cap_run = cap.add_run("Fig. 1 ")
        cap_run.bold = True
        cap_run.font.size = Pt(10)
        txt = cap.add_run(
            "Architecture of the qimp-mi library. Bottom layer: the "
            "external dependencies (Qiskit 2.x, qiskit-aer, NumPy, SciPy, "
            "Pillow). Middle layer: the four problem-domain subpackages "
            "(encoding, processing, qml, io). Above them: the runtime / "
            "metrics / testing helpers used by every other module. Top: "
            "the Streamlit explorer application and its IBM Quantum "
            "Runtime hook."
        )
        txt.font.size = Pt(10)

    add_para(
        doc,
        "The library is organised into five subpackages plus a top-level "
        "runtime layer:",
    )
    add_bullets(
        doc,
        [
            "qimp.encoding — FrqiEncoder, NeqrEncoder, QpieEncoder, "
            "McrqiEncoder, NcqiEncoder, and a compression module implementing "
            "Quine–McCluskey minimisation with a greedy disjoint cover. Each "
            "encoder exposes encode(image) -> QuantumCircuit and "
            "decode(counts) -> ndarray, with n and (where applicable) q "
            "inferred from the input.",
            "qimp.processing — geometric, chromatic, arithmetic, filters "
            "(QHED), and gp_ratio modules. All ops act in place on a supplied "
            "QuantumCircuit and accept a configurable pos_offset so they work "
            "uniformly across single-channel and multi-channel encodings.",
            "qimp.qml — FrqiClassifier, a variational quantum classifier "
            "built on top of qimp.encoding.frqi plus the functional "
            "qiskit.circuit.library.real_amplitudes ansatz, trained against "
            "±1 labels via scipy.optimize.minimize (COBYLA).",
            "qimp.io — TIFF loaders preserving 16-bit depth, "
            "calculate_gp_image for the classical Green–Purple reference, "
            "dataset iterators and batch-processing helpers.",
            "qimp.runtime — Aer-backed simulator selection (CPU / GPU), "
            "performance monitoring, image-buffer memory pool, LRU-cached "
            "base circuits.",
            "qimp.metrics — MSE, PSNR (with explicit float-image warning), "
            "total variation, and a transpile_summary helper used by the "
            "benchmarking page.",
            "qimp.testing — ideal_simulation, noisy_simulation, "
            "exact_counts (statevector → counts, deterministic), and "
            "device_test (IBM Quantum execution via the Sampler primitive).",
            "apps/qimp_explorer — Streamlit application with a 5-step "
            "wizard (Load → Preprocess → Encode → Process → Execute) plus "
            "three add-on pages (Benchmark, GP-ratio, System Info). Circuit "
            "export to OpenQASM 3, IBM Quantum submission, and bulk save of "
            "TIFF outputs are all built in.",
        ],
    )
    add_para(
        doc,
        "The qimp ui CLI shortcut launches the Streamlit application. The "
        "package exposes optional dependency extras [ui], [ibm], [gpu], [qml] "
        "and [docs] so users only install what they need.",
    )

    # ---- Comparison ---------------------------------------------------
    doc.add_heading("Comparison to existing software", level=1)
    add_para(
        doc,
        "There is, to our knowledge, no maintained library on PyPI that "
        "implements the QIMP catalog at this scope. The closest neighbours are:",
    )
    add_bullets(
        doc,
        [
            "Qiskit itself, which provides the base QuantumCircuit and "
            "simulator infrastructure but no QIMP-specific encoders, "
            "processors, or benchmarks.",
            "PennyLane and TensorFlow Quantum, which target variational "
            "quantum machine learning broadly but have no first-class "
            "FRQI/NEQR support.",
            "qiskit-machine-learning, which provides VQC and QNN scaffolding "
            "but again does not implement QIMP encodings.",
            "Numerous unmaintained research scripts on GitHub that implement "
            "one or two encodings, usually with hard-coded qubit counts and "
            "without tests.",
        ],
    )
    add_para(
        doc,
        "qimp-mi is complementary to all of these: it depends on Qiskit for "
        "the circuit primitives and on Aer for simulation, and adds the "
        "QIMP-specific layer on top.",
    )

    # ---- Acknowledgements --------------------------------------------
    doc.add_heading("Acknowledgements", level=1)
    add_para(
        doc,
        "We thank the members of the Metabolic Intelligence Lab for the "
        "fluorescence-microscopy dataset used to validate the GP-ratio "
        "pipeline, and Politecnico di Torino for the original thesis spec that "
        "defines the library's encoder/processor catalogue.",
    )

    doc.add_heading("References", level=1)
    add_para(
        doc,
        "References are managed in joss_paper.bib (BibTeX) and rendered by "
        "JOSS at build time from paper.md. See joss_paper.bib for the full "
        "list (11 entries covering FRQI, NEQR, MCRQI, NCQI, the three recent "
        "QIMP reviews, the Dolciami thesis, Qiskit, and the companion "
        "methods paper in preparation).",
    )

    return doc


def main() -> int:
    doc = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
