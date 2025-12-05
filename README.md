# QIMP codebase (pulizia struttura)

Struttura corrente (solo codice, niente immagini o venv):
- `src/` — script attivi rinominati in snake_case (es. `gp_quantum_4.py`, `quantize_analyze_quantum_gp_v2.py`, `quantum_utils.py`).
- `archive/Old/` — versioni storiche dei vecchi script, conservate ma non mantenute.
- `.gitignore` — esclude `.venv/`, cache Python, immagini (`*.tif`, `*.tiff`), directory di dati (`IMMAGINI PER QUANTUM/`) e configurazioni IDE locali.

Esecuzione rapida:
```bash
python src/gp_quantum_4.py
python src/quantize_analyze_quantum_gp_v2.py
python src/generate_gp_images.py
```

Note:
- Le immagini e l'ambiente virtuale non sono versionati.
- Se servono input da `IMMAGINI PER QUANTUM/`, assicurati che la cartella esista localmente: è ignorata dal repository.
