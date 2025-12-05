# QIMP

Struttura ripulita del progetto di Quantum Image Processing. Abbiamo separato il codice attivo dai file storici e aggiunto ignorati per evitare che asset pesanti finiscano nel repo.

## Struttura
- `src/` – script attivi (es. `quantum_utils.py`, `generate_gp_images.py`, `GP Quantum 4.0.py`, ecc.).
- `archive/` – codice e asset storici ereditati dalla cartella `Old/` (non mantenuti).
- `.gitignore` – esclude cache Python, ambienti virtuali e file immagine (`*.tif`, `*.png`, `*.jpg`).

## Uso rapido
Esegui gli script dalla radice del repo puntando a `src/`, ad esempio:
```bash
python src/quantum_utils.py
python "src/GP Quantum 4.0.py"
```
Se hai bisogno di importare moduli da `src/`, puoi esportare `PYTHONPATH`:
```bash
set PYTHONPATH=%cd%\\src   # Windows PowerShell
```

## Prossimi passi suggeriti
- Consolidare e rinominare gli script principali (nomi senza spazi) e centralizzare la logica condivisa in `src/quantum_utils.py`.
- Aggiungere `requirements.txt` e, se serve, `requirements-dev.txt` con pytest/ruff/black.
- Spostare eventuali test in `tests/` con `pytest` e aggiungere smoke test minimi per encoding/decoding e GP.
