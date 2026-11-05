# PlantVillage to PlantDoc baselines

Leakage-safe ResNet-50 and EfficientNet-B0 baselines for the laboratory-to-field shift from PlantVillage to PlantDoc, plus a classical lesion-preserving edit prototype (the working name in this repository is PPCE).

The checked-in metrics, figures, and manuscript draft come from one CPU run on a capped PlantVillage subset and the aligned PlantDoc images. They are measurements for that run. They are not a full-data benchmark and they are not a claim of a new state of the art.

## What is here

- `ppce/` — class alignment, group splits, training, evaluation, and the editor
- `scripts/prepare_data.py` — download and write a manifest
- `scripts/run_experiments.py` — train, score, plot, and render the draft
- `tests/` — split, editor, and training-step tests that do not need the datasets
- `docs/AUDIT.md` — what was in the repository and how the split was built
- `docs/REPRODUCIBILITY.md` — the command and the environment for the saved run
- `docs/manuscript_draft.md` — prose filled from `results/metrics.json`
- `results/` — metrics, tables, and figures

## Run

```bash
pip install -r requirements.txt
python3 scripts/prepare_data.py --max-train 32 --max-val 12 --max-id-test 12 --seed 42
python3 scripts/run_experiments.py --epochs 3 --batch-size 16 --image-size 224 --seed 42
python3 -m pytest
```

`--max-train 0` removes the per-class cap. That reads the aligned color images from the PlantVillage zip already downloaded by the prepare step. It does not change the group rule.

PlantVillage color images and the leaf map come from the Hugging Face dataset `mohanty/PlantVillage`. PlantDoc images come from `geraldmc/plantdoc-full`. Image files are written under `data/prepared/` and are gitignored.
