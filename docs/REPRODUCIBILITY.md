# Reproducibility

The saved run is one seed on one machine. Repeating it should land close to the same splits, because the split uses seed 42, and close to the same training curves when the data loader uses zero worker processes. CPU kernels are not guaranteed bit-identical across machines.

## Commands

From the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/prepare_data.py --max-train 32 --max-val 12 --max-id-test 12 --seed 42
python3 scripts/run_experiments.py --epochs 3 --batch-size 16 --image-size 224 --seed 42
python3 -m pytest
```

`run_experiments.py` writes `results/metrics.json`, `results/summary.csv`, `results/history.csv`, `results/editing_rows.csv`, `results/figures/`, and `docs/manuscript_draft.md`. The manuscript is generated from the metrics file. Editing it by hand will make it disagree with the JSON.

## Data

- PlantVillage: Hugging Face `mohanty/PlantVillage`, color lists and `data.zip`. The prepare script records the resolved zip path in `data/prepared/split_audit.json`.
- PlantDoc: Hugging Face `geraldmc/plantdoc-full`.
- Prepared images are local and gitignored.

## Training choices

- ResNet-50 and EfficientNet-B0, ImageNet-1K V1 initialization from torchvision
- AdamW, learning rate `1e-4`, weight decay `1e-4`
- 3 epochs, batch size 16, image size 224
- Checkpoint chosen by validation macro-F1
- Runs: `none`, `weak`, and `strong` augmentation for both backbones, plus ResNet-50 with the lesion-preserving edit applied to half of the training images on top of the weak crop and flip
- Device in the saved run: CPU, torch `2.15.0+cpu`, torchvision `0.29.2+cpu`, Python `3.12.3`
- Wall clock recorded in `results/metrics.json`: 1317 seconds
- Other packages in that environment: numpy 2.4.4, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.1.1, Pillow 13.0.0, datasets 5.1.0

## Tests

`python3 -m pytest` does not download datasets. It checks group identity for copy filenames, that the same stem in two classes is not merged, that group splits do not share a group, that the preserving edit copies lesion pixels, that the destroying control changes only those pixels, and that one CPU training step runs.
