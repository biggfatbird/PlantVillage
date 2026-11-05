# Audit

## Repository before this work

The git history on `main` contains a README, a Cloud Agent install script, and one deleted Colab notebook named `GSsynthesis`. That notebook loads two Excel files of dye-adsorption measurements, drops incomplete rows, and fits regressors with a grouped train/test split. It is not a plant-image experiment. It was not used as the method, the data, or the result of this project. No PlantVillage images, PlantDoc images, model code, or written PPCE specification were in the tree.

## Datasets that were available

PlantVillage color files were read from `mohanty/PlantVillage` (`data.zip`, `splits/color_train.txt`, `splits/color_test.txt`, and `leaf_grouping/leaf-map.json`). The color lists contain 54,305 image paths. PlantDoc files were read from `geraldmc/plantdoc-full`. That release has one streaming split and a `split` column on each row. This machine has no GPU. Training used CPU.

## Label alignment

Training and scoring use 27 classes whose PlantVillage and PlantDoc names refer to the same disease or to a healthy leaf. `Pepper,_bell___Bacterial_spot` is not paired with PlantDoc `Bell_pepper leaf spot`, because the PlantDoc name does not say bacterial spot. That class is counted in the audit and then left out. PlantDoc `Corn leaf blight` is paired with northern leaf blight. That pair is an assumption and is marked as one in `ppce/mapping.py`.

The loaded PlantDoc release contains 2,507 images in the aligned classes and 71 `Bell_pepper leaf spot` images that were not used. `Tomato two spotted spider mites leaf` contributes 2 images.

## Why the published color split was not reused

Filename stems are shared by a file and its `copy` duplicates. Stems are compared inside a class only. The same stem text is reused across crops, and a global match would chain unrelated leaves.

On the full color lists, 227 class-qualified groups contain at least one path from `color_train.txt` and one path from `color_test.txt`. 52 of those groups are in the aligned classes. The largest colliding group has 5 images. Examples are corn gray-leaf-spot stems such as `rs_glsp 4355`, where one file is in the published train list and a `copy` file is in the published test list. Non-fallback leaf-map ids were not themselves on both sides; the leaks are stems the leaf map does not contain.

This project builds groups from two links: the class-qualified stem, and the leaf-map id when the stem is in the map. Each group goes entirely to train, validation, or the in-domain test (70/15/15 by group, seed 42, stratified by class). The prepared manifest has 0 groups in more than one of those splits. Caps of 32, 12, and 12 images per class then drop whole groups. The prepared counts are 858 train, 308 validation, 304 in-domain test, and 2,507 PlantDoc.

ImageNet mean and standard deviation are used because the backbones start from ImageNet weights. Those statistics are not computed from the test images. Augmentation is applied only in training. The saved checkpoint for each run is the epoch with the highest validation macro-F1. PlantDoc is not used to choose it.
