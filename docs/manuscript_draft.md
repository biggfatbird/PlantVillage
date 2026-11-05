# PlantVillage to PlantDoc: leakage-safe CNN baselines and a lesion-preserving edit prototype

Working project name: PPCE. In this repository that name means the lesion-preserving counterfactual editing prototype below. It is not a claim that a method by this name is already established for this dataset pair, and it is not a claim of a new state of the art.

## What this draft is

This draft reports one compute-limited run. The numbers below are copied from `results/metrics.json` by `ppce/manuscript.py`. They are in-domain and cross-dataset measurements for the saved configuration. They are not a multi-seed confidence interval, and they are not a full pass over every PlantVillage image.

## Data

PlantVillage color images come from the Hugging Face release `mohanty/PlantVillage` (snapshot recorded in the audit). PlantDoc images come from `geraldmc/plantdoc-full`, which preserves the upstream folder names. The aligned label space has 27 classes. Unaligned PlantVillage and PlantDoc classes were excluded rather than guessed. One included pair is explicitly an assumption: PlantDoc `Corn leaf blight` is scored as PlantVillage northern leaf blight.

The published color train and test file lists place 227 image groups from the full color release on both sides (52 of those groups are in the aligned classes). The colliding files are filename-stem copies, including files marked `copy`, whose stems are missing from the leaf map. This run does not use that published split. Images linked by the same filename stem or the same leaf-map id form one group, and each group is assigned to train, validation, or the in-domain test. After that assignment, a per-class cap keeps whole groups.

Prepared counts: train 858, validation 308, in-domain test 304, PlantDoc external test 2507. Group collisions remaining across our three PlantVillage splits: 0.

## Models and protocol

Backbones are ImageNet-initialized ResNet-50 and EfficientNet-B0. The training configuration is 3 epoch(s), AdamW learning rate 0.0001, weight decay 0.0001, image size 224, batch size 16, seed 42. Augmentations are applied only to training images. `none` is resize and center crop. `weak` adds a random resized crop and horizontal flip. `strong` adds color jitter. `ppce` is the weak policy plus the lesion-preserving edit on a training image with probability 0.5, when the heuristic finds a lesion. The checkpoint for each run is the epoch with the highest validation macro-F1. PlantDoc is not used for that choice.

Normalization uses ImageNet mean and standard deviation. A majority-class baseline predicts the most frequent training label.

## Classification results

| Run | Val macro-F1 | In-domain accuracy | In-domain macro-F1 | PlantDoc accuracy | PlantDoc macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| resnet50-none | 0.900 | 0.921 | 0.925 | 0.203 | 0.153 |
| resnet50-weak | 0.874 | 0.908 | 0.910 | 0.227 | 0.169 |
| resnet50-strong | 0.845 | 0.911 | 0.917 | 0.223 | 0.174 |
| resnet50-ppce | 0.864 | 0.921 | 0.921 | 0.221 | 0.162 |
| efficientnet_b0-none | 0.813 | 0.891 | 0.891 | 0.181 | 0.138 |
| efficientnet_b0-weak | 0.822 | 0.875 | 0.873 | 0.204 | 0.161 |
| efficientnet_b0-strong | 0.796 | 0.849 | 0.850 | 0.227 | 0.171 |

Majority-class accuracy is 0.039 on the in-domain test and 0.037 on PlantDoc. Uniform chance accuracy for 27 classes is 0.037.

The gap between the in-domain test and PlantDoc is the domain-shift measurement for this subset and this training budget. A higher in-domain number than PlantDoc number is consistent with the known laboratory-to-field shift. It does not, by itself, identify which visual factor moved.

## Lesion-preserving edit

The editor finds green tissue with the excess-green index, fills holes inside that tissue, and treats the non-green interior as the lesion. The preserving edit replaces the background with a soil-colored field, shifts the color of non-lesion leaf pixels, and then writes the original lesion pixels back. By construction, lesion pixels match the original whenever a lesion mask exists. The control paints those same pixels with the median color of the other leaf pixels and does not change the rest of the image.

On 36 of 36 sampled diseased in-domain test images, the heuristic returned a lesion. Mean lesion-pixel match was 1.000 for the preserving edit and 0.000 for the destroying control. Mean absolute color change outside the lesion was 33.168 for the preserving edit and 0.000 for the control. Mean absolute change in the true-class probability was 0.223 after the preserving edit and 0.140 after the control. The control moved that probability more often on 38.9% of the segmented images. In this run the preserving edit therefore moved the true-class probability more, on average, than the control. That does not support a claim that the classifier is using only the pixels inside this mask. The saved panel is the first sampled image. On that corn gray-leaf-spot leaf the mask marks pale tissue, including the midrib, and it does not cover every brown spot.

These probability changes are a diagnostic of one trained model on one heuristic mask. They are not a causal identification result. The mask can include veins, shadows, or only part of a symptom, and it can miss a lesion. Empty masks were excluded from the probability comparison rather than scored as successes.

## What this run does not show

- It does not show that either backbone is best on the full PlantVillage test set. Published full-data numbers were not recreated.
- It does not show that the preserving edit improves field accuracy, except for the single `ppce` training row above. That row is one seed and one cap.
- It does not show that the corn-blight label assumption is correct.
- It does not use a learned inpainter, a diffusion model, or a lesion annotation. No such annotation is in these releases.

## Sources

Mohanty, Hughes, and Salathé, 2016, Frontiers in Plant Science, image-based plant disease detection. Singh et al., 2020, PlantDoc. He et al., 2016, ResNet. Tan and Le, 2019, EfficientNet. The PlantVillage files used here are the color split of `mohanty/PlantVillage`. The PlantDoc files used here are `geraldmc/plantdoc-full`.
