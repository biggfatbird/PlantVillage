"""Fill the manuscript draft from saved metrics.

The prose around the numbers is fixed. Every numeric result is read from the
metrics payload so the draft cannot drift from ``results/metrics.json``.
"""

from __future__ import annotations

import json
from pathlib import Path


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return "NA"
    return f"{number:.{digits}f}"


def _pct(value) -> str:
    text = _fmt(value, 3)
    if text == "NA":
        return text
    return f"{float(value) * 100:.1f}%"


def render_manuscript(payload: dict) -> str:
    runs = payload["runs"]
    audit = payload["audit"]
    editing = payload["editing"]["summary"]
    config = payload["config"]
    lines = [
        "# PlantVillage to PlantDoc: leakage-safe CNN baselines and a lesion-preserving edit prototype",
        "",
        "Working project name: PPCE. In this repository that name means the lesion-preserving counterfactual editing prototype below. It is not a claim that a method by this name is already established for this dataset pair, and it is not a claim of a new state of the art.",
        "",
        "## What this draft is",
        "",
        "This draft reports one compute-limited run. The numbers below are copied from `results/metrics.json` by `ppce/manuscript.py`. They are in-domain and cross-dataset measurements for the saved configuration. They are not a multi-seed confidence interval, and they are not a full pass over every PlantVillage image.",
        "",
        "## Data",
        "",
        f"PlantVillage color images come from the Hugging Face release `mohanty/PlantVillage` (snapshot recorded in the audit). PlantDoc images come from `geraldmc/plantdoc-full`, which preserves the upstream folder names. The aligned label space has {audit['n_aligned_classes']} classes. Unaligned PlantVillage and PlantDoc classes were excluded rather than guessed. One included pair is explicitly an assumption: PlantDoc `Corn leaf blight` is scored as PlantVillage northern leaf blight.",
        "",
        f"The published color train and test file lists place {audit['official_split_groups_on_both_sides']} image groups from the full color release on both sides ({audit['official_split_groups_on_both_sides_aligned']} of those groups are in the aligned classes). The colliding files are filename-stem copies, including files marked `copy`, whose stems are missing from the leaf map. This run does not use that published split. Images linked by the same filename stem or the same leaf-map id form one group, and each group is assigned to train, validation, or the in-domain test. After that assignment, a per-class cap keeps whole groups.",
        "",
        f"Prepared counts: train {audit['n_train']}, validation {audit['n_val']}, in-domain test {audit['n_id_test']}, PlantDoc external test {audit['n_plantdoc']}. Group collisions remaining across our three PlantVillage splits: {audit['our_split_group_collisions']}.",
        "",
        "## Models and protocol",
        "",
        f"Backbones are ImageNet-initialized ResNet-50 and EfficientNet-B0. The training configuration is {config['epochs']} epoch(s), AdamW learning rate {config['lr']}, weight decay {config['weight_decay']}, image size {config['image_size']}, batch size {config['batch_size']}, seed {config['seed']}. Augmentations are applied only to training images. `none` is resize and center crop. `weak` adds a random resized crop and horizontal flip. `strong` adds color jitter. `ppce` is the weak policy plus the lesion-preserving edit on a training image with probability {config['ppce_probability']}, when the heuristic finds a lesion. The checkpoint for each run is the epoch with the highest validation macro-F1. PlantDoc is not used for that choice.",
        "",
        "Normalization uses ImageNet mean and standard deviation. A majority-class baseline predicts the most frequent training label.",
        "",
        "## Classification results",
        "",
        "| Run | Val macro-F1 | In-domain accuracy | In-domain macro-F1 | PlantDoc accuracy | PlantDoc macro-F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        lines.append(
            "| {name} | {val} | {ida} | {idf} | {pda} | {pdf} |".format(
                name=run["name"],
                val=_fmt(run["best_val_macro_f1"]),
                ida=_fmt(run["id_test"]["accuracy"]),
                idf=_fmt(run["id_test"]["macro_f1"]),
                pda=_fmt(run["plantdoc"]["accuracy"]),
                pdf=_fmt(run["plantdoc"]["macro_f1"]),
            )
        )
    majority = payload["majority_baseline"]
    lines.extend(
        [
            "",
            f"Majority-class accuracy is {_fmt(majority['id_test_accuracy'])} on the in-domain test and {_fmt(majority['plantdoc_accuracy'])} on PlantDoc. Uniform chance accuracy for {audit['n_aligned_classes']} classes is {_fmt(1 / audit['n_aligned_classes'])}.",
            "",
            "The gap between the in-domain test and PlantDoc is the domain-shift measurement for this subset and this training budget. A higher in-domain number than PlantDoc number is consistent with the known laboratory-to-field shift. It does not, by itself, identify which visual factor moved.",
            "",
            "## Lesion-preserving edit",
            "",
            "The editor finds green tissue with the excess-green index, fills holes inside that tissue, and treats the non-green interior as the lesion. The preserving edit replaces the background with a soil-colored field, shifts the color of non-lesion leaf pixels, and then writes the original lesion pixels back. By construction, lesion pixels match the original whenever a lesion mask exists. The control paints those same pixels with the median color of the other leaf pixels and does not change the rest of the image.",
            "",
            f"On {editing['n_segmented']} of {editing['n_images']} sampled diseased in-domain test images, the heuristic returned a lesion. Mean lesion-pixel match was {_fmt(editing['mean_ppce_lesion_pixel_match'])} for the preserving edit and {_fmt(editing['mean_destroy_lesion_pixel_match'])} for the destroying control. Mean absolute color change outside the lesion was {_fmt(editing['mean_ppce_outside_mae'])} for the preserving edit and {_fmt(editing['mean_destroy_outside_mae'])} for the control. Mean absolute change in the true-class probability was {_fmt(editing['mean_abs_delta_ppce'])} after the preserving edit and {_fmt(editing['mean_abs_delta_destroy'])} after the control. The control moved that probability more often on {_pct(editing['fraction_destroy_moves_prob_more'])} of the segmented images. In this run the preserving edit therefore moved the true-class probability more, on average, than the control. That does not support a claim that the classifier is using only the pixels inside this mask. The saved panel is the first sampled image. On that corn gray-leaf-spot leaf the mask marks pale tissue, including the midrib, and it does not cover every brown spot.",
            "",
            "These probability changes are a diagnostic of one trained model on one heuristic mask. They are not a causal identification result. The mask can include veins, shadows, or only part of a symptom, and it can miss a lesion. Empty masks were excluded from the probability comparison rather than scored as successes.",
            "",
            "## What this run does not show",
            "",
            "- It does not show that either backbone is best on the full PlantVillage test set. Published full-data numbers were not recreated.",
            "- It does not show that the preserving edit improves field accuracy, except for the single `ppce` training row above. That row is one seed and one cap.",
            "- It does not show that the corn-blight label assumption is correct.",
            "- It does not use a learned inpainter, a diffusion model, or a lesion annotation. No such annotation is in these releases.",
            "",
            "## Sources",
            "",
            "Mohanty, Hughes, and Salathé, 2016, Frontiers in Plant Science, image-based plant disease detection. Singh et al., 2020, PlantDoc. He et al., 2016, ResNet. Tan and Le, 2019, EfficientNet. The PlantVillage files used here are the color split of `mohanty/PlantVillage`. The PlantDoc files used here are `geraldmc/plantdoc-full`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_manuscript(metrics_path: Path, destination: Path) -> str:
    payload = json.loads(metrics_path.read_text())
    text = render_manuscript(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)
    return text
