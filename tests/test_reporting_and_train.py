import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ppce.engine import classification_scores, fit_model, majority_baseline
from ppce.manuscript import render_manuscript
from ppce.mapping import ALIGNED, assert_alignment_is_bijective
from ppce.models import build_model


def test_alignment_is_bijective():
    assert_alignment_is_bijective()
    assert len(ALIGNED) == 27


def test_majority_baseline_breaks_ties_toward_smaller_label():
    train = np.array([1, 0, 1, 0])
    evaluation = np.array([0, 0, 1])
    result = majority_baseline(train, evaluation)
    assert result["majority_label"] == 0
    assert result["accuracy"] == 2 / 3


def test_manuscript_copies_metrics():
    payload = {
        "config": {
            "epochs": 1,
            "lr": 0.0001,
            "weight_decay": 0.0001,
            "image_size": 224,
            "batch_size": 8,
            "seed": 42,
            "ppce_probability": 0.5,
        },
        "audit": {
            "n_aligned_classes": 27,
            "official_split_groups_on_both_sides": 12,
            "official_split_groups_on_both_sides_aligned": 9,
            "n_train": 10,
            "n_val": 4,
            "n_id_test": 4,
            "n_plantdoc": 6,
            "our_split_group_collisions": 0,
        },
        "majority_baseline": {"id_test_accuracy": 0.25, "plantdoc_accuracy": 0.125},
        "runs": [
            {
                "name": "resnet50-none",
                "best_val_macro_f1": 0.4,
                "id_test": {"accuracy": 0.5, "macro_f1": 0.45},
                "plantdoc": {"accuracy": 0.2, "macro_f1": 0.15},
            }
        ],
        "editing": {
            "summary": {
                "n_images": 4,
                "n_segmented": 3,
                "mean_ppce_lesion_pixel_match": 1.0,
                "mean_destroy_lesion_pixel_match": 0.0,
                "mean_ppce_outside_mae": 12.5,
                "mean_destroy_outside_mae": 1.5,
                "mean_abs_delta_ppce": 0.01,
                "mean_abs_delta_destroy": 0.2,
                "fraction_destroy_moves_prob_more": 0.6666667,
            }
        },
    }
    text = render_manuscript(payload)
    assert "resnet50-none" in text
    assert "0.500" in text
    assert "0.200" in text
    assert "12 image groups" in text
    assert "state of the art" in text
    assert "moved the true-class probability more" in text


def test_one_training_step_selects_a_checkpoint():
    torch.manual_seed(0)
    images = torch.rand(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    model = build_model("resnet50", num_classes=2, pretrained=False)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    _, history = fit_model(
        model,
        loader,
        loader,
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        device=torch.device("cpu"),
        num_classes=2,
    )
    assert len(history) == 1
    assert history[0].train_loss > 0
    scores = classification_scores(np.array([0, 1]), np.array([0, 1]), 2)
    assert scores["accuracy"] == 1.0
