"""Training and evaluation loops.

The checkpoint written for downstream evaluation is the epoch with the best
validation macro-F1. In-domain test and PlantDoc are not used to choose it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_accuracy: float
    val_macro_f1: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[float, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    seen = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        seen += int(labels.size(0))
    if seen == 0:
        return 0.0, 0.0
    return total_loss / seen, correct / seen


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels_out: list[np.ndarray] = []
    preds_out: list[np.ndarray] = []
    probs_out: list[np.ndarray] = []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        preds_out.append(probabilities.argmax(axis=1))
        probs_out.append(probabilities)
        labels_out.append(labels.numpy())
    if not labels_out:
        empty = np.zeros((0,), dtype=np.int64)
        return empty, empty, np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(labels_out), np.concatenate(preds_out), np.concatenate(probs_out)


def classification_scores(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> dict[str, float]:
    if len(labels) == 0:
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "n": 0}
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=list(range(num_classes)), average="macro", zero_division=0)),
        "n": int(len(labels)),
    }


def majority_baseline(train_labels: np.ndarray, eval_labels: np.ndarray) -> dict[str, float]:
    if len(train_labels) == 0 or len(eval_labels) == 0:
        return {"majority_label": -1, "accuracy": float("nan")}
    values, counts = np.unique(train_labels, return_counts=True)
    # Ties break toward the smaller label so the baseline is deterministic.
    order = np.lexsort((values, -counts))
    majority = int(values[order[0]])
    accuracy = float(np.mean(eval_labels == majority))
    return {"majority_label": majority, "accuracy": accuracy}


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    num_classes: int,
) -> tuple[dict[str, torch.Tensor], list[EpochLog]]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_f1 = -1.0
    history: list[EpochLog] = []
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, device)
        val_labels, val_preds, _ = predict(model, val_loader, device)
        val_scores = classification_scores(val_labels, val_preds, num_classes)
        history.append(
            EpochLog(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_accuracy=val_scores["accuracy"],
                val_macro_f1=val_scores["macro_f1"],
            )
        )
        if val_scores["macro_f1"] > best_f1:
            best_f1 = val_scores["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best_state, history
