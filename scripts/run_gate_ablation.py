#!/usr/bin/env python3
"""Train the three new edit policies. The no-augmentation and standard-augmentation
numbers stay in results/metrics.json and are not retrained.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppce.datasets import ManifestDataset, load_manifest
from ppce.engine import classification_scores, fit_model, predict, set_seed
from ppce.mapping import CANONICAL_LABELS
from ppce.models import build_model


RUNS = ("unconstrained", "constrained", "gated")


def _loader(frame, policy: str, train: bool, batch_size: int, image_size: int) -> DataLoader:
    dataset = ManifestDataset(frame, policy=policy if train else "none", train=train, image_size=image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=0)


def _bootstrap(labels: np.ndarray, predictions: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    scores = []
    count = len(labels)
    for _ in range(1000):
        chosen = rng.integers(0, count, count)
        scores.append(float(np.mean(labels[chosen] == predictions[chosen])))
    array = np.asarray(scores)
    return {"accuracy_ci95_low": float(np.quantile(array, 0.025)), "accuracy_ci95_high": float(np.quantile(array, 0.975))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/prepared/manifest.csv")
    parser.add_argument("--out", default="results/ablation_gate")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    manifest = load_manifest(args.manifest)
    train_frame = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val_frame = manifest[manifest["split"] == "val"].reset_index(drop=True)
    test_frame = manifest[manifest["split"] == "id_test"].reset_index(drop=True)
    doc_frame = manifest[manifest["split"] == "plantdoc"].reset_index(drop=True)
    num_classes = len(CANONICAL_LABELS)
    payloads = []
    for policy in RUNS:
        print(f"training resnet50-{policy}", flush=True)
        set_seed(args.seed)
        model = build_model("resnet50", num_classes, pretrained=True)
        started = time.time()
        state, history = fit_model(
            model,
            _loader(train_frame, policy, True, args.batch_size, args.image_size),
            _loader(val_frame, policy, False, args.batch_size, args.image_size),
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            num_classes=num_classes,
        )
        model.load_state_dict(state)
        model.to(device)
        id_labels, id_preds, _ = predict(model, _loader(test_frame, policy, False, args.batch_size, args.image_size), device)
        doc_labels, doc_preds, _ = predict(model, _loader(doc_frame, policy, False, args.batch_size, args.image_size), device)
        id_scores = classification_scores(id_labels, id_preds, num_classes)
        doc_scores = classification_scores(doc_labels, doc_preds, num_classes)
        payload = {
            "name": f"resnet50-{policy}",
            "policy": policy,
            "seconds": round(time.time() - started, 3),
            "best_val_macro_f1": max(item.val_macro_f1 for item in history),
            "history": [item.__dict__ for item in history],
            "id_test": id_scores,
            "plantdoc": doc_scores,
            "id_test_accuracy_bootstrap": _bootstrap(id_labels, id_preds, args.seed),
            "plantdoc_accuracy_bootstrap": _bootstrap(doc_labels, doc_preds, args.seed),
        }
        payloads.append(payload)
        print(
            f"done {policy} val={payload['best_val_macro_f1']:.3f} id={id_scores['accuracy']:.3f} pd={doc_scores['accuracy']:.3f}",
            flush=True,
        )
    (out / "metrics.json").write_text(json.dumps({"runs": payloads, "seed": args.seed, "epochs": args.epochs}, indent=2))


if __name__ == "__main__":
    main()
