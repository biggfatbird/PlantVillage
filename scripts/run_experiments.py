#!/usr/bin/env python3
"""Train the baselines and write metrics, figures, and the manuscript draft."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ppce.datasets import ManifestDataset, load_manifest
from ppce.editing_study import run_editing_study
from ppce.engine import classification_scores, fit_model, majority_baseline, predict, set_seed
from ppce.manuscript import write_manuscript
from ppce.mapping import CANONICAL_LABELS
from ppce.models import build_model
from ppce.plots import plot_domain_shift, plot_edit_panel, plot_history


RUNS = (
    ("resnet50", "none", 0.0),
    ("resnet50", "weak", 0.0),
    ("resnet50", "strong", 0.0),
    ("resnet50", "ppce", 0.5),
    ("efficientnet_b0", "none", 0.0),
    ("efficientnet_b0", "weak", 0.0),
    ("efficientnet_b0", "strong", 0.0),
)


def _loader(frame: pd.DataFrame, policy: str, train: bool, args, ppce_probability: float) -> DataLoader:
    dataset = ManifestDataset(
        frame,
        policy=policy,
        train=train,
        image_size=args.image_size,
        ppce_probability=ppce_probability,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.workers,
        pin_memory=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/prepared/manifest.csv")
    parser.add_argument("--audit", default="data/prepared/split_audit.json")
    parser.add_argument("--out", default="results")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ppce-probability", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--editing-images", type=int, default=36)
    parser.add_argument("--runs", nargs="*", default=None, help="Optional subset, as model:augmentation")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    manifest = load_manifest(args.manifest)
    train_frame = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val_frame = manifest[manifest["split"] == "val"].reset_index(drop=True)
    test_frame = manifest[manifest["split"] == "id_test"].reset_index(drop=True)
    doc_frame = manifest[manifest["split"] == "plantdoc"].reset_index(drop=True)
    num_classes = len(CANONICAL_LABELS)
    selected = set(args.runs) if args.runs else None

    history_rows = []
    run_payloads = []
    summary_rows = []
    best_name = None
    best_score = float("-inf")
    best_arch = None
    best_state = None
    started = time.time()

    for model_name, augmentation, default_ppce in RUNS:
        run_name = f"{model_name}-{augmentation}"
        if selected and run_name not in selected and f"{model_name}:{augmentation}" not in selected:
            continue
        print(f"training {run_name}", flush=True)
        set_seed(args.seed)
        ppce_probability = args.ppce_probability if augmentation == "ppce" else default_ppce
        model = build_model(model_name, num_classes, pretrained=True)
        train_loader = _loader(train_frame, "weak" if augmentation == "ppce" else augmentation, True, args, ppce_probability)
        val_loader = _loader(val_frame, "none", False, args, 0.0)
        run_started = time.time()
        state, history = fit_model(
            model,
            train_loader,
            val_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            num_classes=num_classes,
        )
        model.load_state_dict(state)
        model.to(device)
        id_labels, id_preds, _ = predict(model, _loader(test_frame, "none", False, args, 0.0), device)
        doc_labels, doc_preds, _ = predict(model, _loader(doc_frame, "none", False, args, 0.0), device)
        id_scores = classification_scores(id_labels, id_preds, num_classes)
        doc_scores = classification_scores(doc_labels, doc_preds, num_classes)
        best_val = max(item.val_macro_f1 for item in history)
        payload = {
            "name": run_name,
            "model": model_name,
            "augmentation": augmentation,
            "seconds": round(time.time() - run_started, 3),
            "best_val_macro_f1": best_val,
            "history": [item.__dict__ for item in history],
            "id_test": id_scores,
            "plantdoc": doc_scores,
        }
        run_payloads.append(payload)
        summary_rows.append(
            {
                "run": run_name,
                "model": model_name,
                "augmentation": augmentation,
                "id_test_accuracy": id_scores["accuracy"],
                "plantdoc_accuracy": doc_scores["accuracy"],
                "id_test_macro_f1": id_scores["macro_f1"],
                "plantdoc_macro_f1": doc_scores["macro_f1"],
            }
        )
        for item in history:
            history_rows.append({"run": run_name, **item.__dict__})
        rank = (best_val, run_name)
        if best_name is None or rank > (best_score, best_name):
            best_score = best_val
            best_name = run_name
            best_arch = model_name
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"done {run_name} val_f1={best_val:.3f} id={id_scores['accuracy']:.3f} pd={doc_scores['accuracy']:.3f}",
            flush=True,
        )

    if best_state is None or best_arch is None or best_name is None:
        raise SystemExit("no runs selected")
    best_model = build_model(best_arch, num_classes, pretrained=False)
    best_model.load_state_dict(best_state)
    best_model.to(device)
    print(f"editing study with {best_name}", flush=True)
    editing = run_editing_study(
        best_model,
        test_frame,
        device=device,
        image_size=args.image_size,
        max_images=args.editing_images,
        seed=args.seed,
    )
    example = editing.pop("example")

    train_labels = train_frame["label_index"].to_numpy()
    majority = {
        "id_test_accuracy": majority_baseline(train_labels, test_frame["label_index"].to_numpy())["accuracy"],
        "plantdoc_accuracy": majority_baseline(train_labels, doc_frame["label_index"].to_numpy())["accuracy"],
        "train_majority_label": int(majority_baseline(train_labels, train_labels)["majority_label"]),
    }
    audit = json.loads(Path(args.audit).read_text())
    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "ppce_probability": args.ppce_probability,
        "seed": args.seed,
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "selected_editing_model": best_name,
        "seconds_total": round(time.time() - started, 3),
    }
    payload = {
        "config": config,
        "audit": audit,
        "majority_baseline": majority,
        "runs": run_payloads,
        "editing": {"summary": editing["summary"], "model": best_name},
    }
    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2))
    pd.DataFrame(history_rows).to_csv(out / "history.csv", index=False)
    pd.DataFrame(editing["rows"]).to_csv(out / "editing_rows.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "summary.csv", index=False)
    plot_domain_shift(summary, out / "figures" / "domain_shift.png")
    plot_history(pd.DataFrame(history_rows), out / "figures" / "val_macro_f1.png")
    if example is not None:
        plot_edit_panel(example, out / "figures" / "editing_panel.png")
    manuscript = write_manuscript(metrics_path, Path("docs/manuscript_draft.md"))
    print(f"wrote {metrics_path} and docs/manuscript_draft.md ({len(manuscript)} chars)", flush=True)


if __name__ == "__main__":
    main()
