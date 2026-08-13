"""Evaluate both trained models and produce every result artefact.

Runs the full comparison from saved checkpoints - no retraining. This is the
script that generates the numbers and figures quoted in the report.

Usage
-----
    python scripts/evaluate.py
    python scripts/evaluate.py --split test --no-figures

Pipeline
--------
 1. Load both checkpoints.
 2. Select each model's confidence threshold on the **validation** split
    (never on test - see `metrics.select_confidence_threshold`).
 3. Evaluate both on the test split at their frozen thresholds.
 4. Benchmark latency under an identical protocol.
 5. Classify every error and aggregate the failure taxonomy.
 6. Write results/comparison.md, results/metrics.json and results/figures/*.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.config import (  # noqa: E402
    CHECKPOINT_DIR, DATA_DIR, FIGURES_DIR, RESULTS_DIR, Config, ensure_directories,
)
from src.data import DetectionDataset, GridDataset, load_ground_truth  # noqa: E402
from src.engine import (  # noqa: E402
    benchmark_latency, load_checkpoint, predict_split, update_checkpoint_threshold,
)
from src.failure_analysis import (  # noqa: E402
    analyse_failures, failure_markdown_table, failure_summary_text,
)
from src.metrics import evaluate_detections, select_confidence_threshold  # noqa: E402
from src.utils import (  # noqa: E402
    count_parameters, file_size_mb, format_hardware, hardware_report,
    load_json, model_size_mb, save_json, set_seed,
)
from src import viz  # noqa: E402

MODEL_LABELS = {
    "scratch": "Model 1 (Scratch Grid)",
    "fasterrcnn": "Model 2 (Faster R-CNN)",
}


def dataset_for(kind: str, data_root, split: str, cfg: Config):
    ds_cls = GridDataset if kind == "scratch" else DetectionDataset
    return ds_cls(data_root, split, cfg, augment=False)


def build_comparison_markdown(rows, results, analyses, thresholds, hardware) -> str:
    """The direct comparison table the guide asks for."""
    lines = [
        "# Model Comparison - Aquarium Object Detection",
        "",
        f"Hardware: `{format_hardware(hardware)}`  ",
        "All figures below come from `scripts/evaluate.py` run on the **test** split, "
        "which was untouched during training and threshold selection.",
        "",
        "## Quality and efficiency",
        "",
        "| Metric | " + " | ".join(r["model"] for r in rows) + " |",
        "|---|" + "---|" * len(rows),
    ]

    def row(label, key, fmt="{:.3f}"):
        cells = []
        for r in rows:
            try:
                cells.append(fmt.format(r[key]))
            except (ValueError, TypeError):
                cells.append(str(r[key]))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    row("**mAP@0.5**", "map50")
    row("mAP@[.5:.95]", "map50_95")
    row("Precision", "precision")
    row("Recall", "recall")
    row("F1", "f1")
    row("True positives", "tp", "{}")
    row("False positives", "fp", "{}")
    row("Missed (FN)", "fn", "{}")
    row("Confidence threshold", "conf_threshold", "{:.2f}")
    row("Latency, median (ms)", "latency_ms", "{:.1f}")
    row("Latency, p95 (ms)", "latency_p95_ms", "{:.1f}")
    row("Throughput (FPS)", "fps", "{:.1f}")
    row("Parameters", "params", "{:,}")
    row("Model size (MB)", "size_mb", "{:.1f}")
    row("Checkpoint on disk (MB)", "checkpoint_mb", "{:.1f}")
    row("Training time (s)", "train_seconds", "{:.0f}")
    row("Best epoch", "best_epoch", "{}")

    lines += ["", "## Per-class AP@0.5", "",
              "| Class | objects | " + " | ".join(r["model"] for r in rows) + " |",
              "|---|---|" + "---|" * len(rows)]

    first = next(iter(results.values()))
    for cls in first.class_names:
        support = first.per_class[cls]["n_gt"]
        cells = []
        for r in rows:
            ap = results[r["model"]].per_class[cls]["ap"]
            cells.append("n/a" if ap != ap else f"{ap:.3f}")   # ap != ap detects NaN
        lines.append(f"| {cls} | {support} | " + " | ".join(cells) + " |")

    lines += ["", "## Failure analysis", ""]
    lines.append(failure_markdown_table(analyses, first.class_names))
    lines += [""]
    for name, analysis in analyses.items():
        lines.append(f"- {failure_summary_text(name, analysis)}")

    lines += [
        "",
        "## Conditions that could not be kept identical",
        "",
        "| Condition | Model 1 | Model 2 | Why it differs |",
        "|---|---|---|---|",
        "| Optimiser | Adam | SGD + momentum | Adam converges faster from random init; "
        "SGD is the reference recipe for fine-tuning pretrained detection backbones and "
        "disturbs them less. |",
        "| Learning rate | 1e-3 | 5e-3 | Tuned per optimiser; the two values are not "
        "comparable across optimisers. |",
        "| LR schedule | cosine annealing | step decay | Matched to each optimiser's "
        "convention. |",
        "| Batch size | 16 | 8 | Faster R-CNN's activations are far larger; 8 is the "
        "biggest batch that fits a 16 GB T4 at 224px. |",
        "| Epoch budget | 80 (max) | 20 (max) | A model trained from scratch needs more "
        "passes than one starting from COCO features. Both are governed by the *same* "
        "early-stopping rule (patience 12 on validation mAP). |",
        "| Initialisation | random (He) | COCO-pretrained | This is the comparison itself. |",
        "",
        "Everything else is shared verbatim: dataset, splits, resize, augmentation policy, "
        "normalisation statistics, NMS implementation and IoU, evaluation code, "
        "threshold-selection procedure, and timing protocol.",
        "",
        "### Did either model finish converging?",
        "",
    ]

    # State honestly whether each run stopped early or simply ran out of budget.
    # A best epoch equal to the last epoch run means the model was still
    # improving when training ended - which caps what the comparison can claim.
    for r in rows:
        best, ran = r["best_epoch"], r["epochs_run"]
        if isinstance(best, int) and isinstance(ran, int) and ran > 0:
            if best >= ran:
                lines.append(
                    f"- **{r['model']}**: best epoch was **{best} of {ran}** - the last "
                    f"epoch run. Early stopping never triggered, so this model was **still "
                    f"improving when the budget ran out**. Its score here is a lower bound, "
                    f"not a converged result."
                )
            else:
                lines.append(
                    f"- **{r['model']}**: best epoch {best} of {ran}, with no improvement "
                    f"over the following {ran - best} epochs. This model had converged; the "
                    f"saved weights are from the validation peak, not the final epoch."
                )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--checkpoints", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-latency", action="store_true",
                        help="skip the latency benchmark (much faster on CPU)")
    args = parser.parse_args()

    ensure_directories()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hardware = hardware_report()

    print("=" * 74)
    print(f"Evaluating on the '{args.split}' split")
    print(f"  hardware: {format_hardware(hardware)}")
    print("=" * 74)

    cfg_ref = Config()
    set_seed(cfg_ref.seed, cfg_ref.deterministic)

    gt_valid = load_ground_truth(args.data, "valid", cfg_ref)
    gt_test = load_ground_truth(args.data, args.split, cfg_ref)

    rows, results, analyses, thresholds, sweeps = [], {}, {}, {}, {}

    for kind in ("scratch", "fasterrcnn"):
        label = MODEL_LABELS[kind]
        ckpt_path = Path(args.checkpoints) / f"{kind}_best.pt"
        print(f"\n[{label}] loading {ckpt_path.name}")
        model, cfg, ckpt = load_checkpoint(ckpt_path, device)

        # ---- 1. choose the operating point on VALIDATION -----------------
        print("  selecting confidence threshold on the validation split...")
        valid_ds = dataset_for(kind, args.data, "valid", cfg)
        valid_preds = predict_split(model, kind, valid_ds, cfg, device)
        best_thr, sweep = select_confidence_threshold(
            valid_preds, gt_valid, cfg.class_names, cfg.eval_iou_threshold
        )
        cfg.conf_threshold = best_thr
        thresholds[label], sweeps[label] = best_thr, sweep
        # Persist it so the demo script runs at the calibrated operating point
        # rather than the config default.
        update_checkpoint_threshold(ckpt_path, best_thr)
        print(f"  -> threshold {best_thr:.2f} (chosen by validation F1, then frozen)")

        # ---- 2. evaluate on TEST ----------------------------------------
        test_ds = dataset_for(kind, args.data, args.split, cfg)
        test_preds = predict_split(model, kind, test_ds, cfg, device)
        result = evaluate_detections(
            test_preds, gt_test, cfg.class_names,
            iou_threshold=cfg.eval_iou_threshold,
            conf_threshold=best_thr,
        )
        results[label] = result
        print(f"  mAP@0.5 {result.map_50:.4f} | mAP@[.5:.95] {result.map_50_95:.4f} | "
              f"P {result.precision:.3f} R {result.recall:.3f} F1 {result.f1:.3f}")

        # ---- 3. latency --------------------------------------------------
        if args.no_latency:
            latency = {"median_ms": float("nan"), "p95_ms": float("nan"), "fps": float("nan")}
        else:
            latency = benchmark_latency(model, kind, test_ds, cfg, device)
            print(f"  latency {latency['median_ms']:.1f} ms median "
                  f"({latency['fps']:.1f} FPS, p95 {latency['p95_ms']:.1f} ms)")

        # ---- 4. failure taxonomy ----------------------------------------
        analyses[label] = analyse_failures(
            test_preds, gt_test, cfg.class_names,
            iou_threshold=cfg.eval_iou_threshold, conf_threshold=best_thr,
        )

        training_json = RESULTS_DIR / f"{kind}_training.json"
        train_summary = load_json(training_json) if training_json.exists() else {}

        rows.append({
            "model": label,
            "map50": result.map_50,
            "map50_95": result.map_50_95,
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
            "tp": result.tp, "fp": result.fp, "fn": result.fn,
            "conf_threshold": best_thr,
            "latency_ms": latency["median_ms"],
            "latency_p95_ms": latency["p95_ms"],
            "fps": latency["fps"],
            "params": count_parameters(model),
            "size_mb": model_size_mb(model),
            "checkpoint_mb": file_size_mb(ckpt_path),
            "train_seconds": train_summary.get("train_seconds", float("nan")),
            "best_epoch": ckpt.get("epoch", "?"),
            "epochs_run": train_summary.get("epochs_run", len(ckpt.get("history", []))),
        })

        # ---- 5. per-model figures ---------------------------------------
        if not args.no_figures:
            viz.plot_threshold_sweep(sweep, best_thr, label,
                                     FIGURES_DIR / f"{kind}_threshold_sweep.png")
            viz.plot_confusion_matrix(result, FIGURES_DIR / f"{kind}_confusion_matrix.png")
            history = ckpt.get("history", [])
            if history:
                viz.plot_training_curves(history, label,
                                         FIGURES_DIR / f"{kind}_training_curves.png")
                if kind == "scratch":
                    viz.plot_loss_components(history,
                                             FIGURES_DIR / "scratch_loss_components.png")

    # ---- comparison figures ---------------------------------------------
    if not args.no_figures:
        viz.plot_pr_curves(results, FIGURES_DIR / "pr_curves.png")
        viz.plot_per_class_ap(results, FIGURES_DIR / "per_class_ap.png")
        viz.plot_speed_accuracy(rows, FIGURES_DIR / "speed_vs_accuracy.png")
        viz.plot_architecture_diagram(FIGURES_DIR / "system_workflow.png")

    # ---- write results ---------------------------------------------------
    markdown = build_comparison_markdown(rows, results, analyses, thresholds, hardware)
    (RESULTS_DIR / "comparison.md").write_text(markdown, encoding="utf-8")

    save_json(
        {
            "hardware": hardware,
            "split": args.split,
            "thresholds": thresholds,
            "models": {r["model"]: r for r in rows},
            "detail": {name: res.summary_dict() for name, res in results.items()},
            "failures": {
                name: {
                    "counts": a["counts"],
                    "error_share": a["error_share"],
                    "miss_rate_by_bucket": a["miss_rate_by_bucket"],
                    "top_confusions": [list(c) for c in a["top_confusions"]],
                }
                for name, a in analyses.items()
            },
        },
        RESULTS_DIR / "metrics.json",
    )

    print("\n" + "=" * 74)
    print(f"  results/comparison.md   <- the comparison table")
    print(f"  results/metrics.json    <- machine-readable metrics")
    if not args.no_figures:
        print(f"  results/figures/        <- {len(list(FIGURES_DIR.glob('*.png')))} figures")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
