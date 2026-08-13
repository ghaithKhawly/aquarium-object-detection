"""All plotting for the project.

Kept separate from the analysis so that every figure is reproducible from
saved result dictionaries without re-running a model. Each function returns
the matplotlib Figure and, when given `save_path`, writes it to disk at a
consistent DPI.

A single colour-blind-safe palette is used throughout, and Model 1 / Model 2
keep the same two colours in every figure so the comparison reads at a glance.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

__all__ = [
    "MODEL_COLORS",
    "save_figure",
    "plot_class_distribution",
    "plot_box_size_distribution",
    "plot_grid_collision",
    "plot_training_curves",
    "plot_loss_components",
    "plot_pr_curves",
    "plot_per_class_ap",
    "plot_confusion_matrix",
    "plot_threshold_sweep",
    "draw_boxes",
    "plot_qualitative_comparison",
    "plot_failure_gallery",
    "plot_speed_accuracy",
    "plot_architecture_diagram",
]

# Okabe-Ito derived: distinguishable under the common forms of colour blindness
MODEL_COLORS = {
    "Model 1 (Scratch Grid)": "#E69F00",
    "Model 2 (Faster R-CNN)": "#0072B2",
    "ground_truth": "#009E73",
    "prediction": "#D55E00",
}
CLASS_PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7",
    "#56B4E9", "#D55E00", "#F0E442",
]

DPI = 150


def save_figure(fig, save_path: str | Path | None) -> None:
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")


# --------------------------------------------------------------------------
# Dataset figures
# --------------------------------------------------------------------------
def plot_class_distribution(distribution: dict, save_path=None):
    """Grouped bars: object count per class, per split."""
    per_split = distribution["per_split"]
    splits = list(per_split.keys())
    classes = sorted(distribution["totals"], key=lambda c: -distribution["totals"][c])

    fig, axes = plt.subplots(1, len(splits) + 1, figsize=(5 * (len(splits) + 1), 4.5))

    for ax, split in zip(axes, splits):
        counts = [per_split[split].get(c, 0) for c in classes]
        ax.bar(range(len(classes)), counts, color=CLASS_PALETTE[: len(classes)])
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_title(f"{split}  ({sum(counts)} objects)")
        ax.set_ylabel("objects")
        ax.grid(axis="y", alpha=0.3)

    ax = axes[-1]
    totals = [distribution["totals"].get(c, 0) for c in classes]
    ax.barh(range(len(classes)), totals, color=CLASS_PALETTE[: len(classes)])
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.invert_yaxis()
    ax.set_title(f"All splits (imbalance {distribution['imbalance_ratio']:.1f}:1)")
    ax.set_xlabel("objects")
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Class distribution and imbalance", fontsize=13, y=1.02)
    fig.tight_layout()
    save_figure(fig, save_path)
    return fig


def plot_box_size_distribution(stats: dict, save_path=None):
    """Object area histogram plus the small/medium/large breakdown."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    areas = np.sqrt(np.asarray(stats["areas"], dtype=np.float64))
    axes[0].hist(areas, bins=40, color="#0072B2", alpha=0.85)
    axes[0].axvline(32, color="#D55E00", ls="--", label="small / medium (32 px)")
    axes[0].axvline(96, color="#009E73", ls="--", label="medium / large (96 px)")
    axes[0].set_xlabel("sqrt(object area) in pixels @224")
    axes[0].set_ylabel("objects")
    axes[0].set_title("Object scale")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    buckets = stats["size_buckets"]
    labels = ["small\n<32px", "medium\n32-96px", "large\n>96px"]
    values = [buckets["small"], buckets["medium"], buckets["large"]]
    bars = axes[1].bar(labels, values, color=["#D55E00", "#E69F00", "#009E73"])
    for bar, value in zip(bars, values):
        pct = 100.0 * value / max(sum(values), 1)
        axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{pct:.1f}%",
                     ha="center", va="bottom", fontsize=10)
    axes[1].set_ylabel("objects")
    axes[1].set_title("COCO size buckets")
    axes[1].grid(axis="y", alpha=0.3)

    counts = stats["per_image_counts"]
    axes[2].hist(counts, bins=range(0, max(counts) + 2), color="#CC79A7", alpha=0.85)
    axes[2].axvline(np.mean(counts), color="black", ls="--",
                    label=f"mean {np.mean(counts):.1f}")
    axes[2].set_xlabel("objects per image")
    axes[2].set_ylabel("images")
    axes[2].set_title("Scene density")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    save_figure(fig, save_path)
    return fig


def plot_grid_collision(report: dict, save_path=None):
    """Ceiling on recall imposed by each candidate grid size."""
    sizes = sorted(report)
    loss = [report[s]["loss_pct"] for s in sizes]
    ceiling = [100 * report[s]["max_recall"] for s in sizes]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(sizes))
    ax.bar(x - 0.2, loss, 0.4, label="objects that cannot be encoded (%)", color="#D55E00")
    ax.bar(x + 0.2, ceiling, 0.4, label="maximum achievable recall (%)", color="#009E73")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S={s}\n({report[s]['cells']} cells)" for s in sizes])
    ax.set_ylabel("percent")
    ax.set_title("Grid resolution vs. representable objects\n"
                 "(one object per cell; centre-based assignment)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    for xi, (l, c) in enumerate(zip(loss, ceiling)):
        ax.text(xi - 0.2, l, f"{l:.1f}", ha="center", va="bottom", fontsize=9)
        ax.text(xi + 0.2, c, f"{c:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    save_figure(fig, save_path)
    return fig


# --------------------------------------------------------------------------
# Training figures
# --------------------------------------------------------------------------
def plot_training_curves(history: list[dict], title: str, save_path=None):
    """Loss curves plus the validation mAP that drove model selection."""
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train", color="#0072B2")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="validation", color="#D55E00")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title(f"{title} - loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    val_map = [h["val_map50"] for h in history]
    axes[1].plot(epochs, val_map, color="#009E73", label="val mAP@0.5")
    best_idx = int(np.argmax(val_map))
    axes[1].axvline(epochs[best_idx], color="black", ls="--", alpha=0.6,
                    label=f"best: epoch {epochs[best_idx]} ({val_map[best_idx]:.3f})")
    axes[1].scatter([epochs[best_idx]], [val_map[best_idx]], color="black", zorder=5)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("mAP@0.5")
    axes[1].set_title(f"{title} - validation mAP (checkpoint criterion)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    save_figure(fig, save_path)
    return fig


def plot_loss_components(history: list[dict], save_path=None):
    """Per-term breakdown of the multi-task loss (Model 1 only).

    Shows which of the four objectives is actually driving the total, and
    exposes the classic grid-detector pathology where the background term
    dominates and localisation never improves.
    """
    keys = [("train_loc", "localisation"), ("train_obj", "objectness (object cells)"),
            ("train_noobj", "objectness (background)"), ("train_cls", "classification")]
    available = [(k, label) for k, label in keys if k in history[0]]
    if not available:
        return None

    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (key, label), color in zip(available, CLASS_PALETTE):
        ax.plot(epochs, [h[key] for h in history], label=label, color=color)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss term (unweighted)")
    ax.set_title("Model 1 - multi-task loss components")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, save_path)
    return fig


# --------------------------------------------------------------------------
# Evaluation figures
# --------------------------------------------------------------------------
def plot_pr_curves(results: dict, save_path=None):
    """Micro PR curve per model, plus per-class curves for the better model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for (name, result), color in zip(results.items(), MODEL_COLORS.values()):
        all_rec, all_prec = [], []
        for cls_name, (rec, prec) in result.pr_curves.items():
            if len(rec):
                all_rec.append(rec)
                all_prec.append(prec)
        if not all_rec:
            continue
        grid = np.linspace(0, 1, 101)
        interp = []
        for rec, prec in zip(all_rec, all_prec):
            envelope = np.maximum.accumulate(prec[::-1])[::-1]
            interp.append(np.interp(grid, rec, envelope, left=envelope[0], right=0.0))
        mean_curve = np.mean(interp, axis=0)
        axes[0].plot(grid, mean_curve, label=f"{name}  (mAP@0.5 = {result.map_50:.3f})",
                     color=MODEL_COLORS.get(name, color), linewidth=2)

    axes[0].set_xlabel("recall")
    axes[0].set_ylabel("precision")
    axes[0].set_title("Precision-Recall, averaged over classes (IoU >= 0.5)")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    best_name = max(results, key=lambda n: results[n].map_50)
    best = results[best_name]
    for (cls_name, (rec, prec)), color in zip(best.pr_curves.items(), CLASS_PALETTE):
        if not len(rec):
            continue
        envelope = np.maximum.accumulate(prec[::-1])[::-1]
        ap = best.per_class[cls_name]["ap"]
        axes[1].plot(rec, envelope, label=f"{cls_name} (AP {ap:.3f})", color=color)
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title(f"Per-class PR - {best_name}")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    save_figure(fig, save_path)
    return fig


def plot_per_class_ap(results: dict, save_path=None):
    """Side-by-side per-class AP, annotated with support."""
    names = list(results)
    classes = list(next(iter(results.values())).class_names)
    x = np.arange(len(classes))
    width = 0.8 / len(names)

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, name in enumerate(names):
        res = results[name]
        aps = [res.per_class[c]["ap"] if not np.isnan(res.per_class[c]["ap"]) else 0.0
               for c in classes]
        ax.bar(x + i * width - 0.4 + width / 2, aps, width,
               label=f"{name} (mAP {res.map_50:.3f})",
               color=MODEL_COLORS.get(name, CLASS_PALETTE[i]))

    support = next(iter(results.values())).per_class
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\nn={support[c]['n_gt']}" for c in classes])
    ax.set_ylabel("AP@0.5")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class Average Precision (test split)\n"
                 "n = number of ground-truth objects of that class")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, save_path)
    return fig


def plot_confusion_matrix(result, save_path=None, normalize: bool = True):
    """Confusion matrix with an explicit background row and column."""
    cm = result.confusion_matrix.astype(np.float64)
    labels = list(result.class_names) + ["background"]

    display = cm.copy()
    if normalize:
        row_sums = display.sum(axis=1, keepdims=True)
        display = np.divide(display, row_sums, out=np.zeros_like(display),
                            where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(8, 6.8))
    im = ax.imshow(display, cmap="Blues", vmin=0, vmax=display.max() or 1)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title("Confusion matrix (IoU >= 0.5, class-agnostic matching)\n"
                 "last row = false positives, last column = missed detections")

    threshold = display.max() / 2 if display.max() else 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = f"{display[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            if normalize and cm[i, j] > 0:
                text = f"{display[i, j]:.2f}\n({int(cm[i, j])})"
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="white" if display[i, j] > threshold else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised" if normalize else "count")
    fig.tight_layout()
    save_figure(fig, save_path)
    return fig


def plot_threshold_sweep(sweep: list[dict], chosen: float, model_name: str, save_path=None):
    """Precision/recall/F1 against confidence, with the selected operating point."""
    thresholds = [s["threshold"] for s in sweep]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(thresholds, [s["precision"] for s in sweep], label="precision", color="#0072B2")
    ax.plot(thresholds, [s["recall"] for s in sweep], label="recall", color="#D55E00")
    ax.plot(thresholds, [s["f1"] for s in sweep], label="F1", color="#009E73", linewidth=2)
    ax.axvline(chosen, color="black", ls="--", alpha=0.7, label=f"chosen: {chosen:.2f}")
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("score")
    ax.set_title(f"{model_name} - operating point selected on the VALIDATION split")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, save_path)
    return fig


def plot_speed_accuracy(comparison: list[dict], save_path=None):
    """Accuracy against latency - the trade-off the project is really about.

    Returns None when no usable latency measurement exists (for example after
    `evaluate.py --no-latency`), rather than failing on a log axis with no
    positive data.
    """
    usable = [
        r for r in comparison
        if np.isfinite(r.get("latency_ms", np.nan)) and r["latency_ms"] > 0
    ]
    if not usable:
        return None

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for row in usable:
        color = MODEL_COLORS.get(row["model"], "#666666")
        ax.scatter(row["latency_ms"], row["map50"], s=np.sqrt(row["params"]) / 3,
                   color=color, alpha=0.85, edgecolors="black", zorder=3)
        ax.annotate(
            f"{row['model']}\n{row['params']/1e6:.1f}M params\n{row['fps']:.0f} FPS",
            (row["latency_ms"], row["map50"]),
            textcoords="offset points", xytext=(12, -6), fontsize=9,
        )

    # Log scale only when the two models are far enough apart to need it.
    latencies = [r["latency_ms"] for r in usable]
    log_scale = len(latencies) > 1 and max(latencies) / min(latencies) >= 5
    if log_scale:
        ax.set_xscale("log")
    ax.set_xlabel("median inference latency, ms/image"
                  + (" (log scale)" if log_scale else ""))
    ax.set_ylabel("mAP@0.5 on test split")
    ax.set_title("Quality vs. efficiency\n(marker area proportional to parameter count)")
    ax.grid(alpha=0.3, which="both")
    ax.margins(x=0.25, y=0.15)
    fig.tight_layout()
    save_figure(fig, save_path)
    return fig


# --------------------------------------------------------------------------
# Qualitative figures
# --------------------------------------------------------------------------
def draw_boxes(ax, image, boxes, labels=None, scores=None, class_names=None,
               color="#D55E00", linewidth=2, linestyle="-", show_labels=True):
    """Draw an image with boxes onto an existing axis."""
    ax.imshow(image)
    boxes = np.asarray(boxes).reshape(-1, 4)

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                               edgecolor=color, linewidth=linewidth, linestyle=linestyle))
        if not show_labels:
            continue
        parts = []
        if labels is not None and class_names is not None and i < len(labels):
            idx = int(labels[i])
            parts.append(class_names[idx] if 0 <= idx < len(class_names) else f"id{idx}")
        if scores is not None and i < len(scores):
            parts.append(f"{scores[i]:.2f}")
        if parts:
            ax.text(x1, max(y1 - 3, 8), " ".join(parts), color="white", fontsize=7,
                    weight="bold",
                    bbox=dict(facecolor=color, alpha=0.85, pad=1, edgecolor="none"))
    ax.axis("off")
    return ax


def plot_qualitative_comparison(samples, class_names, save_path=None, title=None):
    """Rows of (ground truth | Model 1 | Model 2) for the same images.

    `samples` is a list of dicts with keys: image, gt, pred1, pred2, name.
    Each prediction entry is (boxes, scores, labels); gt is (boxes, labels).
    """
    n = len(samples)
    if n == 0:
        return None

    fig, axes = plt.subplots(n, 3, figsize=(13, 4.4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    headers = ["Ground truth", "Model 1 (Scratch Grid)", "Model 2 (Faster R-CNN)"]
    colors = [MODEL_COLORS["ground_truth"],
              MODEL_COLORS["Model 1 (Scratch Grid)"],
              MODEL_COLORS["Model 2 (Faster R-CNN)"]]

    for r, sample in enumerate(samples):
        gt_boxes, gt_labels = sample["gt"]
        panels = [
            (gt_boxes, None, gt_labels),
            sample["pred1"],
            sample["pred2"],
        ]
        for c, (payload, color) in enumerate(zip(panels, colors)):
            boxes, scores, labels = payload
            ax = axes[r, c]
            draw_boxes(ax, sample["image"], boxes, labels, scores, class_names, color=color)
            if r == 0:
                ax.set_title(headers[c], fontsize=12, weight="bold")
            if c == 0:
                ax.text(-0.04, 0.5, sample.get("name", ""), transform=ax.transAxes,
                        rotation=90, va="center", ha="right", fontsize=8)

    if title:
        fig.suptitle(title, fontsize=13, y=1.005)
    fig.tight_layout()
    save_figure(fig, save_path)
    return fig


def plot_failure_gallery(cases, class_names, save_path=None, title="Failure analysis"):
    """Worst cases with ground truth and predictions overlaid.

    `cases` is a list of dicts: image, gt (boxes, labels), pred (boxes, scores,
    labels), caption.
    """
    n = len(cases)
    if n == 0:
        return None
    cols = min(4, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.9 * rows), squeeze=False)
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i >= n:
            ax.axis("off")
            continue
        case = cases[i]
        gt_boxes, gt_labels = case["gt"]
        pr_boxes, pr_scores, pr_labels = case["pred"]

        draw_boxes(ax, case["image"], gt_boxes, gt_labels, None, class_names,
                   color=MODEL_COLORS["ground_truth"], linewidth=2)
        draw_boxes(ax, case["image"], pr_boxes, pr_labels, pr_scores, class_names,
                   color=MODEL_COLORS["prediction"], linewidth=1.6, linestyle="--")
        ax.set_title(case.get("caption", ""), fontsize=8)

    fig.suptitle(f"{title}\nsolid green = ground truth, dashed orange = prediction",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    save_figure(fig, save_path)
    return fig


# --------------------------------------------------------------------------
# System diagram
# --------------------------------------------------------------------------
def plot_architecture_diagram(save_path=None):
    """Block diagram of the whole system (requirement 1: system workflow)."""
    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def block(x, y, w, h, text, color, fontsize=9, text_color="black"):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="#333333",
                               linewidth=1.3, alpha=0.92, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, zorder=3, color=text_color, weight="bold" if fontsize > 9 else "normal")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#333333", lw=1.6), zorder=1)

    block(0.2, 3.4, 2.1, 1.3, "Aquarium images\n638 images\n7 classes\nYOLO labels", "#DDEEFF")
    arrow(2.3, 4.05, 3.0, 4.05)

    block(3.0, 3.4, 2.2, 1.3, "Shared preprocessing\nresize 224x224\nlabel parsing\naugmentation (train)", "#FFEEDD")
    arrow(5.2, 4.6, 6.0, 6.0)
    arrow(5.2, 3.5, 6.0, 2.1)

    block(6.0, 5.3, 2.6, 1.5,
          "MODEL 1\nScratch grid detector\n5 conv stages -> 14x14\n1x1 detection head", "#FFE0B2")
    block(6.0, 1.3, 2.6, 1.5,
          "MODEL 2\nFaster R-CNN\nResNet-50 + FPN\nRPN -> ROI heads", "#B3D9F2")

    arrow(8.6, 6.05, 9.4, 5.0)
    arrow(8.6, 2.05, 9.4, 3.1)

    block(9.4, 3.4, 2.0, 1.3, "Shared decode\n+ class-aware NMS\nIoU 0.45", "#E8E8E8")
    arrow(11.4, 4.05, 12.0, 4.05)

    block(12.0, 2.6, 1.85, 2.9,
          "Evaluation\nmAP@0.5\nmAP@.5:.95\nP / R / F1\nconfusion matrix\nlatency, FPS\nfailure analysis", "#D5F0E3", fontsize=8)

    ax.text(7.0, 7.6, "Aquarium Object Detection - System Workflow",
            ha="center", fontsize=14, weight="bold")
    ax.text(7.0, 0.55,
            "Everything outside the two model blocks is shared, which is what makes the comparison controlled.",
            ha="center", fontsize=9, style="italic", color="#444444")

    fig.tight_layout()
    save_figure(fig, save_path)
    return fig
