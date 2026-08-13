"""Automatic classification and aggregation of detection errors.

The project guide asks for "representative failures and a technical
explanation of likely causes". Eyeballing a few pictures and asserting a cause
is not evidence, so this module assigns every prediction and every missed
object to one of six mutually exclusive outcomes, then aggregates them. The
resulting table says *what kind* of mistake each model makes and *how often*,
which turns the write-up into something measured rather than claimed.

Error taxonomy
--------------
For each prediction, in descending confidence order:

  true_positive            correct class, IoU >= t against an unclaimed object
  duplicate_detection      correct class and IoU >= t, but that object was
                           already claimed by a higher-scoring box -> NMS let
                           two boxes through for one object
  wrong_class              localised an object well (IoU >= t) but named it
                           something else -> a classification failure, not a
                           detection failure
  poor_localization        right class, best IoU falls in [t_low, t) -> the
                           object was found but the box regression is loose
  background_false_positive  best IoU below t_low against anything -> the model
                           hallucinated an object where there is none

and for each ground-truth object left unclaimed:

  missed_detection         nothing of that class landed on it

Distinguishing these matters because they have different fixes: wrong_class
points at the classifier head or at visually similar classes, poor_localization
at the box regression or grid resolution, background_false_positive at the
confidence threshold or at background texture, missed_detection at object
scale, occlusion, or class imbalance.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .boxes import iou_matrix
from .metrics import ImageGroundTruth, ImagePrediction

__all__ = [
    "ErrorRecord",
    "classify_image_errors",
    "analyse_failures",
    "rank_worst_images",
    "failure_markdown_table",
    "failure_summary_text",
    "size_bucket",
]

ERROR_TYPES = [
    "true_positive",
    "duplicate_detection",
    "wrong_class",
    "poor_localization",
    "background_false_positive",
    "missed_detection",
]


def size_bucket(box: np.ndarray) -> str:
    """COCO size convention, applied in the 224x224 evaluation frame."""
    w = max(0.0, float(box[2]) - float(box[0]))
    h = max(0.0, float(box[3]) - float(box[1]))
    area = w * h
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


@dataclass
class ErrorRecord:
    image_id: str
    kind: str
    gt_class: int | None = None
    pred_class: int | None = None
    score: float | None = None
    iou: float = 0.0
    bucket: str = ""


def classify_image_errors(
    pred: ImagePrediction,
    gt: ImageGroundTruth,
    iou_threshold: float = 0.5,
    localization_floor: float = 0.10,
) -> list[ErrorRecord]:
    """Assign every prediction and every ground-truth object an outcome."""
    records: list[ErrorRecord] = []

    n_pred, n_gt = len(pred.boxes), len(gt.boxes)

    if n_pred == 0 and n_gt == 0:
        return records

    if n_pred == 0:
        return [
            ErrorRecord(pred.image_id, "missed_detection", gt_class=int(gt.labels[g]),
                        bucket=size_bucket(gt.boxes[g]))
            for g in range(n_gt)
        ]

    if n_gt == 0:
        return [
            ErrorRecord(pred.image_id, "background_false_positive",
                        pred_class=int(pred.labels[p]), score=float(pred.scores[p]),
                        bucket=size_bucket(pred.boxes[p]))
            for p in range(n_pred)
        ]

    order = np.argsort(-pred.scores, kind="stable")
    p_boxes = pred.boxes[order]
    p_labels = pred.labels[order]
    p_scores = pred.scores[order]

    ious = iou_matrix(p_boxes, gt.boxes)                 # (P, G)
    same_class = p_labels[:, None] == gt.labels[None, :]
    claimed = np.zeros(n_gt, dtype=bool)

    for p in range(len(p_boxes)):
        row = ious[p]
        same = np.where(same_class[p], row, -1.0)

        unclaimed_same = np.where(~claimed, same, -1.0)
        best_unclaimed = float(unclaimed_same.max())
        best_same_any = float(same.max())
        best_any_class = float(row.max())

        if best_unclaimed >= iou_threshold:
            g = int(np.argmax(unclaimed_same))
            claimed[g] = True
            kind, iou, gt_cls = "true_positive", best_unclaimed, int(gt.labels[g])
        elif best_same_any >= iou_threshold:
            kind, iou, gt_cls = "duplicate_detection", best_same_any, int(p_labels[p])
        elif best_any_class >= iou_threshold:
            g = int(np.argmax(row))
            kind, iou, gt_cls = "wrong_class", best_any_class, int(gt.labels[g])
        elif best_same_any >= localization_floor:
            g = int(np.argmax(same))
            kind, iou, gt_cls = "poor_localization", best_same_any, int(gt.labels[g])
        else:
            kind, iou, gt_cls = "background_false_positive", max(best_any_class, 0.0), None

        records.append(
            ErrorRecord(
                image_id=pred.image_id, kind=kind, gt_class=gt_cls,
                pred_class=int(p_labels[p]), score=float(p_scores[p]),
                iou=iou, bucket=size_bucket(p_boxes[p]),
            )
        )

    for g in np.nonzero(~claimed)[0]:
        records.append(
            ErrorRecord(
                image_id=pred.image_id, kind="missed_detection",
                gt_class=int(gt.labels[g]), iou=float(ious[:, g].max()) if len(ious) else 0.0,
                bucket=size_bucket(gt.boxes[g]),
            )
        )

    return records


def analyse_failures(
    predictions: list[ImagePrediction],
    ground_truths: list[ImageGroundTruth],
    class_names: list[str],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.0,
) -> dict:
    """Aggregate the error taxonomy over a whole split."""
    kept = [p.filter_by_score(conf_threshold) for p in predictions]

    all_records: list[ErrorRecord] = []
    per_image: dict[str, list[ErrorRecord]] = {}

    for pred, gt in zip(kept, ground_truths):
        recs = classify_image_errors(pred, gt, iou_threshold)
        per_image[pred.image_id or gt.image_id] = recs
        all_records.extend(recs)

    by_kind = Counter(r.kind for r in all_records)
    total_errors = sum(v for k, v in by_kind.items() if k != "true_positive")

    # Missed detections broken down by object size - the single most useful
    # cut, because it separates "the model is weak" from "the objects are
    # smaller than the architecture can resolve".
    missed_by_bucket = Counter(r.bucket for r in all_records if r.kind == "missed_detection")
    all_gt_buckets = Counter(
        size_bucket(box) for gt in ground_truths for box in gt.boxes
    )
    miss_rate_by_bucket = {
        bucket: (missed_by_bucket.get(bucket, 0) / count if count else 0.0)
        for bucket, count in all_gt_buckets.items()
    }

    by_class: dict[str, Counter] = defaultdict(Counter)
    for r in all_records:
        cls_idx = r.gt_class if r.gt_class is not None else r.pred_class
        if cls_idx is None or not (0 <= cls_idx < len(class_names)):
            by_class["background"][r.kind] += 1
        else:
            by_class[class_names[cls_idx]][r.kind] += 1

    # Which class is mistaken for which, restricted to well-localised boxes.
    confusions = Counter(
        (class_names[r.gt_class], class_names[r.pred_class])
        for r in all_records
        if r.kind == "wrong_class"
        and r.gt_class is not None and r.pred_class is not None
        and 0 <= r.gt_class < len(class_names) and 0 <= r.pred_class < len(class_names)
    )

    fp_kinds = ("background_false_positive", "wrong_class", "poor_localization",
                "duplicate_detection")
    return {
        "counts": {k: by_kind.get(k, 0) for k in ERROR_TYPES},
        "total_predictions": sum(by_kind.get(k, 0) for k in ERROR_TYPES if k != "missed_detection"),
        "total_errors": total_errors,
        "error_share": {
            k: (by_kind.get(k, 0) / total_errors if total_errors else 0.0)
            for k in ERROR_TYPES if k != "true_positive"
        },
        "false_positive_breakdown": {k: by_kind.get(k, 0) for k in fp_kinds},
        "missed_by_bucket": dict(missed_by_bucket),
        "gt_by_bucket": dict(all_gt_buckets),
        "miss_rate_by_bucket": miss_rate_by_bucket,
        "by_class": {k: dict(v) for k, v in by_class.items()},
        "top_confusions": confusions.most_common(8),
        "per_image": per_image,
        "records": all_records,
    }


def rank_worst_images(
    analysis: dict,
    predictions: list[ImagePrediction],
    ground_truths: list[ImageGroundTruth],
    n: int = 4,
) -> list[dict]:
    """Rank images by error rate, normalised by how many objects they contain.

    Normalising matters: without it the ranking simply returns the most
    crowded images every time, which says more about the dataset than about
    the model.
    """
    gt_by_id = {gt.image_id: gt for gt in ground_truths}
    pred_by_id = {p.image_id: p for p in predictions}

    scored = []
    for image_id, records in analysis["per_image"].items():
        gt = gt_by_id.get(image_id)
        if gt is None or len(gt.boxes) == 0:
            continue
        errors = sum(1 for r in records if r.kind != "true_positive")
        badness = errors / len(gt.boxes)
        kinds = Counter(r.kind for r in records if r.kind != "true_positive")
        dominant = kinds.most_common(1)[0][0] if kinds else "none"
        scored.append(
            {
                "image_id": image_id,
                "badness": badness,
                "n_gt": len(gt.boxes),
                "n_errors": errors,
                "dominant_error": dominant,
                "breakdown": dict(kinds),
                "prediction": pred_by_id.get(image_id),
                "ground_truth": gt,
            }
        )

    scored.sort(key=lambda d: (-d["badness"], -d["n_errors"]))
    return scored[:n]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
_CAUSES = {
    "missed_detection":
        "object never proposed - dominated by small/occluded instances and rare classes",
    "background_false_positive":
        "texture or lighting resembling an object; threshold set too permissively",
    "wrong_class":
        "object found but mislabelled - visually similar classes and few training examples",
    "poor_localization":
        "object found, box regression imprecise - limited spatial resolution at the output",
    "duplicate_detection":
        "two boxes survived NMS for one object - suppression IoU too high for crowded scenes",
}


def failure_markdown_table(analyses: dict[str, dict], class_names: list[str]) -> str:
    """Render the comparative failure table as markdown.

    `analyses` maps model name -> the dict returned by `analyse_failures`.
    """
    models = list(analyses)
    lines: list[str] = []

    lines.append("#### Error composition\n")
    header = "| Failure type | " + " | ".join(f"{m} (count / share)" for m in models) + " | Likely cause |"
    lines.append(header)
    lines.append("|---|" + "---|" * len(models) + "---|")

    for kind in ERROR_TYPES:
        if kind == "true_positive":
            continue
        cells = []
        for m in models:
            count = analyses[m]["counts"].get(kind, 0)
            share = analyses[m]["error_share"].get(kind, 0.0)
            cells.append(f"{count} / {share:.0%}")
        lines.append(f"| `{kind}` | " + " | ".join(cells) + f" | {_CAUSES.get(kind, '')} |")

    lines.append("\n#### Miss rate by object size\n")
    lines.append("| Object size | Ground-truth objects | " +
                 " | ".join(f"{m} miss rate" for m in models) + " |")
    lines.append("|---|---|" + "---|" * len(models))
    first = analyses[models[0]]
    for bucket in ("small", "medium", "large"):
        n_gt = first["gt_by_bucket"].get(bucket, 0)
        cells = [f"{analyses[m]['miss_rate_by_bucket'].get(bucket, 0.0):.1%}" for m in models]
        lines.append(f"| {bucket} | {n_gt} | " + " | ".join(cells) + " |")

    lines.append("\n#### Most frequent class confusions (well-localised but mislabelled)\n")
    for m in models:
        confusions = analyses[m]["top_confusions"]
        if not confusions:
            lines.append(f"- **{m}**: none recorded.")
            continue
        pretty = ", ".join(f"{a} -> {b} ({n})" for (a, b), n in confusions[:5])
        lines.append(f"- **{m}**: {pretty}")

    return "\n".join(lines)


def failure_summary_text(name: str, analysis: dict) -> str:
    """One-paragraph plain-language reading of a model's error profile."""
    counts = analysis["counts"]
    share = analysis["error_share"]
    total = analysis["total_errors"]
    if total == 0:
        return f"{name}: no errors recorded at this operating point."

    dominant = max(share, key=share.get)
    summary = (
        f"{name}: {total} errors at the chosen operating point. The dominant mode is "
        f"`{dominant}` ({share[dominant]:.0%} of all errors, {counts[dominant]} cases)."
    )

    miss = analysis["miss_rate_by_bucket"]
    support = analysis.get("gt_by_bucket", {})
    small, large = miss.get("small", 0.0), miss.get("large", 0.0)
    n_small, n_large = support.get("small", 0), support.get("large", 0)

    # The small-vs-large comparison is only worth drawing when both buckets
    # have enough objects to support it. On this dataset the 'large' bucket is
    # often a handful of instances, where one miss swings the rate by tens of
    # percent - stating a ratio from that would be inventing a finding.
    MIN_SUPPORT = 20

    if n_small >= MIN_SUPPORT and n_large >= MIN_SUPPORT and small > 0 and large > 0:
        if small > large:
            summary += (
                f" Missed detections run at {small:.0%} for small objects "
                f"(n={n_small}) against {large:.0%} for large ones (n={n_large}) - "
                f"a {small / large:.1f}x gap, which points at output resolution "
                f"rather than at classification."
            )
        else:
            summary += (
                f" Missed detections are {small:.0%} for small objects "
                f"(n={n_small}) and {large:.0%} for large ones (n={n_large}), so "
                f"object scale is not the dominant driver of misses here."
            )
    elif n_small >= MIN_SUPPORT and small > 0:
        summary += (
            f" Missed detections run at {small:.0%} on small objects "
            f"(n={n_small}); the large-object bucket holds only {n_large} "
            f"instances, too few to compare against."
        )
    return summary
