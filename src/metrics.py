"""Detection metrics: per-class AP, mAP, precision/recall/F1, confusion matrix.

Design note - why one matching pass feeds everything
----------------------------------------------------
A common bug in detection evaluation is to compute the operating-point
precision/recall with a "each ground-truth box may be matched once" rule, and
then compute the PR curve with a looser rule such as
`any(iou(pred, gt) >= 0.5 for gt in gts)`. The second rule lets two predictions
on the same object both count as true positives, so cumulative TPs can exceed
the number of ground-truth boxes and recall can climb above 1.0 - the reported
AP then does not correspond to the reported precision and recall at all.

This module computes a single greedy matching per class, and derives the PR
curve, AP, and the operating-point counts from that one result. Whatever the
numbers are, they are mutually consistent.

Matching rule (standard VOC/COCO greedy assignment):
  * predictions of a class are sorted by descending confidence
  * each prediction takes the highest-IoU unclaimed ground-truth box of the
    same class in the same image
  * a match with IoU >= threshold is a TP; anything else is a FP
  * ground-truth boxes never matched are FNs
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .boxes import iou_matrix

__all__ = [
    "ImagePrediction",
    "ImageGroundTruth",
    "DetectionEvalResult",
    "evaluate_detections",
    "compute_average_precision",
    "select_confidence_threshold",
]


# --------------------------------------------------------------------------
# Lightweight containers
# --------------------------------------------------------------------------
@dataclass
class ImagePrediction:
    """Detections for one image."""

    boxes: np.ndarray            # (N, 4) xyxy pixels
    scores: np.ndarray           # (N,)
    labels: np.ndarray           # (N,) int class ids
    image_id: str = ""

    def filter_by_score(self, threshold: float) -> "ImagePrediction":
        keep = self.scores >= threshold
        return ImagePrediction(
            boxes=self.boxes[keep],
            scores=self.scores[keep],
            labels=self.labels[keep],
            image_id=self.image_id,
        )


@dataclass
class ImageGroundTruth:
    """Annotations for one image."""

    boxes: np.ndarray            # (M, 4) xyxy pixels
    labels: np.ndarray           # (M,) int class ids
    image_id: str = ""


@dataclass
class DetectionEvalResult:
    """Everything the report needs from one evaluation run."""

    map_50: float
    map_50_95: float
    precision: float                       # micro-averaged, at the operating threshold
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    per_class: dict[str, dict] = field(default_factory=dict)
    pr_curves: dict[str, tuple] = field(default_factory=dict)
    confusion_matrix: np.ndarray | None = None
    class_names: list[str] = field(default_factory=list)
    conf_threshold: float = 0.0

    def summary_dict(self) -> dict:
        """Flat, JSON-serialisable view for results/*.json."""
        return {
            "mAP@0.5": round(self.map_50, 4),
            "mAP@[.5:.95]": round(self.map_50_95, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "TP": self.tp,
            "FP": self.fp,
            "FN": self.fn,
            "conf_threshold": self.conf_threshold,
            "per_class": {
                name: {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in stats.items()}
                for name, stats in self.per_class.items()
            },
        }


# --------------------------------------------------------------------------
# Core matching
# --------------------------------------------------------------------------
def _match_class(
    predictions: list[ImagePrediction],
    ground_truths: list[ImageGroundTruth],
    class_id: int,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Greedy match every prediction of one class across the whole split.

    Returns (scores, is_tp, n_gt) where `scores` and `is_tp` are aligned and
    sorted by descending score.
    """
    scores_all: list[float] = []
    is_tp_all: list[bool] = []
    n_gt = 0

    for pred, gt in zip(predictions, ground_truths):
        gt_mask = gt.labels == class_id
        gt_boxes = gt.boxes[gt_mask]
        n_gt += len(gt_boxes)

        pred_mask = pred.labels == class_id
        pred_boxes = pred.boxes[pred_mask]
        pred_scores = pred.scores[pred_mask]

        if len(pred_boxes) == 0:
            continue

        # Highest confidence first, so the best prediction gets first claim.
        order = np.argsort(-pred_scores, kind="stable")
        pred_boxes = pred_boxes[order]
        pred_scores = pred_scores[order]

        if len(gt_boxes) == 0:
            scores_all.extend(pred_scores.tolist())
            is_tp_all.extend([False] * len(pred_scores))
            continue

        ious = iou_matrix(pred_boxes, gt_boxes)      # (P, G)
        claimed = np.zeros(len(gt_boxes), dtype=bool)

        for p in range(len(pred_boxes)):
            row = ious[p].copy()
            row[claimed] = -1.0                      # cannot take a claimed GT
            best = int(np.argmax(row)) if len(row) else -1
            if best >= 0 and row[best] >= iou_threshold:
                claimed[best] = True
                is_tp_all.append(True)
            else:
                is_tp_all.append(False)
            scores_all.append(float(pred_scores[p]))

    scores_arr = np.asarray(scores_all, dtype=np.float64)
    is_tp_arr = np.asarray(is_tp_all, dtype=bool)

    if len(scores_arr):
        order = np.argsort(-scores_arr, kind="stable")
        scores_arr, is_tp_arr = scores_arr[order], is_tp_arr[order]

    return scores_arr, is_tp_arr, n_gt


def compute_average_precision(
    scores: np.ndarray, is_tp: np.ndarray, n_gt: int
) -> tuple[float, np.ndarray, np.ndarray]:
    """Average Precision by all-point interpolation (VOC 2010+ / COCO style).

    Precision is made monotonically non-increasing before integrating - the
    "precision envelope". Without it, the sawtooth of the raw curve makes AP
    depend on the order of equal-scoring detections rather than on model
    quality. AP is then the exact area under the stepped envelope, which is
    why we sum rectangles at each recall change rather than calling a
    trapezoidal integrator.

    Returns (ap, recall_curve, precision_curve).
    """
    if n_gt == 0:
        return float("nan"), np.array([]), np.array([])
    if len(scores) == 0:
        return 0.0, np.array([0.0]), np.array([0.0])

    tp_cum = np.cumsum(is_tp.astype(np.float64))
    fp_cum = np.cumsum(~is_tp)

    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)

    # Sentinels at recall 0 and 1 so the envelope covers the full range.
    mrec = np.concatenate(([0.0], recall, [recall[-1]]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    changes = np.nonzero(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]))

    return ap, recall, precision


# --------------------------------------------------------------------------
# Confusion matrix
# --------------------------------------------------------------------------
def _confusion_matrix(
    predictions: list[ImagePrediction],
    ground_truths: list[ImageGroundTruth],
    num_classes: int,
    iou_threshold: float,
) -> np.ndarray:
    """(C+1, C+1) matrix; last row/column is 'background'.

    Matching here is deliberately *class-agnostic*: a prediction is paired with
    whichever ground-truth box it overlaps most, regardless of label. That is
    what makes the off-diagonal cells meaningful - they show which classes the
    model confuses with which, instead of collapsing every mislabelled but
    well-localised box into the background bucket.

    Reading it: row = true class, column = predicted class.
      * cell [c, background] -> objects of class c that were missed
      * cell [background, c] -> class-c detections with no object behind them
    """
    bg = num_classes
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    for pred, gt in zip(predictions, ground_truths):
        if len(gt.boxes) == 0:
            for label in pred.labels:
                cm[bg, int(label)] += 1
            continue
        if len(pred.boxes) == 0:
            for label in gt.labels:
                cm[int(label), bg] += 1
            continue

        order = np.argsort(-pred.scores, kind="stable")
        p_boxes, p_labels = pred.boxes[order], pred.labels[order]

        ious = iou_matrix(p_boxes, gt.boxes)
        claimed = np.zeros(len(gt.boxes), dtype=bool)
        matched_pred = np.zeros(len(p_boxes), dtype=bool)

        for p in range(len(p_boxes)):
            row = ious[p].copy()
            row[claimed] = -1.0
            best = int(np.argmax(row)) if len(row) else -1
            if best >= 0 and row[best] >= iou_threshold:
                claimed[best] = True
                matched_pred[p] = True
                cm[int(gt.labels[best]), int(p_labels[p])] += 1

        for p in np.nonzero(~matched_pred)[0]:
            cm[bg, int(p_labels[p])] += 1           # false positive
        for g in np.nonzero(~claimed)[0]:
            cm[int(gt.labels[g]), bg] += 1          # missed detection

    return cm


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def evaluate_detections(
    predictions: list[ImagePrediction],
    ground_truths: list[ImageGroundTruth],
    class_names: list[str],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.0,
    compute_map_range: bool = True,
) -> DetectionEvalResult:
    """Full evaluation of one model on one split.

    `predictions` should contain *all* detections the model produced (already
    NMS-ed). Threshold-free metrics (AP, mAP) are computed on the full set,
    because AP is defined as an average over all operating points. The
    operating-point metrics (precision, recall, F1, confusion matrix) are
    computed on detections above `conf_threshold`.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"predictions ({len(predictions)}) and ground_truths "
            f"({len(ground_truths)}) must describe the same images"
        )

    num_classes = len(class_names)

    # ---- Threshold-free: per-class AP at the primary IoU threshold -------
    per_class: dict[str, dict] = {}
    pr_curves: dict[str, tuple] = {}
    aps: list[float] = []

    for cls_id, name in enumerate(class_names):
        scores, is_tp, n_gt = _match_class(
            predictions, ground_truths, cls_id, iou_threshold
        )
        ap, rec, prec = compute_average_precision(scores, is_tp, n_gt)
        if n_gt > 0:
            aps.append(ap)
            pr_curves[name] = (rec, prec)
        per_class[name] = {"ap": ap, "n_gt": int(n_gt)}

    map_50 = float(np.mean(aps)) if aps else 0.0

    # ---- COCO-style mAP averaged over IoU 0.50:0.05:0.95 ----------------
    map_50_95 = float("nan")
    if compute_map_range:
        per_threshold = []
        for thr in np.arange(0.50, 0.96, 0.05):
            thr_aps = []
            for cls_id, name in enumerate(class_names):
                if per_class[name]["n_gt"] == 0:
                    continue
                s, t, n = _match_class(predictions, ground_truths, cls_id, float(thr))
                thr_aps.append(compute_average_precision(s, t, n)[0])
            if thr_aps:
                per_threshold.append(float(np.mean(thr_aps)))
        map_50_95 = float(np.mean(per_threshold)) if per_threshold else 0.0

    # ---- Operating point: precision / recall / F1 -----------------------
    kept = [p.filter_by_score(conf_threshold) for p in predictions]
    total_tp = total_fp = total_fn = 0

    for cls_id, name in enumerate(class_names):
        _, is_tp, n_gt = _match_class(kept, ground_truths, cls_id, iou_threshold)
        tp = int(is_tp.sum())
        fp = int((~is_tp).sum())
        fn = int(n_gt - tp)

        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / n_gt if n_gt else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0

        per_class[name].update(
            {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    )

    cm = _confusion_matrix(kept, ground_truths, num_classes, iou_threshold)

    return DetectionEvalResult(
        map_50=map_50,
        map_50_95=map_50_95,
        precision=micro_p,
        recall=micro_r,
        f1=micro_f1,
        tp=total_tp,
        fp=total_fp,
        fn=total_fn,
        per_class=per_class,
        pr_curves=pr_curves,
        confusion_matrix=cm,
        class_names=list(class_names),
        conf_threshold=conf_threshold,
    )


def select_confidence_threshold(
    predictions: list[ImagePrediction],
    ground_truths: list[ImageGroundTruth],
    class_names: list[str],
    iou_threshold: float = 0.5,
    candidates: np.ndarray | None = None,
) -> tuple[float, list[dict]]:
    """Pick the operating threshold that maximises micro-F1.

    This is run on the **validation** split only. Choosing the threshold on the
    test split would be a form of leakage: the reported test numbers would be
    the best of many peeks at data that is supposed to be untouched. Selecting
    on validation and then freezing the value keeps the test result honest.

    Returns (best_threshold, sweep) where `sweep` records every candidate so
    the choice can be plotted and defended.
    """
    if candidates is None:
        candidates = np.arange(0.05, 0.91, 0.05)

    sweep: list[dict] = []
    best_thr, best_f1 = float(candidates[0]), -1.0

    for thr in candidates:
        res = evaluate_detections(
            predictions,
            ground_truths,
            class_names,
            iou_threshold=iou_threshold,
            conf_threshold=float(thr),
            compute_map_range=False,
        )
        sweep.append(
            {
                "threshold": float(thr),
                "precision": res.precision,
                "recall": res.recall,
                "f1": res.f1,
            }
        )
        if res.f1 > best_f1:
            best_f1, best_thr = res.f1, float(thr)

    return best_thr, sweep
