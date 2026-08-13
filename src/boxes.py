"""Bounding-box geometry: format conversion, IoU, and non-maximum suppression.

Written against NumPy rather than PyTorch on purpose. These are the pieces the
project claims to implement "from scratch", so they are kept free of any
framework detection utilities and are unit-tested in isolation
(`tests/test_core.py`).

Box format convention used everywhere in this project:
  * "yolo"  -> (x_center, y_center, width, height), all normalised to [0, 1]
  * "xyxy"  -> (x_min, y_min, x_max, y_max), in absolute pixels
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "yolo_to_xyxy",
    "xyxy_to_yolo",
    "iou_xyxy",
    "iou_matrix",
    "non_max_suppression",
    "clip_boxes",
]


# --------------------------------------------------------------------------
# Format conversion
# --------------------------------------------------------------------------
def yolo_to_xyxy(boxes: np.ndarray, img_w: float, img_h: float) -> np.ndarray:
    """(N, 4) normalised cx,cy,w,h  ->  (N, 4) absolute x1,y1,x2,y2."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack(
        [
            (cx - w / 2) * img_w,
            (cy - h / 2) * img_h,
            (cx + w / 2) * img_w,
            (cy + h / 2) * img_h,
        ],
        axis=1,
    )


def xyxy_to_yolo(boxes: np.ndarray, img_w: float, img_h: float) -> np.ndarray:
    """(N, 4) absolute x1,y1,x2,y2  ->  (N, 4) normalised cx,cy,w,h."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack(
        [
            ((x1 + x2) / 2) / img_w,
            ((y1 + y2) / 2) / img_h,
            (x2 - x1) / img_w,
            (y2 - y1) / img_h,
        ],
        axis=1,
    )


def clip_boxes(boxes: np.ndarray, img_w: float, img_h: float) -> np.ndarray:
    """Clamp xyxy boxes to the image rectangle (used after geometric aug)."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    boxes[:, 0] = boxes[:, 0].clip(0, img_w)
    boxes[:, 1] = boxes[:, 1].clip(0, img_h)
    boxes[:, 2] = boxes[:, 2].clip(0, img_w)
    boxes[:, 3] = boxes[:, 3].clip(0, img_h)
    return boxes


# --------------------------------------------------------------------------
# Intersection over Union
# --------------------------------------------------------------------------
def iou_xyxy(box_a, box_b) -> float:
    """IoU of two single boxes.

        IoU = area(A n B) / area(A u B)

    The union is computed as area(A) + area(B) - area(intersection) rather
    than by counting pixels, which is why the intersection term appears twice.
    Degenerate boxes (zero or negative area) yield 0.0 instead of a NaN.
    """
    ax1, ay1, ax2, ay2 = (float(v) for v in box_a)
    bx1, by1, bx2, by2 = (float(v) for v in box_b)

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    return float(intersection / union) if union > 0 else 0.0


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vectorised pairwise IoU -> (Na, Nb) matrix.

    Used by the evaluator, where the O(Na*Nb) Python loop of `iou_xyxy` would
    dominate runtime over the whole test split.
    """
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    # Broadcast a as (Na, 1, 4) against b as (1, Nb, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter

    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


# --------------------------------------------------------------------------
# Non-maximum suppression
# --------------------------------------------------------------------------
def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    iou_threshold: float = 0.45,
    score_threshold: float = 0.0,
    max_detections: int | None = None,
) -> np.ndarray:
    """Class-aware greedy NMS. Returns the indices of the boxes to keep.

    Algorithm (as described in section 5 of the notebook):
      1. Drop every box below `score_threshold`.
      2. Sort what remains by score, descending.
      3. Take the top box, keep it, and delete every *same-class* box whose
         IoU against it exceeds `iou_threshold`.
      4. Repeat on what is left.

    Suppression is class-aware: a shark box does not suppress an overlapping
    fish box, because two different objects genuinely can occupy the same
    region. Suppressing across classes would silently destroy recall in the
    crowded, overlapping scenes this dataset is full of.

    Returned indices refer to the *original* arrays and are ordered by
    descending score.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)

    candidates = np.nonzero(scores >= score_threshold)[0]
    if len(candidates) == 0:
        return np.empty((0,), dtype=np.int64)

    # Sort candidates by descending score.
    candidates = candidates[np.argsort(-scores[candidates], kind="stable")]

    keep: list[int] = []
    for cls in np.unique(labels[candidates]):
        cls_idx = candidates[labels[candidates] == cls]
        cls_boxes = boxes[cls_idx]

        alive = np.ones(len(cls_idx), dtype=bool)
        for i in range(len(cls_idx)):
            if not alive[i]:
                continue
            keep.append(int(cls_idx[i]))
            if i + 1 < len(cls_idx):
                ious = iou_matrix(cls_boxes[i : i + 1], cls_boxes[i + 1 :])[0]
                alive[i + 1 :] &= ious <= iou_threshold

    keep_arr = np.array(sorted(keep, key=lambda j: -scores[j]), dtype=np.int64)
    if max_detections is not None:
        keep_arr = keep_arr[:max_detections]
    return keep_arr
