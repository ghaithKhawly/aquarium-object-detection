"""Dataset, preprocessing, box-aware augmentation, and grid encoding/decoding.

Both models are fed by the *same* base class (`AquariumBase`), so resizing and
augmentation are provably identical between them. Only the final target format
differs, because a grid detector and a two-stage detector need different
supervision:

    AquariumBase
    |-- GridDataset       -> (image, [S, S, 5+C] target grid)     ... Model 1
    +-- DetectionDataset  -> (image, {"boxes", "labels"})          ... Model 2

Keeping the geometry in one place is what makes the comparison controlled: a
difference in measured accuracy cannot be blamed on one model having seen
better-preprocessed images.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from .boxes import clip_boxes, yolo_to_xyxy
from .config import CLASS_NAMES, Config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------
# Label parsing
# --------------------------------------------------------------------------
def parse_yolo_label(label_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read one YOLO `.txt` -> (boxes_yolo (N,4) normalised, labels (N,) int).

    Each line is `class x_center y_center width height`, with the four
    coordinates normalised to [0, 1]. Missing files are legal and mean
    "no objects in this image" - a negative sample, not an error.
    """
    label_path = Path(label_path)
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float64), np.zeros((0,), dtype=np.int64)

    boxes, labels = [], []
    for line_no, raw in enumerate(label_path.read_text().splitlines(), start=1):
        parts = raw.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(
                f"{label_path}:{line_no} has {len(parts)} fields, expected 5"
            )
        cls, xc, yc, w, h = parts
        boxes.append([float(xc), float(yc), float(w), float(h)])
        labels.append(int(float(cls)))

    return (
        np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
        np.asarray(labels, dtype=np.int64),
    )


def list_split_files(root: str | Path, split: str) -> list[str]:
    """Sorted image filenames for one split (sorted => deterministic order)."""
    img_dir = Path(root) / split / "images"
    if not img_dir.exists():
        raise FileNotFoundError(
            f"Missing image directory: {img_dir}\n"
            "Run `python scripts/download_data.py` first (see README section 3)."
        )
    return sorted(
        f.name for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
    )


# --------------------------------------------------------------------------
# Grid encoding / decoding (Model 1)
# --------------------------------------------------------------------------
def encode_grid_targets(
    boxes_xyxy: np.ndarray,
    labels: np.ndarray,
    grid_size: int,
    num_classes: int,
    img_size: int,
) -> tuple[torch.Tensor, int]:
    """Boxes -> [S, S, 5+C] supervision tensor. Also returns objects dropped.

    Channel layout per cell:
        0      objectness p_c            (1 if this cell owns an object)
        1..2   t_x, t_y                  (centre offset *inside* the cell, [0,1])
        3..4   w, h                      (normalised to the whole image, [0,1])
        5..    one-hot class vector

    A cell owns an object when the object's **centre** falls inside it. Two
    objects whose centres land in the same cell cannot both be represented -
    this single-box-per-cell limitation is inherent to the plain grid design.
    When it happens we keep the **larger** object (larger objects carry more
    pixels of evidence and are the ones a coarse detector can plausibly find)
    and return the number dropped, so the encoding loss can be quantified in
    the EDA rather than silently ignored. See `grid_collision_report`.
    """
    target = torch.zeros(grid_size, grid_size, 5 + num_classes, dtype=torch.float32)
    if len(boxes_xyxy) == 0:
        return target, 0

    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2 / img_size
    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2 / img_size
    w = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / img_size
    h = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / img_size
    areas = w * h

    # Largest first, so the winner of any cell collision is assigned first.
    order = np.argsort(-areas, kind="stable")
    dropped = 0

    for i in order:
        if not (0.0 <= cx[i] < 1.0 and 0.0 <= cy[i] < 1.0):
            dropped += 1                       # centre outside the frame
            continue
        if w[i] <= 0 or h[i] <= 0:
            dropped += 1
            continue

        col = min(int(cx[i] * grid_size), grid_size - 1)
        row = min(int(cy[i] * grid_size), grid_size - 1)

        if target[row, col, 0] == 1.0:
            dropped += 1                       # cell already taken
            continue

        target[row, col, 0] = 1.0
        target[row, col, 1] = float(cx[i] * grid_size - col)
        target[row, col, 2] = float(cy[i] * grid_size - row)
        target[row, col, 3] = float(np.clip(w[i], 1e-6, 1.0))
        target[row, col, 4] = float(np.clip(h[i], 1e-6, 1.0))
        target[row, col, 5 + int(labels[i])] = 1.0

    return target, dropped


def decode_grid_predictions(
    raw_output: torch.Tensor,
    img_size: int,
    conf_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[S, S, 5+C] raw logits -> (boxes_xyxy, scores, labels).

    The model emits unbounded logits; the activations live here so that
    training can use numerically stable `*_with_logits` losses (see
    `losses.py`). Applying them in one place also guarantees that training and
    inference agree on what the numbers mean.

        p_c        = sigmoid(logit)             objectness
        t_x, t_y   = sigmoid(logit)             offset within the cell, [0,1]
        w, h       = sigmoid(logit)             fraction of the image, [0,1]
        class      = softmax(logits)

    The detection score is `p_c * max(class_prob)`, the standard YOLO
    class-specific confidence: a box is only as good as the product of "there
    is something here" and "it is this class". Using p_c alone would push
    confidently-located but ambiguously-classified boxes to the top of the
    ranking and distort AP.
    """
    if raw_output.dim() != 3:
        raise ValueError(f"expected [S, S, 5+C], got {tuple(raw_output.shape)}")

    grid_size = raw_output.shape[0]
    out = raw_output.detach().float().cpu()

    obj = torch.sigmoid(out[..., 0])
    txy = torch.sigmoid(out[..., 1:3])
    wh = torch.sigmoid(out[..., 3:5])
    cls_prob = torch.softmax(out[..., 5:], dim=-1)
    cls_conf, cls_id = cls_prob.max(dim=-1)

    score = (obj * cls_conf).numpy()
    keep = np.nonzero(score >= conf_threshold)
    rows, cols = keep[0], keep[1]

    if len(rows) == 0:
        return (
            np.zeros((0, 4), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
        )

    tx = txy[rows, cols, 0].numpy()
    ty = txy[rows, cols, 1].numpy()
    bw = wh[rows, cols, 0].numpy()
    bh = wh[rows, cols, 1].numpy()

    cx = (cols + tx) / grid_size
    cy = (rows + ty) / grid_size

    boxes_yolo = np.stack([cx, cy, bw, bh], axis=1)
    boxes_xyxy = clip_boxes(yolo_to_xyxy(boxes_yolo, img_size, img_size), img_size, img_size)

    return boxes_xyxy, score[rows, cols].astype(np.float64), cls_id[rows, cols].numpy().astype(np.int64)


# --------------------------------------------------------------------------
# Box-aware augmentation
# --------------------------------------------------------------------------
class BoxAwareAugment:
    """Training-time augmentation that keeps boxes consistent with pixels.

    Off-the-shelf `torchvision.transforms` cannot be used directly here: they
    transform the image only, so any geometric operation silently invalidates
    the annotations. Each geometric op below therefore applies the identical
    mapping to the box coordinates.

    Chosen operations and why:
      * horizontal flip  - underwater scenes have no meaningful left/right
        orientation, so this is a free doubling of the effective dataset.
      * colour jitter    - water colour, turbidity and lighting vary hugely
        between tanks and depths; this is the dominant nuisance variable in
        the domain, and the real-world test images come from other sources.
      * small scale/translate - objects appear at many distances from the
        camera; ±15% scale and ±8% translation cover realistic framing
        differences without inventing implausible geometry.

    Deliberately excluded: vertical flip and large rotations (a upside-down
    penguin is not a sample the deployed system would ever see) and mosaic
    (it fabricates object co-occurrences that do not exist in this domain).
    """

    def __init__(self, cfg: Config, enabled: bool = True):
        self.cfg = cfg
        self.enabled = enabled

    def __call__(
        self, img: Image.Image, boxes: np.ndarray, labels: np.ndarray
    ) -> tuple[Image.Image, np.ndarray, np.ndarray]:
        if not self.enabled:
            return img, boxes, labels
        if len(boxes) == 0:
            # No annotations: geometry is a no-op, but photometric jitter is
            # still useful (empty images teach the model what background is).
            return self._jitter(img), boxes, labels

        cfg = self.cfg
        W, H = img.size
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()

        # ---- horizontal flip --------------------------------------------
        if np.random.rand() < cfg.aug_hflip_prob:
            img = TF.hflip(img)
            x1 = boxes[:, 0].copy()
            boxes[:, 0] = W - boxes[:, 2]
            boxes[:, 2] = W - x1

        # ---- scale + translate about the image centre -------------------
        if np.random.rand() < cfg.aug_scale_translate_prob:
            scale = float(np.random.uniform(*cfg.aug_scale_range))
            max_shift = cfg.aug_translate_frac
            tx = float(np.random.uniform(-max_shift, max_shift) * W)
            ty = float(np.random.uniform(-max_shift, max_shift) * H)

            img = TF.affine(
                img, angle=0.0, translate=[tx, ty], scale=scale, shear=[0.0, 0.0]
            )

            # torchvision's affine scales about the image centre, then
            # translates. A point p maps to (p - centre) * scale + centre + t.
            cx_img, cy_img = W / 2.0, H / 2.0
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - cx_img) * scale + cx_img + tx
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - cy_img) * scale + cy_img + ty

            boxes, labels = self._drop_degenerate(boxes, labels, W, H)

        # ---- photometric -------------------------------------------------
        img = self._jitter(img)
        return img, boxes, labels

    # -- helpers -----------------------------------------------------------
    def _jitter(self, img: Image.Image) -> Image.Image:
        cfg = self.cfg
        if not self.enabled or np.random.rand() >= cfg.aug_color_jitter_prob:
            return img
        if cfg.aug_brightness:
            img = TF.adjust_brightness(img, 1 + np.random.uniform(-cfg.aug_brightness, cfg.aug_brightness))
        if cfg.aug_contrast:
            img = TF.adjust_contrast(img, 1 + np.random.uniform(-cfg.aug_contrast, cfg.aug_contrast))
        if cfg.aug_saturation:
            img = TF.adjust_saturation(img, 1 + np.random.uniform(-cfg.aug_saturation, cfg.aug_saturation))
        if cfg.aug_hue:
            img = TF.adjust_hue(img, float(np.random.uniform(-cfg.aug_hue, cfg.aug_hue)))
        return img

    @staticmethod
    def _drop_degenerate(
        boxes: np.ndarray, labels: np.ndarray, W: float, H: float, min_visible: float = 0.35
    ) -> tuple[np.ndarray, np.ndarray]:
        """Clip to the frame and discard boxes mostly pushed out of view.

        A box scaled or shifted until only a sliver remains is a mislabelled
        training signal, not a hard example - the pixels of the object are
        gone but the label still claims a full instance.
        """
        if len(boxes) == 0:
            return boxes, labels
        area_before = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
            boxes[:, 3] - boxes[:, 1], 0, None
        )
        clipped = clip_boxes(boxes, W, H)
        area_after = np.clip(clipped[:, 2] - clipped[:, 0], 0, None) * np.clip(
            clipped[:, 3] - clipped[:, 1], 0, None
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            visible = np.where(area_before > 0, area_after / area_before, 0.0)
        keep = (visible >= min_visible) & (area_after > 1.0)
        return clipped[keep], labels[keep]


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------
class AquariumBase(Dataset):
    """Shared image/annotation loading, resizing, and augmentation.

    Everything that could bias the comparison lives here and is therefore
    shared verbatim by both models.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        cfg: Config,
        augment: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.cfg = cfg
        self.img_size = cfg.img_size
        self.img_dir = self.root / split / "images"
        self.label_dir = self.root / split / "labels"
        self.files = list_split_files(root, split)
        self.augment = augment
        self.augmenter = BoxAwareAugment(cfg, enabled=augment)

    def __len__(self) -> int:
        return len(self.files)

    def label_path(self, fname: str) -> Path:
        return self.label_dir / (Path(fname).stem + ".txt")

    def load_raw(self, idx: int):
        """Load one sample: resized RGB image + boxes in resized pixel space.

        Resizing happens *before* the boxes are converted to pixels, so the
        normalised YOLO coordinates map exactly onto the resized frame and no
        rescaling error can creep in. The aspect ratio is not preserved - a
        deliberate choice, applied identically to both models, that keeps the
        tensor shape fixed and avoids letterbox padding that would otherwise
        need to be undone at evaluation time.
        """
        fname = self.files[idx]
        img = Image.open(self.img_dir / fname).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        boxes_yolo, labels = parse_yolo_label(self.label_path(fname))
        boxes_xyxy = yolo_to_xyxy(boxes_yolo, self.img_size, self.img_size)

        img, boxes_xyxy, labels = self.augmenter(img, boxes_xyxy, labels)
        boxes_xyxy = clip_boxes(boxes_xyxy, self.img_size, self.img_size)

        return img, boxes_xyxy, labels, fname

    @staticmethod
    def to_tensor(img: Image.Image) -> torch.Tensor:
        """PIL -> float tensor in [0, 1].

        Normalisation is intentionally *not* applied here. Both models
        normalise internally with the same ImageNet statistics (Model 1 via a
        registered buffer, Model 2 via torchvision's built-in
        GeneralizedRCNNTransform), so doing it here as well would normalise
        twice for Model 2 and silently corrupt its pretrained features.
        """
        return TF.to_tensor(img)


class GridDataset(AquariumBase):
    """Model 1 view: image -> [S, S, 5+C] target grid."""

    def __getitem__(self, idx: int):
        img, boxes, labels, fname = self.load_raw(idx)
        target, dropped = encode_grid_targets(
            boxes,
            labels,
            grid_size=self.cfg.grid_size,
            num_classes=self.cfg.num_classes,
            img_size=self.img_size,
        )
        return self.to_tensor(img), target, {"file": fname, "dropped": dropped}


class DetectionDataset(AquariumBase):
    """Model 2 view: image -> torchvision detection target dict."""

    def __getitem__(self, idx: int):
        img, boxes, labels, fname = self.load_raw(idx)

        # torchvision reserves label 0 for background, so class ids shift by 1.
        if len(boxes):
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels + 1, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
            "area": (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0]),
            "iscrowd": torch.zeros((len(boxes_t),), dtype=torch.int64),
        }
        return self.to_tensor(img), target, {"file": fname}


def detection_collate(batch):
    """Detection targets have variable length, so they stay as tuples."""
    return tuple(zip(*batch))


def grid_collate(batch):
    imgs, targets, metas = zip(*batch)
    return torch.stack(imgs, 0), torch.stack(targets, 0), list(metas)


# --------------------------------------------------------------------------
# Ground-truth access for evaluation (no augmentation, ever)
# --------------------------------------------------------------------------
def load_ground_truth(root: str | Path, split: str, cfg: Config):
    """Ground truth for a split in evaluation space (resized pixels).

    Evaluation always reads annotations through this function rather than
    through a Dataset, so there is no way for augmentation to leak into the
    reference boxes.
    """
    from .metrics import ImageGroundTruth

    root = Path(root)
    out = []
    for fname in list_split_files(root, split):
        boxes_yolo, labels = parse_yolo_label(root / split / "labels" / (Path(fname).stem + ".txt"))
        out.append(
            ImageGroundTruth(
                boxes=yolo_to_xyxy(boxes_yolo, cfg.img_size, cfg.img_size),
                labels=labels,
                image_id=fname,
            )
        )
    return out


# --------------------------------------------------------------------------
# Dataset auditing helpers (used by eda.py)
# --------------------------------------------------------------------------
def image_content_hash(path: str | Path, size: int = 16) -> str:
    """Perceptual-ish hash: downscale to greyscale and threshold at the mean.

    Exact-byte hashing would miss a duplicate that was re-encoded or resized,
    which is the usual way duplicates enter a scraped dataset. Comparing a
    small thumbnail against its own mean is robust to both.
    """
    with Image.open(path) as im:
        small = im.convert("L").resize((size, size), Image.BILINEAR)
        arr = np.asarray(small, dtype=np.float64)
    bits = (arr > arr.mean()).flatten()
    return hashlib.md5(np.packbits(bits).tobytes()).hexdigest()


def find_duplicates(root: str | Path, splits=("train", "valid", "test")) -> dict:
    """Group images by content hash to expose duplicates within/across splits.

    Duplicates that straddle a split boundary are a leakage channel that a
    filename comparison cannot detect: the same photo saved under two names
    would pass a filename check and still put a test image in the training set.
    """
    root = Path(root)
    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for split in splits:
        for fname in list_split_files(root, split):
            h = image_content_hash(root / split / "images" / fname)
            buckets[h].append((split, fname))

    duplicate_groups = {h: items for h, items in buckets.items() if len(items) > 1}
    cross_split = {
        h: items
        for h, items in duplicate_groups.items()
        if len({s for s, _ in items}) > 1
    }
    return {
        "n_images": sum(len(v) for v in buckets.values()),
        "n_unique": len(buckets),
        "duplicate_groups": duplicate_groups,
        "cross_split_groups": cross_split,
    }


def grid_collision_report(
    root: str | Path, split: str, cfg: Config, grid_sizes=(7, 14, 28)
) -> dict[int, dict]:
    """How many objects each grid resolution can physically represent.

    This is the evidence behind the choice of S. A grid detector can encode at
    most one object per cell, so on a densely populated dataset the grid size
    puts a hard ceiling on achievable recall before a single weight is trained.
    """
    root = Path(root)
    report: dict[int, dict] = {}
    files = list_split_files(root, split)

    for S in grid_sizes:
        total = dropped_total = 0
        for fname in files:
            boxes_yolo, labels = parse_yolo_label(
                root / split / "labels" / (Path(fname).stem + ".txt")
            )
            boxes = yolo_to_xyxy(boxes_yolo, cfg.img_size, cfg.img_size)
            _, dropped = encode_grid_targets(
                boxes, labels, S, cfg.num_classes, cfg.img_size
            )
            total += len(boxes)
            dropped_total += dropped

        report[S] = {
            "grid_size": S,
            "cells": S * S,
            "objects": total,
            "unrepresentable": dropped_total,
            "loss_pct": 100.0 * dropped_total / total if total else 0.0,
            "max_recall": 1.0 - (dropped_total / total if total else 0.0),
        }
    return report
