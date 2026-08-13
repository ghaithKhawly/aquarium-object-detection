"""Dataset exploration and quality auditing.

Covers requirement 3 of the project guide: class balance, sample quality,
corrupted files, duplicates, label quality, and representative examples.

Every function returns plain data structures; plotting lives in `viz.py` so
the analysis can be reused from a script, a test, or the notebook.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .config import Config
from .data import list_split_files, parse_yolo_label

__all__ = [
    "split_overview",
    "class_distribution",
    "image_quality_report",
    "box_statistics",
    "label_sanity_report",
    "leakage_report",
]

SPLITS = ("train", "valid", "test")


def split_overview(root: str | Path, cfg: Config, splits=SPLITS) -> dict:
    """Image and object counts per split, plus the realised split ratios."""
    root = Path(root)
    rows = {}
    for split in splits:
        files = list_split_files(root, split)
        n_objects = 0
        n_empty = 0
        for fname in files:
            _, labels = parse_yolo_label(root / split / "labels" / (Path(fname).stem + ".txt"))
            n_objects += len(labels)
            n_empty += int(len(labels) == 0)
        rows[split] = {
            "images": len(files),
            "objects": n_objects,
            "empty_images": n_empty,
            "objects_per_image": n_objects / len(files) if files else 0.0,
        }

    total_images = sum(r["images"] for r in rows.values())
    for split, r in rows.items():
        r["share_pct"] = 100.0 * r["images"] / total_images if total_images else 0.0

    return {"per_split": rows, "total_images": total_images}


def class_distribution(root: str | Path, cfg: Config, splits=SPLITS) -> dict:
    """Object counts per class per split.

    Class balance drives how the results must be read: with a long-tailed
    distribution, a high micro-averaged score can hide a class the model never
    detects at all, which is why the evaluation reports per-class AP as well
    as the mean.
    """
    root = Path(root)
    names = cfg.class_names
    table: dict[str, Counter] = {}

    for split in splits:
        counts: Counter = Counter()
        for fname in list_split_files(root, split):
            _, labels = parse_yolo_label(root / split / "labels" / (Path(fname).stem + ".txt"))
            for label in labels:
                if 0 <= int(label) < len(names):
                    counts[names[int(label)]] += 1
                else:
                    counts[f"INVALID_ID_{int(label)}"] += 1
        table[split] = counts

    totals = Counter()
    for counts in table.values():
        totals.update(counts)

    max_count = max(totals.values()) if totals else 0
    min_count = min((totals[n] for n in names if totals[n] > 0), default=0)

    return {
        "per_split": {s: dict(c) for s, c in table.items()},
        "totals": dict(totals),
        "imbalance_ratio": (max_count / min_count) if min_count else float("inf"),
        "rarest": min(names, key=lambda n: totals.get(n, 0)) if names else None,
        "most_common": max(names, key=lambda n: totals.get(n, 0)) if names else None,
    }


def image_quality_report(root: str | Path, splits=SPLITS) -> dict:
    """Resolution statistics and unreadable-file detection.

    Every file is fully decoded rather than just opened: PIL is lazy, so a
    truncated JPEG will open without complaint and only fail later, in the
    middle of training.
    """
    root = Path(root)
    per_split = {}
    all_sizes: list[tuple[int, int]] = []
    corrupted: list[str] = []

    for split in splits:
        sizes = []
        for fname in list_split_files(root, split):
            path = root / split / "images" / fname
            try:
                with Image.open(path) as im:
                    im.load()                      # force full decode
                    sizes.append(im.size)
                    mode = im.mode
                if mode not in {"RGB", "L", "RGBA", "P"}:
                    corrupted.append(f"{split}/{fname} (unexpected mode {mode})")
            except Exception as exc:
                corrupted.append(f"{split}/{fname} ({type(exc).__name__}: {exc})")

        if sizes:
            widths, heights = zip(*sizes)
            per_split[split] = {
                "count": len(sizes),
                "width_min": min(widths), "width_max": max(widths),
                "width_mean": float(np.mean(widths)),
                "height_min": min(heights), "height_max": max(heights),
                "height_mean": float(np.mean(heights)),
                "unique_resolutions": len(set(sizes)),
            }
            all_sizes.extend(sizes)

    return {
        "per_split": per_split,
        "corrupted": corrupted,
        "n_corrupted": len(corrupted),
        "all_sizes": all_sizes,
    }


def box_statistics(root: str | Path, cfg: Config, split: str = "train") -> dict:
    """Object size distribution, in the resized 224x224 evaluation frame.

    Object scale is the single most predictive property for detector
    behaviour. Following the COCO convention, an object is 'small' below
    32x32 pixels and 'large' above 96x96; the proportion of small objects
    largely determines how much the coarse grid of Model 1 can achieve.
    """
    root = Path(root)
    img_size = cfg.img_size

    areas, widths, heights, per_image = [], [], [], []

    for fname in list_split_files(root, split):
        boxes, _ = parse_yolo_label(root / split / "labels" / (Path(fname).stem + ".txt"))
        per_image.append(len(boxes))
        for _, _, w, h in boxes:
            pw, ph = w * img_size, h * img_size
            widths.append(pw)
            heights.append(ph)
            areas.append(pw * ph)

    areas_arr = np.asarray(areas, dtype=np.float64)
    small = int((areas_arr < 32**2).sum())
    medium = int(((areas_arr >= 32**2) & (areas_arr < 96**2)).sum())
    large = int((areas_arr >= 96**2).sum())
    total = max(len(areas_arr), 1)

    return {
        "n_boxes": int(len(areas_arr)),
        "boxes_per_image": {
            "mean": float(np.mean(per_image)) if per_image else 0.0,
            "median": float(np.median(per_image)) if per_image else 0.0,
            "max": int(np.max(per_image)) if per_image else 0,
            "min": int(np.min(per_image)) if per_image else 0,
        },
        "width_px": {"mean": float(np.mean(widths)) if widths else 0.0,
                     "median": float(np.median(widths)) if widths else 0.0},
        "height_px": {"mean": float(np.mean(heights)) if heights else 0.0,
                      "median": float(np.median(heights)) if heights else 0.0},
        "size_buckets": {
            "small_pct": 100.0 * small / total,
            "medium_pct": 100.0 * medium / total,
            "large_pct": 100.0 * large / total,
            "small": small, "medium": medium, "large": large,
        },
        "areas": areas,
        "per_image_counts": per_image,
    }


def label_sanity_report(root: str | Path, cfg: Config, splits=SPLITS) -> dict:
    """Structural validation of every annotation file.

    Catches the label defects that would otherwise surface as silent accuracy
    loss: class ids outside the declared range, coordinates outside [0,1],
    zero-area boxes, and images with no annotation file at all.
    """
    root = Path(root)
    issues = {
        "invalid_class_id": [],
        "out_of_range_coords": [],
        "degenerate_boxes": [],
        "missing_label_file": [],
        "unparseable": [],
    }
    n_boxes = 0

    for split in splits:
        for fname in list_split_files(root, split):
            label_path = root / split / "labels" / (Path(fname).stem + ".txt")
            ref = f"{split}/{fname}"

            if not label_path.exists():
                issues["missing_label_file"].append(ref)
                continue
            try:
                boxes, labels = parse_yolo_label(label_path)
            except ValueError as exc:
                issues["unparseable"].append(f"{ref}: {exc}")
                continue

            n_boxes += len(boxes)
            for i, (cx, cy, w, h) in enumerate(boxes):
                cls = int(labels[i])
                if not (0 <= cls < cfg.num_classes):
                    issues["invalid_class_id"].append(f"{ref} box {i}: id {cls}")
                if not all(-1e-6 <= v <= 1 + 1e-6 for v in (cx, cy, w, h)):
                    issues["out_of_range_coords"].append(
                        f"{ref} box {i}: ({cx:.3f},{cy:.3f},{w:.3f},{h:.3f})"
                    )
                if w <= 0 or h <= 0 or w * h * cfg.img_size**2 < 4:
                    issues["degenerate_boxes"].append(f"{ref} box {i}: {w:.4f}x{h:.4f}")

    return {
        "n_boxes_checked": n_boxes,
        "issues": issues,
        "n_issues": sum(len(v) for v in issues.values()),
        "clean": sum(len(v) for v in issues.values()) == 0,
    }


def leakage_report(root: str | Path, splits=SPLITS) -> dict:
    """Filename-level overlap between splits.

    This is the cheap check. The stronger content-level check
    (`data.find_duplicates`) catches the same photograph saved under two
    different names, which this cannot see.
    """
    root = Path(root)
    sets = {s: set(list_split_files(root, s)) for s in splits}
    pairs = {}
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            overlap = sets[a] & sets[b]
            pairs[f"{a}|{b}"] = sorted(overlap)

    return {
        "sizes": {s: len(v) for s, v in sets.items()},
        "overlaps": {k: len(v) for k, v in pairs.items()},
        "overlap_files": {k: v for k, v in pairs.items() if v},
        "clean": all(len(v) == 0 for v in pairs.values()),
    }
