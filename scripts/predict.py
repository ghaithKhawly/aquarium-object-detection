"""Run detection on new images using saved weights. No training required.

This is the demonstration entry point: it loads a checkpoint, runs inference
on images it has never seen, and writes an annotated figure. Nothing here
touches the training pipeline or the dataset, so it works on a fresh machine
given only the checkpoint files.

Usage
-----
    # both models on the bundled real-world samples
    python scripts/predict.py

    # one model, your own folder or a single file
    python scripts/predict.py --model scratch --images path/to/images
    python scripts/predict.py --images photo.jpg --conf 0.4

Outputs
-------
    results/predictions/<name>_comparison.png   side-by-side annotated images
    results/predictions/detections.json         every detection, machine-readable
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402
from PIL import Image  # noqa: E402

from src.boxes import non_max_suppression  # noqa: E402
from src.config import (  # noqa: E402
    CHECKPOINT_DIR, REAL_WORLD_DIR, RESULTS_DIR, ensure_directories,
)
from src.data import IMAGE_EXTENSIONS, decode_grid_predictions  # noqa: E402
from src.engine import EVAL_SCORE_FLOOR, load_checkpoint  # noqa: E402
from src.utils import Timer, save_json, set_seed  # noqa: E402
from src import viz  # noqa: E402

MODEL_LABELS = {
    "scratch": "Model 1 (Scratch Grid)",
    "fasterrcnn": "Model 2 (Faster R-CNN)",
}


def collect_images(path: Path) -> list[Path]:
    """Accept a single file, a folder, or a folder of categorised subfolders."""
    if path.is_file():
        return [path]
    if not path.exists():
        raise SystemExit(f"No such path: {path}")
    files = sorted(
        p for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise SystemExit(
            f"No images found under {path}.\n"
            f"Put a few .jpg/.png files there - see {REAL_WORLD_DIR / 'README.md'}."
        )
    return files


def load_image(path: Path, img_size: int):
    """Returns (tensor in [0,1], displayable array, original size)."""
    with Image.open(path) as im:
        original = im.convert("RGB")
        original_size = original.size
        resized = original.resize((img_size, img_size), Image.BILINEAR)
    tensor = TF.to_tensor(resized)
    return tensor, np.asarray(resized), original_size


@torch.no_grad()
def detect(model, kind: str, tensor: torch.Tensor, cfg, device, conf: float):
    """One image through one model. Returns (boxes, scores, labels, seconds)."""
    with Timer() as timer:
        if kind == "scratch":
            raw = model(tensor.unsqueeze(0).to(device))[0]
            boxes, scores, labels = decode_grid_predictions(
                raw, img_size=cfg.img_size, conf_threshold=EVAL_SCORE_FLOOR
            )
        else:
            out = model([tensor.to(device)])[0]
            boxes = out["boxes"].cpu().numpy().astype(np.float64)
            scores = out["scores"].cpu().numpy().astype(np.float64)
            labels = out["labels"].cpu().numpy().astype(np.int64) - 1

        keep = non_max_suppression(
            boxes, scores, labels,
            iou_threshold=cfg.nms_iou_threshold,
            score_threshold=conf,
            max_detections=cfg.max_detections,
        )
    return boxes[keep], scores[keep], labels[keep], timer.elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["scratch", "fasterrcnn", "both"], default="both")
    parser.add_argument("--images", type=Path, default=REAL_WORLD_DIR)
    parser.add_argument("--checkpoints", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--conf", type=float, default=None,
                        help="confidence threshold (default: the value stored in the checkpoint)")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "predictions")
    parser.add_argument("--max-images", type=int, default=12)
    args = parser.parse_args()

    ensure_directories()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    kinds = ["scratch", "fasterrcnn"] if args.model == "both" else [args.model]

    loaded = {}
    for kind in kinds:
        path = Path(args.checkpoints) / f"{kind}_best.pt"
        model, cfg, ckpt = load_checkpoint(path, device)
        conf = args.conf if args.conf is not None else cfg.conf_threshold
        loaded[kind] = (model, cfg, conf)
        print(f"loaded {MODEL_LABELS[kind]}: epoch {ckpt.get('epoch')}, "
              f"val mAP@0.5 {ckpt.get('metrics', {}).get('val_map50', float('nan')):.4f}, "
              f"conf {conf:.2f}")

    images = collect_images(args.images)[: args.max_images]
    print(f"\nrunning on {len(images)} image(s) from {args.images}\n")

    records = []
    panels = []

    for image_path in images:
        tensor, display, original_size = load_image(
            image_path, next(iter(loaded.values()))[1].img_size
        )
        # The folder an image sits in labels its difficulty category, so the
        # required "normal / difficult / failure-oriented" grouping comes
        # straight from the directory layout instead of a manual note.
        category = image_path.parent.name if image_path.parent != args.images else "uncategorised"

        row = {"image": display, "name": f"{image_path.name}\n[{category}]", "detections": {}}
        for kind in kinds:
            model, cfg, conf = loaded[kind]
            boxes, scores, labels, seconds = detect(model, kind, tensor, cfg, device, conf)
            row["detections"][kind] = (boxes, scores, labels)
            records.append({
                "image": str(image_path),
                "category": category,
                "model": MODEL_LABELS[kind],
                "original_size": list(original_size),
                "inference_seconds": seconds,
                "n_detections": int(len(boxes)),
                "detections": [
                    {
                        "class": cfg.class_names[int(l)] if 0 <= int(l) < len(cfg.class_names) else f"id{int(l)}",
                        "score": round(float(s), 4),
                        "box_xyxy_224": [round(float(v), 1) for v in b],
                    }
                    for b, s, l in zip(boxes, scores, labels)
                ],
            })
            print(f"  {image_path.name:<42} {MODEL_LABELS[kind]:<24} "
                  f"{len(boxes):>3} detections  {seconds*1000:6.1f} ms")
        panels.append(row)

    # ---- figure ----------------------------------------------------------
    class_names = next(iter(loaded.values()))[1].class_names
    n_cols = len(kinds)
    fig, axes = plt.subplots(len(panels), n_cols,
                             figsize=(6.2 * n_cols, 5.4 * len(panels)), squeeze=False)

    for r, row in enumerate(panels):
        for c, kind in enumerate(kinds):
            boxes, scores, labels = row["detections"][kind]
            ax = axes[r][c]
            viz.draw_boxes(ax, row["image"], boxes, labels, scores, class_names,
                           color=viz.MODEL_COLORS[MODEL_LABELS[kind]])
            title = MODEL_LABELS[kind] if r == 0 else ""
            ax.set_title(f"{title}\n{len(boxes)} detections", fontsize=10,
                         weight="bold" if r == 0 else "normal")
            if c == 0:
                ax.text(-0.03, 0.5, row["name"], transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=8)

    fig.suptitle("Inference on unseen real-world images (from saved weights)",
                 fontsize=13, y=1.002)
    fig.tight_layout()
    out_fig = args.out / "real_world_comparison.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    save_json(records, args.out / "detections.json")

    print(f"\n  {out_fig}")
    print(f"  {args.out / 'detections.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
