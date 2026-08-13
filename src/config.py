"""Central configuration for the aquarium detection project.

Every tunable value used anywhere in the project is declared here so that a
single object fully describes an experiment. This is what makes runs
reproducible: `Config` is serialised next to each checkpoint, so a saved model
always travels with the exact settings that produced it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Project layout. All paths are resolved relative to the repository root so
# the project runs unchanged on Colab, Windows, or Linux with no path edits.
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REAL_WORLD_DIR = PROJECT_ROOT / "real_world_samples"

# The 7 classes of the Aquarium Combined dataset, in the exact order used by
# the YOLO label files (class id == index in this list).
CLASS_NAMES = [
    "fish",
    "jellyfish",
    "penguin",
    "puffin",
    "shark",
    "starfish",
    "stingray",
]
NUM_CLASSES = len(CLASS_NAMES)

# ImageNet channel statistics. Both models normalise with these values so the
# comparison is not confounded by a difference in input scaling (Model 2's
# torchvision backbone was pretrained under exactly these statistics).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Config:
    """Full experiment description."""

    # ---- Reproducibility ------------------------------------------------
    seed: int = 42
    deterministic: bool = True

    # ---- Shared data settings (identical for both models) ---------------
    img_size: int = 224
    dataset_slug: str = "aquarium-combined"
    roboflow_workspace: str = "brad-dwyer"
    roboflow_version: int = 2

    # ---- Augmentation (training split only) -----------------------------
    aug_hflip_prob: float = 0.5
    aug_color_jitter_prob: float = 0.5
    aug_brightness: float = 0.25
    aug_contrast: float = 0.25
    aug_saturation: float = 0.25
    aug_hue: float = 0.03
    aug_scale_translate_prob: float = 0.5
    aug_scale_range: tuple[float, float] = (0.85, 1.15)
    aug_translate_frac: float = 0.08

    # ---- Model 1: from-scratch grid detector ----------------------------
    grid_size: int = 14              # S x S output grid
    m1_epochs: int = 80
    m1_batch_size: int = 16
    m1_lr: float = 1e-3
    m1_weight_decay: float = 5e-4
    m1_lambda_coord: float = 5.0
    m1_lambda_obj: float = 1.0
    m1_lambda_noobj: float = 0.5
    m1_lambda_cls: float = 1.0
    m1_grad_clip: float = 5.0

    # ---- Model 2: Faster R-CNN transfer learning ------------------------
    m2_epochs: int = 20
    m2_batch_size: int = 8
    m2_lr: float = 5e-3
    m2_momentum: float = 0.9
    m2_weight_decay: float = 5e-4
    m2_lr_step_size: int = 5
    m2_lr_gamma: float = 0.5
    m2_trainable_backbone_layers: int = 3

    # ---- Early stopping (same policy for both models) -------------------
    early_stopping_patience: int = 12
    early_stopping_min_delta: float = 1e-4

    # ---- Inference / evaluation -----------------------------------------
    # The operating confidence threshold is *selected on the validation split*
    # (see notebook section 8) and then frozen before touching the test set.
    conf_threshold: float = 0.30
    nms_iou_threshold: float = 0.45
    eval_iou_threshold: float = 0.50
    max_detections: int = 100

    # ---- Latency benchmark ----------------------------------------------
    latency_warmup: int = 10
    latency_repeats: int = 50

    dataloader_workers: int = 2

    class_names: list[str] = field(default_factory=lambda: list(CLASS_NAMES))

    # ---------------------------------------------------------------------
    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # tuples survive a JSON round trip as lists; restore the ones we need
        for key in ("aug_scale_range",):
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
        return cls(**data)


def ensure_directories() -> None:
    """Create the output folders the project writes into."""
    for directory in (DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
