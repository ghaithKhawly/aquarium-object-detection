"""Training loops, checkpointing, inference, and latency benchmarking.

Both models are driven through the same policy - same early-stopping rule,
same checkpoint-selection criterion, same post-processing, same timing
protocol - so that the reported differences come from the architectures and
not from one model having been trained more carefully than the other.

Model selection
---------------
Checkpoints are selected on **validation mAP@0.5**, not on validation loss.
The two losses are not comparable (one is a weighted multi-task sum over a
grid, the other a sum of RPN and ROI head terms), and loss is in any case a
proxy: the epoch with the lowest validation loss is not necessarily the one
that detects best. Selecting on the metric the project actually reports also
avoids the failure mode where a model that started overfitting at epoch 5 is
saved at epoch 20 and then evaluated in its worst state.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from .boxes import non_max_suppression
from .config import Config
from .data import (
    DetectionDataset,
    GridDataset,
    decode_grid_predictions,
    detection_collate,
    grid_collate,
    load_ground_truth,
)
from .losses import GridDetectionLoss
from .metrics import ImagePrediction, evaluate_detections
from .models import build_model
from .utils import (
    Timer,
    hardware_report,
    make_generator,
    seed_worker,
)

# Detections below this score are discarded before anything else. AP needs the
# low-confidence tail to trace the full precision/recall curve, so the floor is
# far below the operating threshold - but not zero, which would keep tens of
# thousands of meaningless boxes per image and make evaluation crawl.
EVAL_SCORE_FLOOR = 0.01

__all__ = [
    "EarlyStopping",
    "build_dataloaders",
    "train_model",
    "predict_split",
    "benchmark_latency",
    "save_checkpoint",
    "load_checkpoint",
    "update_checkpoint_threshold",
]


# --------------------------------------------------------------------------
# Early stopping
# --------------------------------------------------------------------------
class EarlyStopping:
    """Stop when the monitored metric has not improved for `patience` epochs.

    Applied identically to both models. Without it, the smaller model needs
    many more epochs than the fine-tuned one and any fixed epoch budget would
    unfairly favour whichever model happens to suit it.
    """

    def __init__(self, patience: int = 12, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: float | None = None
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False

    def _improved(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def step(self, value: float, epoch: int) -> bool:
        """Returns True when this epoch is the new best."""
        if self._improved(value):
            self.best, self.best_epoch, self.counter = value, epoch, 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


# --------------------------------------------------------------------------
# Data loaders
# --------------------------------------------------------------------------
def build_dataloaders(kind: str, data_root: str | Path, cfg: Config):
    """Train/valid/test loaders for one model kind.

    Augmentation is enabled on the training split only. Validation and test
    must stay a fixed target - augmenting them would make the metric noisy and
    incomparable between epochs and between models.
    """
    kind = kind.lower()
    if kind in {"scratch", "model1", "grid"}:
        ds_cls, collate, batch = GridDataset, grid_collate, cfg.m1_batch_size
    else:
        ds_cls, collate, batch = DetectionDataset, detection_collate, cfg.m2_batch_size

    datasets = {
        "train": ds_cls(data_root, "train", cfg, augment=True),
        "valid": ds_cls(data_root, "valid", cfg, augment=False),
        "test": ds_cls(data_root, "test", cfg, augment=False),
    }

    loaders = {
        split: DataLoader(
            ds,
            batch_size=batch,
            shuffle=(split == "train"),
            collate_fn=collate,
            num_workers=cfg.dataloader_workers,
            worker_init_fn=seed_worker,
            generator=make_generator(cfg.seed) if split == "train" else None,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )
        for split, ds in datasets.items()
    }
    return datasets, loaders


# --------------------------------------------------------------------------
# Per-epoch passes
# --------------------------------------------------------------------------
def _run_grid_epoch(model, loader, criterion, optimizer, device, cfg) -> dict:
    is_train = optimizer is not None
    model.train(is_train)

    totals = {"loss": 0.0, "loc": 0.0, "obj": 0.0, "noobj": 0.0, "cls": 0.0}
    n_batches = 0

    with torch.set_grad_enabled(is_train):
        for images, targets, _ in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            predictions = model(images)
            loss, parts = criterion(predictions, targets)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Gradient clipping: the multi-task loss can spike when a
                # batch happens to contain a very dense image, and one
                # exploding step can undo many good ones.
                if cfg.m1_grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.m1_grad_clip)
                optimizer.step()

            for k in totals:
                totals[k] += parts[k]
            n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def _run_detection_epoch(model, loader, optimizer, device) -> dict:
    """One pass over the Faster R-CNN loaders.

    torchvision detection models only return their loss dictionary in
    `train()` mode, so validation also runs with `model.train()` but under
    `no_grad` and with no optimiser step. This does not corrupt the model:
    these backbones use `FrozenBatchNorm2d`, whose statistics are constant, so
    a forward pass in train mode updates nothing.
    """
    is_train = optimizer is not None
    model.train()

    totals: dict[str, float] = {}
    n_batches = 0

    with torch.set_grad_enabled(is_train):
        for images, targets, _ in loader:
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [
                {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
                for t in targets
            ]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()

            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach())
            for k, v in loss_dict.items():
                totals[k] = totals.get(k, 0.0) + float(v.detach())
            n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
@torch.no_grad()
def predict_split(
    model,
    kind: str,
    dataset,
    cfg: Config,
    device,
    score_floor: float = EVAL_SCORE_FLOOR,
) -> list[ImagePrediction]:
    """Run a model over a dataset and return NMS-ed detections per image.

    Both models leave this function through the *same* post-processing:
    `boxes.non_max_suppression` at `cfg.nms_iou_threshold`. torchvision has
    already applied its own class-wise NMS at that same threshold internally,
    so for Model 2 this second pass is idempotent - it changes nothing, and it
    means neither model can be accused of benefiting from a different
    suppression rule.
    """
    model.eval()
    kind = kind.lower()
    predictions: list[ImagePrediction] = []

    for idx in range(len(dataset)):
        if kind in {"scratch", "model1", "grid"}:
            image, _, meta = dataset[idx]
            raw = model(image.unsqueeze(0).to(device))[0]
            boxes, scores, labels = decode_grid_predictions(
                raw, img_size=cfg.img_size, conf_threshold=score_floor
            )
        else:
            image, _, meta = dataset[idx]
            out = model([image.to(device)])[0]
            boxes = out["boxes"].detach().cpu().numpy().astype(np.float64)
            scores = out["scores"].detach().cpu().numpy().astype(np.float64)
            # torchvision reserves 0 for background, so shift ids back down.
            labels = out["labels"].detach().cpu().numpy().astype(np.int64) - 1
            keep_floor = scores >= score_floor
            boxes, scores, labels = boxes[keep_floor], scores[keep_floor], labels[keep_floor]

        keep = non_max_suppression(
            boxes,
            scores,
            labels,
            iou_threshold=cfg.nms_iou_threshold,
            score_threshold=score_floor,
            max_detections=cfg.max_detections,
        )
        predictions.append(
            ImagePrediction(
                boxes=boxes[keep],
                scores=scores[keep],
                labels=labels[keep],
                image_id=meta["file"],
            )
        )

    return predictions


def _validation_map(model, kind, dataset, ground_truth, cfg, device) -> float:
    preds = predict_split(model, kind, dataset, cfg, device)
    result = evaluate_detections(
        preds,
        ground_truth,
        cfg.class_names,
        iou_threshold=cfg.eval_iou_threshold,
        conf_threshold=cfg.conf_threshold,
        compute_map_range=False,
    )
    return result.map_50


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------
def save_checkpoint(
    path: str | Path,
    model,
    kind: str,
    cfg: Config,
    epoch: int,
    metrics: dict,
    history: list | None = None,
) -> None:
    """Save weights together with everything needed to rebuild the model.

    The config and class names travel inside the checkpoint so that
    `scripts/predict.py` can reconstruct the exact architecture without being
    told anything but the file path. A bare `state_dict` would require the
    caller to already know the grid size and class count, which is precisely
    the kind of undocumented coupling that breaks a demo.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "kind": kind,
            "config": cfg.to_dict(),
            "class_names": cfg.class_names,
            "epoch": epoch,
            "metrics": metrics,
            "history": history or [],
            "hardware": hardware_report(),
            "format_version": 1,
        },
        path,
    )


def update_checkpoint_threshold(path: str | Path, threshold: float) -> None:
    """Write a calibrated confidence threshold back into a checkpoint.

    The operating point is chosen on the validation split during evaluation,
    not during training, so the value is not known when the checkpoint is
    first written. Storing it here keeps the checkpoint self-describing:
    `scripts/predict.py` then runs the demo at the calibrated threshold
    without being told it, instead of falling back to the config default.
    """
    path = Path(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt["config"]["conf_threshold"] = float(threshold)
    ckpt.setdefault("metrics", {})["selected_conf_threshold"] = float(threshold)
    torch.save(ckpt, path)


def load_checkpoint(path: str | Path, device=None, strict: bool = True):
    """Rebuild a trained model from disk. Returns (model, cfg, checkpoint)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {path}. Train first with "
            f"`python scripts/train.py --model <name>`, or download the "
            f"released weights (see README section 6)."
        )

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(**ckpt["config"])
    model = build_model(ckpt["kind"], cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    model.to(device).eval()
    return model, cfg, ckpt


# --------------------------------------------------------------------------
# Training driver
# --------------------------------------------------------------------------
def train_model(
    kind: str,
    data_root: str | Path,
    cfg: Config,
    device,
    checkpoint_path: str | Path,
    log_fn: Callable[[str], None] = print,
) -> dict:
    """Train one model end to end. Returns a history/summary dictionary."""
    kind = kind.lower()
    is_grid = kind in {"scratch", "model1", "grid"}

    datasets, loaders = build_dataloaders(kind, data_root, cfg)
    val_gt = load_ground_truth(data_root, "valid", cfg)

    model = build_model(kind, cfg).to(device)

    if is_grid:
        epochs = cfg.m1_epochs
        criterion = GridDetectionLoss(
            lambda_coord=cfg.m1_lambda_coord,
            lambda_obj=cfg.m1_lambda_obj,
            lambda_noobj=cfg.m1_lambda_noobj,
            lambda_cls=cfg.m1_lambda_cls,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.m1_lr, weight_decay=cfg.m1_weight_decay
        )
        # Cosine annealing: a high initial rate to escape the poor random
        # initialisation, decaying smoothly so late epochs refine rather than
        # bounce around the minimum.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        epochs = cfg.m2_epochs
        criterion = None
        params = [p for p in model.parameters() if p.requires_grad]
        # SGD with momentum, as used in the Faster R-CNN paper and the
        # torchvision reference recipe; Adam tends to disturb pretrained
        # detection backbones more than it helps at this dataset size.
        optimizer = torch.optim.SGD(
            params, lr=cfg.m2_lr, momentum=cfg.m2_momentum, weight_decay=cfg.m2_weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.m2_lr_step_size, gamma=cfg.m2_lr_gamma
        )
        # Match the shared post-processing settings.
        model.roi_heads.nms_thresh = cfg.nms_iou_threshold
        model.roi_heads.score_thresh = EVAL_SCORE_FLOOR
        model.roi_heads.detections_per_img = cfg.max_detections

    stopper = EarlyStopping(
        patience=cfg.early_stopping_patience,
        min_delta=cfg.early_stopping_min_delta,
        mode="max",
    )

    history: list[dict] = []
    best_state = None
    best_metrics: dict = {}

    log_fn(f"Training {kind} for up to {epochs} epochs on {device}")
    total_timer = Timer()
    total_timer.__enter__()

    for epoch in range(1, epochs + 1):
        with Timer() as epoch_timer:
            if is_grid:
                train_stats = _run_grid_epoch(
                    model, loaders["train"], criterion, optimizer, device, cfg
                )
                val_stats = _run_grid_epoch(
                    model, loaders["valid"], criterion, None, device, cfg
                )
            else:
                train_stats = _run_detection_epoch(model, loaders["train"], optimizer, device)
                val_stats = _run_detection_epoch(model, loaders["valid"], None, device)

            scheduler.step()
            val_map = _validation_map(model, kind, datasets["valid"], val_gt, cfg, device)

        record = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "val_map50": val_map,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": epoch_timer.elapsed,
            **{f"train_{k}": v for k, v in train_stats.items() if k != "loss"},
        }
        history.append(record)

        is_best = stopper.step(val_map, epoch)
        if is_best:
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = {"epoch": epoch, "val_map50": val_map, "val_loss": val_stats["loss"]}

        log_fn(
            f"  epoch {epoch:3d}/{epochs} | train {train_stats['loss']:.4f} | "
            f"val {val_stats['loss']:.4f} | val mAP@0.5 {val_map:.4f}"
            f"{'  <- best' if is_best else ''} | {epoch_timer.elapsed:.1f}s"
        )

        if stopper.should_stop:
            log_fn(
                f"  early stop at epoch {epoch}: no improvement for "
                f"{cfg.early_stopping_patience} epochs "
                f"(best epoch {stopper.best_epoch}, mAP {stopper.best:.4f})"
            )
            break

    total_timer.__exit__()

    # Restore the best weights before saving, so the checkpoint on disk is the
    # best model this run produced - not whatever state the last epoch left.
    if best_state is not None:
        model.load_state_dict(best_state)

    save_checkpoint(
        checkpoint_path,
        model,
        kind,
        cfg,
        epoch=best_metrics.get("epoch", epochs),
        metrics=best_metrics,
        history=history,
    )

    log_fn(
        f"Finished {kind}: {total_timer.elapsed:.1f}s total, best epoch "
        f"{best_metrics.get('epoch')} (val mAP@0.5 {best_metrics.get('val_map50', 0):.4f}) "
        f"-> {checkpoint_path}"
    )

    return {
        "kind": kind,
        "history": history,
        "best": best_metrics,
        "train_seconds": total_timer.elapsed,
        "epochs_run": len(history),
        "checkpoint": str(checkpoint_path),
        "hardware": hardware_report(),
    }


# --------------------------------------------------------------------------
# Latency
# --------------------------------------------------------------------------
@torch.no_grad()
def benchmark_latency(model, kind: str, dataset, cfg: Config, device) -> dict:
    """Single-image inference latency, measured identically for both models.

    Protocol, and why each part is there:
      * warm-up iterations are discarded - the first calls pay for CUDA
        context creation, kernel autotuning and lazy memory allocation, and
        including them would overstate latency by an order of magnitude;
      * every measurement is bracketed by `torch.cuda.synchronize()` (inside
        `Timer`), because CUDA launches are asynchronous and an unsynchronised
        clock measures queueing, not computation;
      * batch size is fixed at 1, the deployment condition this project cares
        about;
      * the median and 95th percentile are reported, not just the mean, since
        a single stall would dominate a mean over a few dozen samples.

    Post-processing (decode + NMS) is included: it is part of the cost of
    getting a detection out of the system, and excluding it would flatter
    whichever model does more work outside the network.
    """
    model.eval()
    is_grid = kind.lower() in {"scratch", "model1", "grid"}
    n = len(dataset)
    if n == 0:
        raise ValueError("cannot benchmark on an empty dataset")

    def one_pass(idx: int) -> None:
        image = dataset[idx][0]
        if is_grid:
            raw = model(image.unsqueeze(0).to(device))[0]
            boxes, scores, labels = decode_grid_predictions(
                raw, img_size=cfg.img_size, conf_threshold=cfg.conf_threshold
            )
        else:
            out = model([image.to(device)])[0]
            boxes = out["boxes"].detach().cpu().numpy()
            scores = out["scores"].detach().cpu().numpy()
            labels = out["labels"].detach().cpu().numpy() - 1
            keep_floor = scores >= cfg.conf_threshold
            boxes, scores, labels = boxes[keep_floor], scores[keep_floor], labels[keep_floor]
        non_max_suppression(
            boxes, scores, labels,
            iou_threshold=cfg.nms_iou_threshold,
            score_threshold=cfg.conf_threshold,
            max_detections=cfg.max_detections,
        )

    for i in range(cfg.latency_warmup):
        one_pass(i % n)

    timings: list[float] = []
    for i in range(cfg.latency_repeats):
        with Timer() as t:
            one_pass(i % n)
        timings.append(t.elapsed)

    arr = np.asarray(timings, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean() * 1000),
        "median_ms": float(np.median(arr) * 1000),
        "p95_ms": float(np.percentile(arr, 95) * 1000),
        "std_ms": float(arr.std() * 1000),
        "fps": float(1.0 / np.median(arr)),
        "samples": int(len(arr)),
        "device": str(device),
        "batch_size": 1,
    }
