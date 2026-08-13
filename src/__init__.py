"""Aquarium object detection - from-scratch grid detector vs. Faster R-CNN.

Neural Networks Laboratory final project, 2025-2026.

Module map
----------
    config              experiment settings, paths, class names
    utils               seeding, device, GPU-aware timing, checkpoint I/O
    boxes               IoU, NMS, coordinate conversion  (NumPy, no torch)
    data                datasets, augmentation, grid encoding/decoding
    eda                 dataset exploration and quality auditing
    models              ScratchGridDetector and the Faster R-CNN builder
    losses              multi-task grid detection loss
    engine              training loops, early stopping, inference, latency
    metrics             per-class AP, mAP, P/R/F1, confusion matrix
    failure_analysis    error taxonomy and aggregation
    viz                 every figure in the report
"""

__version__ = "1.0.0"

from . import (  # noqa: F401
    boxes,
    config,
    data,
    eda,
    engine,
    failure_analysis,
    losses,
    metrics,
    models,
    utils,
    viz,
)

__all__ = [
    "boxes", "config", "data", "eda", "engine", "failure_analysis",
    "losses", "metrics", "models", "utils", "viz",
]
