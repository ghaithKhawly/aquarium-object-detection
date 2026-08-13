"""Reproducibility, device, timing, and small I/O helpers."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed every source of randomness the project touches.

    `random` and `numpy` cover sampling and shuffling in our own code;
    `torch.manual_seed` covers weight initialisation and dropout;
    `cuda.manual_seed_all` covers GPU kernels. Without the torch seeds a rerun
    produces different weights and therefore different metrics, which would
    make the reported comparison irreproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # cudnn picks the fastest convolution algorithm by benchmarking, which
        # is nondeterministic. Trading a little speed for exact repeatability
        # is the right call for a project that must be defended.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker seeding, so augmentation is reproducible."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = 42) -> torch.Generator:
    """Generator handed to DataLoader so shuffling order is reproducible."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# --------------------------------------------------------------------------
# Device / hardware
# --------------------------------------------------------------------------
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hardware_report() -> dict[str, Any]:
    """Machine description recorded alongside every result.

    Requirement 7 of the project guide asks for the hardware to be documented;
    timing numbers are meaningless without it.
    """
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        report["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        report["gpu_memory_gb"] = round(props.total_memory / 1024**3, 2)
        report["cuda_version"] = torch.version.cuda
    else:
        report["gpu_name"] = "CPU only"
        try:
            report["cpu"] = platform.processor() or "unknown"
        except Exception:
            report["cpu"] = "unknown"
    return report


def format_hardware(report: dict[str, Any] | None = None) -> str:
    r = report or hardware_report()
    device = r.get("gpu_name", "CPU only")
    mem = f", {r['gpu_memory_gb']} GB" if "gpu_memory_gb" in r else ""
    return f"{device}{mem} | torch {r['torch']} | Python {r['python']}"


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------
class Timer:
    """Context-manager timer that is GPU-aware.

    CUDA kernels are asynchronous: without `torch.cuda.synchronize()` the
    clock stops before the GPU has finished, and inference looks far faster
    than it is. Every timing number in this project goes through here.
    """

    def __init__(self, label: str = "", verbose: bool = False):
        self.label = label
        self.verbose = verbose
        self.elapsed = 0.0

    def _sync(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def __enter__(self) -> "Timer":
        self._sync()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self._sync()
        self.elapsed = time.perf_counter() - self._start
        if self.verbose:
            print(f"[{self.label}] {self.elapsed:.3f} s")


@contextmanager
def timed(label: str, sink: dict | None = None):
    """`with timed('train', results):` records results['train'] = seconds."""
    with Timer(label) as t:
        yield t
    if sink is not None:
        sink[label] = t.elapsed


# --------------------------------------------------------------------------
# Model introspection
# --------------------------------------------------------------------------
def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def model_size_mb(model: torch.nn.Module) -> float:
    """On-disk footprint in MB (parameters + buffers, fp32)."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / 1024**2


def file_size_mb(path: str | Path) -> float:
    return Path(path).stat().st_size / 1024**2


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
