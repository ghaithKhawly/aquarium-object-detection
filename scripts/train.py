"""Train one or both detectors.

Usage
-----
    python scripts/train.py --model scratch
    python scripts/train.py --model fasterrcnn
    python scripts/train.py --model both

    # quick smoke test that the whole pipeline runs (2 epochs each)
    python scripts/train.py --model both --epochs 2

Every run writes:
    checkpoints/<model>_best.pt      best weights + config + history
    results/<model>_training.json    per-epoch history and timings
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.config import CHECKPOINT_DIR, DATA_DIR, RESULTS_DIR, Config, ensure_directories  # noqa: E402
from src.engine import train_model  # noqa: E402
from src.utils import format_hardware, save_json, set_seed  # noqa: E402

MODEL_KEYS = {"scratch": "scratch", "fasterrcnn": "fasterrcnn"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["scratch", "fasterrcnn", "both"],
                        default="both", help="which model to train")
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=None,
                        help="override the configured epoch budget for both models")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", type=Path, default=CHECKPOINT_DIR)
    args = parser.parse_args()

    ensure_directories()
    cfg = Config()

    if args.seed is not None:
        cfg.seed = args.seed
    if args.epochs is not None:
        cfg.m1_epochs = cfg.m2_epochs = args.epochs
    if args.batch_size is not None:
        cfg.m1_batch_size = cfg.m2_batch_size = args.batch_size
    if args.workers is not None:
        cfg.dataloader_workers = args.workers

    set_seed(cfg.seed, cfg.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 74)
    print("Aquarium detection - training")
    print(f"  hardware : {format_hardware()}")
    print(f"  data     : {args.data}")
    print(f"  seed     : {cfg.seed} (deterministic={cfg.deterministic})")
    print("=" * 74)

    if device.type == "cpu":
        print("\n  WARNING: no GPU detected. Faster R-CNN fine-tuning on CPU takes\n"
              "  hours. Use Colab with a T4, or pass --epochs 2 for a smoke test.\n")

    targets = ["scratch", "fasterrcnn"] if args.model == "both" else [args.model]
    summaries = {}

    for kind in targets:
        # Reseed before each model so the two runs are independent and
        # reproducible regardless of whether they are trained together.
        set_seed(cfg.seed, cfg.deterministic)

        print(f"\n--- {kind} " + "-" * (68 - len(kind)))
        checkpoint = Path(args.out) / f"{kind}_best.pt"
        summary = train_model(kind, args.data, cfg, device, checkpoint)
        summaries[kind] = summary

        save_json(summary, RESULTS_DIR / f"{kind}_training.json")

    print("\n" + "=" * 74)
    for kind, summary in summaries.items():
        best = summary["best"]
        print(f"  {kind:<12} best epoch {best.get('epoch'):>3} | "
              f"val mAP@0.5 {best.get('val_map50', 0):.4f} | "
              f"{summary['train_seconds']:.0f}s over {summary['epochs_run']} epochs")
    print("=" * 74)
    print("\nNext: python scripts/evaluate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
