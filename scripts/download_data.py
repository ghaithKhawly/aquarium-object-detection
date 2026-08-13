"""Download the Aquarium Combined dataset from Roboflow Universe.

The dataset is not shipped inside the submission archive - the project guide
asks for a source link and placement instructions instead of a bundled copy of
a public dataset.

Usage
-----
    # key from the environment (recommended)
    set ROBOFLOW_API_KEY=xxxxxxxx        # Windows
    export ROBOFLOW_API_KEY=xxxxxxxx     # Linux / macOS / Colab
    python scripts/download_data.py

    # or be prompted for it, without it appearing on screen or in history
    python scripts/download_data.py --prompt-key

The API key is never written to a file and never hardcoded. A key committed
into a notebook is a live credential that anyone reading the submission can
use, so it is read from the environment or typed at the prompt.

If the download fails (no key, no network, quota), see the manual instructions
printed on failure - a hand-placed copy of the dataset works exactly as well.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from getpass import getpass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, Config  # noqa: E402

EXPECTED_LAYOUT = """
Expected final layout (relative to the project root):

    data/
      train/  images/*.jpg   labels/*.txt
      valid/  images/*.jpg   labels/*.txt
      test/   images/*.jpg   labels/*.txt

Manual alternative
------------------
 1. Open https://universe.roboflow.com/brad-dwyer/aquarium-combined
 2. Choose version 2, "Download this Dataset", format **YOLOv8**
 3. Unzip it and move the train/, valid/ and test/ folders into data/
    so the paths match the layout above.
"""


def resolve_api_key(prompt: bool) -> str:
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if key:
        return key
    if prompt or sys.stdin.isatty():
        return getpass("Roboflow API key (input hidden): ").strip()
    raise SystemExit(
        "No API key found.\n"
        "Set ROBOFLOW_API_KEY in the environment, or rerun with --prompt-key.\n"
        "A free key is at https://app.roboflow.com/settings/api"
        + EXPECTED_LAYOUT
    )


def flatten_download(downloaded_root: Path, target: Path) -> None:
    """Move Roboflow's extracted folders into `data/`, whatever it named them."""
    target.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        src = downloaded_root / split
        dst = target / split
        if not src.exists():
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))

    # Keep the dataset YAML for reference; drop the now-empty download folder.
    for extra in downloaded_root.glob("*.yaml"):
        shutil.copy2(extra, target / extra.name)
    if downloaded_root.exists() and downloaded_root.resolve() != target.resolve():
        shutil.rmtree(downloaded_root, ignore_errors=True)


def verify(target: Path) -> bool:
    ok = True
    for split in ("train", "valid", "test"):
        images = target / split / "images"
        labels = target / split / "labels"
        if not images.is_dir() or not labels.is_dir():
            print(f"  MISSING  {split}/  (expected images/ and labels/)")
            ok = False
            continue
        n_img = len(list(images.iterdir()))
        n_lbl = len(list(labels.iterdir()))
        print(f"  {split:<6} {n_img:>4} images, {n_lbl:>4} label files")
        if n_img == 0:
            ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt-key", action="store_true",
                        help="ask for the API key interactively")
    parser.add_argument("--output", type=Path, default=DATA_DIR,
                        help=f"destination folder (default: {DATA_DIR})")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if data/ already looks complete")
    args = parser.parse_args()

    cfg = Config()
    target = Path(args.output)

    if not args.force and (target / "train" / "images").is_dir():
        print(f"Dataset already present at {target}:")
        if verify(target):
            print("\nNothing to do. Use --force to re-download.")
            return 0
        print("\n...but it looks incomplete. Re-downloading.")

    try:
        from roboflow import Roboflow
    except ImportError:
        print("The `roboflow` package is not installed.\n"
              "    pip install roboflow\n" + EXPECTED_LAYOUT)
        return 1

    api_key = resolve_api_key(args.prompt_key)

    print(f"Downloading {cfg.roboflow_workspace}/{cfg.dataset_slug} "
          f"v{cfg.roboflow_version} in YOLOv8 format...")
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(cfg.roboflow_workspace).project(cfg.dataset_slug)
        version = project.version(cfg.roboflow_version)
        dataset = version.download("yolov8", location=str(target / "_download"))
    except Exception as exc:
        print(f"\nDownload failed: {type(exc).__name__}: {exc}" + EXPECTED_LAYOUT)
        return 1

    flatten_download(Path(dataset.location), target)

    print(f"\nDataset ready at {target}")
    return 0 if verify(target) else 1


if __name__ == "__main__":
    raise SystemExit(main())
