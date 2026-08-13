# Aquarium Object Detection — From-Scratch Grid Detector vs. Faster R-CNN

**Neural Networks Laboratory — Final Project, Academic Year 2025–2026**
Instructor: ENG. Samma Al Muhammad

| | |
|---|---|
| **Section** | `نسيت` |
| **Student 1** | `Ammar Alzoubi` |
| **Student 2** | `Ghaith Khawly` |
| **Main entry file** | [`main_project.ipynb`](main_project.ipynb) |
| **Demo entry point** | `python scripts/predict.py` (inference from saved weights, no training) |

---

## 1. Objective

Detect and classify seven aquatic species — *fish, jellyfish, penguin, puffin,
shark, starfish, stingray* — in underwater photographs, and use that task to
compare two fundamentally different detector designs under controlled
conditions.

**Inputs:** RGB images of arbitrary resolution, resized to 224×224.
**Outputs:** a set of bounding boxes, each with a class label and a confidence.
**Users:** aquarium monitoring operators and marine-survey researchers who
currently annotate footage by hand.
**Operating mode:** offline batch analysis, with latency measured so the
real-time feasibility of each model can be judged rather than assumed.

### The two approaches

| | **Model 1 — Scratch Grid Detector** | **Model 2 — Faster R-CNN** |
|---|---|---|
| Paradigm | single-shot, anchor-free, dense grid | two-stage: propose, then classify |
| Weights | random initialisation (He) | COCO-pretrained ResNet-50 + FPN |
| Written | entirely in this repository | torchvision, fine-tuned |
| Modern technique | — | ✅ transfer learning / fine-tuning |

Model 2 supplies the "modern course technique" the guide requires. Model 1
exists so the comparison has a meaningful baseline whose every component —
architecture, target encoding, loss, IoU, NMS, decoding — is implemented and
understood rather than imported.

**Why this pairing.** Comparing two pretrained detectors would measure library
defaults. Comparing a from-scratch dense detector against a fine-tuned
two-stage one isolates the two variables the project is actually about: what
pretraining buys you, and what it costs you at inference time.

---

## 2. Environment and installation

Developed on Google Colab (Tesla T4) and verified on Windows 11 + CPU.
Python 3.10 or newer.

```bash
git clone <your-repo-or-unzip-the-archive>
cd finalproject
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

On Colab everything except `roboflow` is preinstalled; the notebook's first
cell handles it.

Verify the installation — this needs no dataset and no GPU:

```bash
pytest tests/ -v
```

---

## 3. Dataset

| | |
|---|---|
| **Name** | Aquarium Combined |
| **Source** | https://universe.roboflow.com/brad-dwyer/aquarium-combined (version 2) |
| **License** | CC BY 4.0 |
| **Size** | 638 images, 7 classes, 4,817 annotated objects |
| **Annotation** | axis-aligned bounding boxes, YOLO `.txt` format |
| **Resolution** | 576–1024 px, mixed aspect ratios |
| **Split** | 448 train / 127 valid / 63 test (70 / 20 / 10), as published |

The dataset is **not** included in the archive — the guide asks for a source
link rather than a bundled copy of a public dataset.

```bash
# option A: automatic (needs a free Roboflow API key)
export ROBOFLOW_API_KEY=your_key_here     # Windows: set ROBOFLOW_API_KEY=...
python scripts/download_data.py

# option B: manual
#   1. open the URL above, version 2, "Download this Dataset", format YOLOv8
#   2. unzip and move train/, valid/, test/ into data/
```

Either way the result must look like this:

```
data/
  train/  images/*.jpg   labels/*.txt
  valid/  images/*.jpg   labels/*.txt
  test/   images/*.jpg   labels/*.txt
```

> **The API key is never stored in this repository.** It is read from the
> environment or typed at a hidden prompt. A key committed into a notebook is
> a live credential visible to everyone who opens the submission.

### Why this dataset suits the problem

Measured properties (all produced by `main_project.ipynb` section 3, saved to
`results/eda_report.json`):

| Property | Value | Why it matters |
|---|---|---|
| Objects per image | mean **7.4**, median 4, max **56** | Crowded scenes exercise occlusion and NMS in a single image |
| Object scale | **76.1% small** (<32²px), 21.7% medium, 2.2% large | Median object is only **23×16 px** at 224 — this is the property that most separates the two architectures |
| Class imbalance | **23 : 1** (fish 55.4% → starfish 2.4%) | Forces per-class AP reporting; a micro average would be almost entirely "fish" |
| Corrupted files | 0 of 638 | — |
| Duplicate images | 0 (content-hash checked) | No hidden cross-split leakage |
| Label defects | 5 degenerate boxes (<4 px² area) | Small but real; reported rather than silently dropped |

It is also small enough to train two detectors within a lab compute budget, and
genuinely difficult: turbid water, glass reflections and motion blur.

---

## 4. Quick start

```bash
python scripts/download_data.py     # 1. get the data
python scripts/train.py --model both  # 2. train (see timings below)
python scripts/evaluate.py          # 3. all metrics + figures
python scripts/predict.py           # 4. demo on unseen images
```

Or open [`main_project.ipynb`](main_project.ipynb) and run it top to bottom —
it does all four and explains each step.

**Smoke test** (verifies the whole pipeline in a couple of minutes, on CPU,
without producing meaningful accuracy):

```bash
python scripts/train.py --model both --epochs 2 --workers 0
```

---

## 5. Training

```bash
python scripts/train.py --model scratch       # Model 1 only
python scripts/train.py --model fasterrcnn    # Model 2 only
python scripts/train.py --model both          # both, sequentially
```

Useful flags: `--epochs N`, `--seed N`, `--batch-size N`, `--workers N`,
`--data PATH`, `--out PATH`.

### Settings

Every value below lives in [`src/config.py`](src/config.py) and is saved
inside each checkpoint, so a saved model always travels with the exact
settings that produced it.

| | Model 1 (Scratch) | Model 2 (Faster R-CNN) |
|---|---|---|
| Input | 224×224 RGB | 224×224 RGB |
| Optimiser | Adam | SGD, momentum 0.9 |
| Learning rate | 1e-3 | 5e-3 |
| Schedule | cosine annealing | StepLR (×0.5 every 5) |
| Weight decay | 5e-4 | 5e-4 |
| Batch size | 16 | 8 |
| Max epochs | 80 | 20 |
| Gradient clipping | 5.0 | 10.0 |
| Early stopping | patience 12 on val mAP@0.5 | patience 12 on val mAP@0.5 |
| Seed | 42 (`random`, `numpy`, `torch`, CUDA, DataLoader workers) | same |
| Augmentation | h-flip, colour jitter, ±15% scale, ±8% translate | identical |
| Loss | multi-task: BCE objectness + Smooth-L1 box + CE class | RPN + ROI losses |

**Checkpoint selection.** Both models are checkpointed on **validation
mAP@0.5**, not on validation loss, and the best epoch's weights — not the last
epoch's — are what gets written to disk. Faster R-CNN typically begins
overfitting well before its epoch budget runs out; saving the final epoch
would mean reporting the model in its worst state.

**Approximate training time on a Colab T4:** Model 1 ≈ 12 min for 80 epochs,
Model 2 ≈ 25 min for 20 epochs. On CPU, Model 2 is impractical — use Colab.

---

## 6. Saved weights

| File | Contents |
|---|---|
| `checkpoints/scratch_best.pt` | Model 1 weights + config + class names + per-epoch history + hardware |
| `checkpoints/fasterrcnn_best.pt` | Model 2, same structure |

Checkpoints carry everything needed to rebuild the model, so inference needs
nothing but the file path:

```python
from src.engine import load_checkpoint
model, cfg, ckpt = load_checkpoint("checkpoints/scratch_best.pt")
print(ckpt["epoch"], ckpt["metrics"], cfg.class_names)
```

`checkpoints/fasterrcnn_best.pt` is ~165 MB (41M parameters). If the archive
must stay small, keep `scratch_best.pt` (~19 MB), note the Faster R-CNN
checkpoint's location, and regenerate it with
`python scripts/train.py --model fasterrcnn`.

---

## 7. Evaluation

```bash
python scripts/evaluate.py
```

Produces `results/comparison.md`, `results/metrics.json`, and every figure in
`results/figures/`. It runs entirely from saved checkpoints — no retraining.

**Metrics reported:** mAP@0.5, mAP@[.50:.95], per-class AP, precision, recall,
F1, confusion matrix, median and p95 latency, FPS, parameter count, model size,
training time.

Three methodological points the discussion is likely to probe:

1. **mAP is a mean of per-class APs**, computed with all-point interpolation
   over a precision envelope — not a single pooled curve across all classes.
2. **One matching pass feeds everything.** The PR curve, AP, and the
   precision/recall counts all derive from the same greedy assignment in which
   each ground-truth box can be claimed once. A second, looser rule for the
   curve would let two boxes on one object both count as true positives and
   push recall above 1.0. `tests/test_core.py::TestAveragePrecision::test_recall_never_exceeds_one`
   guards against exactly that.
3. **The confidence threshold is chosen on validation, then frozen.** Tuning
   it on the test split would make the reported test numbers the best of many
   peeks at data that is supposed to be untouched.

### Results

Run `scripts/evaluate.py` to generate them. The numbers land in
`results/comparison.md` and are reproduced in section 9 of the notebook.

> This README deliberately contains **no hardcoded metric values.** Everything
> quoted in the report is generated by the submitted code on the machine that
> ran it, as the guide requires.

---

## 8. Testing on unseen real-world images

```bash
python scripts/predict.py                      # bundled samples, both models
python scripts/predict.py --images my_photos/  # your own folder
python scripts/predict.py --model scratch --conf 0.4
```

Images live in `real_world_samples/{normal,difficult,failure_oriented}/`; the
subfolder name becomes the image's category in the output, so the required
normal / difficult / failure-oriented grouping is structural rather than a
note added afterwards. See
[`real_world_samples/README.md`](real_world_samples/README.md) for what to put
in each.

---

## 9. Project structure

```
finalproject/
├── main_project.ipynb          MAIN FILE — full documented run
├── README.md
├── PRESENTATION.md             discussion prep: every question in guide §5, answered
├── requirements.txt
│
├── src/
│   ├── config.py               all settings; serialised into every checkpoint
│   ├── utils.py                seeding, device, GPU-aware timing, JSON I/O
│   ├── boxes.py                IoU, class-aware NMS, coordinate conversion
│   ├── data.py                 datasets, box-aware augmentation, grid encode/decode
│   ├── eda.py                  class balance, duplicates, label validation
│   ├── models.py               ScratchGridDetector + Faster R-CNN builder
│   ├── losses.py               multi-task grid detection loss
│   ├── engine.py               training, early stopping, inference, latency
│   ├── metrics.py              per-class AP, mAP, P/R/F1, confusion matrix
│   ├── failure_analysis.py     six-way error taxonomy and aggregation
│   └── viz.py                  every figure
│
├── scripts/
│   ├── download_data.py        fetch the dataset (key from env, never stored)
│   ├── train.py                train one or both models
│   ├── evaluate.py             full comparison from saved weights
│   └── predict.py              demo on new images
│
├── tests/test_core.py          60+ unit tests of the from-scratch components
│
├── data/                       dataset goes here (not in the archive)
├── checkpoints/                trained weights
├── results/                    metrics, comparison table, figures
│   ├── eda_report.json         measured dataset properties
│   ├── comparison.md           the direct comparison table
│   ├── metrics.json            machine-readable results
│   ├── failure_analysis.md     generated error taxonomy
│   └── figures/                every plot in the report
└── real_world_samples/         unseen demo images, by difficulty
    └── README.md               what to download and where to get it
```

---

## 10. Reproducibility

- All randomness seeded: `random`, `numpy`, `torch`, CUDA, and DataLoader
  workers (`src/utils.py::set_seed`).
- `torch.backends.cudnn.deterministic = True`, `benchmark = False` — a little
  slower, exactly repeatable.
- Config, class names, epoch, metrics, history and hardware are stored inside
  every checkpoint.
- Hardware is recorded in `results/metrics.json`; timing numbers are
  meaningless without it.
- All paths are resolved relative to the project root, so nothing needs
  editing to run on a different machine.

Exact bit-for-bit reproduction across *different* GPUs is not guaranteed —
cuDNN kernel selection and floating-point reduction order vary by architecture.
On the same hardware and seed, results reproduce.

---

## 11. Known limitations

- **One object per grid cell (Model 1).** A plain grid detector cannot encode
  two objects whose centres fall in the same cell. Section 4 of the notebook
  measures exactly how many objects this costs at S=7, 14 and 28, which is the
  evidence behind choosing S=14. Anchors or a finer grid would relax it.
- **Aspect ratio is not preserved.** Resizing to a square distorts wide images.
  Applied identically to both models, so it does not bias the comparison, but
  it does cost absolute accuracy. Letterbox padding would fix it at the cost
  of undoing the padding at evaluation time.
- **Small dataset.** 448 training images is very little for a detector trained
  from scratch; Model 1's ceiling is set by data volume as much as by
  architecture.
- **Single-run results.** Metrics come from one seed. Proper error bars would
  need several seeds per model, which was outside the compute budget.
- **Latency is measured at batch size 1 on one machine.** It reflects this
  hardware, not a deployment target.

---

## 12. References

**Dataset**
- Dwyer, B. *Aquarium Combined Dataset*. Roboflow Universe, 2021.
  https://universe.roboflow.com/brad-dwyer/aquarium-combined (CC BY 4.0)

**Papers**
- Redmon, J., Divvala, S., Girshick, R., Farhadi, A. *You Only Look Once:
  Unified, Real-Time Object Detection*. CVPR 2016. — grid encoding, the
  multi-task loss weighting, and the √w/√h regression trick used in Model 1.
- Ren, S., He, K., Girshick, R., Sun, J. *Faster R-CNN: Towards Real-Time
  Object Detection with Region Proposal Networks*. NeurIPS 2015. — Model 2.
- Lin, T.-Y. et al. *Feature Pyramid Networks for Object Detection*. CVPR 2017.
  — the FPN neck in Model 2's backbone.
- Lin, T.-Y. et al. *Focal Loss for Dense Object Detection*. ICCV 2017. — the
  negative bias initialisation on the objectness channel.
- He, K. et al. *Deep Residual Learning for Image Recognition*. CVPR 2016. —
  ResNet-50 backbone.
- Ioffe, S., Szegedy, C. *Batch Normalization*. ICML 2015.
- Everingham, M. et al. *The PASCAL Visual Object Classes Challenge*. IJCV
  2010. — the all-point interpolated AP definition used in `src/metrics.py`.

**Libraries**
- PyTorch and torchvision — https://pytorch.org
- NumPy, Matplotlib, Pillow
- Roboflow Python SDK — https://docs.roboflow.com

**Pre-trained weights**
- `FasterRCNN_ResNet50_FPN_Weights.DEFAULT` (torchvision), trained on COCO
  train2017.

**AI tools**
- Claude (Anthropic) was used for code structuring, refactoring, debugging
  assistance, and drafting documentation. Every design decision, experiment
  and reported result was reviewed, executed and verified by the team, and we
  are responsible for all submitted content.
