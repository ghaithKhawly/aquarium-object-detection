# Discussion Preparation

Everything the guide's section 5 says you must be ready to explain, with the
answer and where in the repository it lives.

Read this alongside `main_project.ipynb`. **All numbers below are the actual
measured results**, produced by `scripts/evaluate.py` on the untouched test
split and reproduced in `results/comparison.md`.

---

## 0. The results, in one place

Memorise the first three rows. Everything else you can read off the screen.

| Metric | Model 1 (Scratch) | Model 2 (Faster R-CNN) | Ratio |
|---|---|---|---|
| **mAP@0.5** | 0.272 | **0.536** | 2.0× |
| **mAP@[.5:.95]** | 0.082 | **0.263** | 3.2× |
| **Latency (median)** | **36.0 ms** | 329.4 ms | 9.2× faster |
| Precision | 0.273 | 0.668 | |
| Recall | 0.279 | 0.503 | |
| F1 | 0.276 | 0.574 | |
| TP / FP / FN | 163 / 434 / 421 | 294 / 146 / 290 | |
| Throughput | **27.8 FPS** | 3.0 FPS | |
| Parameters | **4.72 M** | 41.33 M | 8.8× smaller |
| Model size | **18.0 MB** | 158.1 MB | |
| Training time | 66 min | 140 min | |
| Best epoch | 72 of 80 | 20 of 20 | |
| Operating threshold | 0.90 | 0.65 | |

**The one-sentence summary:** *Faster R-CNN is twice as accurate at loose
localisation and three times as accurate at strict localisation, for nine times
the latency and nine times the model size.*

### The three findings worth leading with

**1. The mAP@0.5 → mAP@[.5:.95] collapse is the real story.**
Model 1 loses **70%** of its score when the IoU requirement tightens
(0.272 → 0.082); Model 2 loses **51%** (0.536 → 0.263). Model 1 is
*finding* objects far better than it is *localising* them. This is confirmed
independently by the error taxonomy: `poor_localization` is 28% of Model 1's
errors but only 14% of Model 2's. Two different measurements, same conclusion —
that is what makes it a finding rather than an observation.

**2. Model 1's best class is its rarest.**

| Class | objects | Model 1 AP | Model 2 AP |
|---|---|---|---|
| fish | 249 | **0.105** | 0.422 |
| jellyfish | 154 | 0.361 | **0.788** |
| penguin | 82 | **0.038** | 0.384 |
| puffin | 35 | 0.070 | 0.288 |
| shark | 38 | 0.392 | 0.652 |
| starfish | **11** | **0.545** | 0.616 |
| stingray | 15 | 0.391 | 0.605 |

Model 1 scores highest on **starfish (AP 0.545)** — the class with the fewest
instances in the entire dataset — and worst on **fish (0.105)**, the class with
the most. That inverts the usual "more data is better" expectation, and the
explanation is scale, not volume: starfish are large, static and high-contrast,
while fish are small and appear in dense overlapping schools. **This is your
strongest evidence that Model 1's limitation is spatial resolution, not
training data.** Model 2 wins every class.

**3. Both models are limited by missed detections, not false alarms.**

| Failure type | Model 1 | Model 2 | What it points at |
|---|---|---|---|
| `missed_detection` | 421 (49%) | 290 (67%) | object scale, occlusion |
| `poor_localization` | 240 (28%) | 63 (14%) | box regression, output resolution |
| `background_false_positive` | 187 (22%) | 70 (16%) | threshold, background texture |
| `wrong_class` | 3 (0%) | 10 (2%) | classifier — **negligible for both** |
| `duplicate_detection` | 4 (0%) | 3 (1%) | NMS threshold — working correctly |

Miss rate by object size — the clearest single cut in the whole analysis:

| Size | objects | Model 1 | Model 2 |
|---|---|---|---|
| small (<32²px) | 462 | **76.8%** | 56.3% |
| medium | 116 | 51.7% | **24.1%** |
| large | 6 | (too few to read) | (too few) |

Model 1 misses **more than three quarters of all small objects**. Since 76% of
the dataset *is* small objects, that single number largely explains the mAP gap.

### Honest caveats to state before you are asked

- **Model 2 never converged.** Its best epoch was **20 of 20** — the last one.
  Early stopping never triggered, so 0.536 is a **lower bound**, not a
  converged result. More epochs would likely have improved it.
- **Model 1 did converge**, peaking at epoch 72 of 80 with no improvement over
  the final 8 epochs.
- **Model 1's operating threshold is 0.90**, chosen on validation. A threshold
  that extreme indicates poorly calibrated confidence scores — the model is
  confidently wrong often enough that only the very top of its score
  distribution is trustworthy.
- **6 large objects in the test set** is too few to support any claim about
  that bucket. The generated report says so explicitly rather than quoting a
  ratio from it.
- Single seed, 63 test images.
- **CPU latency is load-sensitive.** These timings come from one machine with
  no GPU. The same benchmark run while the CPU was busy produced ~52 ms and
  ~450 ms instead of 36 ms and 329 ms — the *ratio* between the two models
  held at roughly 9×, but the absolute values moved by 40%. If the notebook's
  inline latency figure differs slightly from `results/comparison.md`, that is
  why: they were measured in separate runs under different load. **The ratio is
  the finding; the absolute milliseconds are hardware- and load-specific.**
  On a GPU both models would be far faster and the gap would narrow, because
  Faster R-CNN parallelises better than its sequential two-stage structure
  suggests.

---

## 1. The required discussion flow

The guide prescribes six steps. Here is what to say at each, and what to have
on screen.

### Step 1 — Problem, importance, objective *(~2 min)*

> "We detect and classify seven aquatic species in underwater photographs. The
> input is an RGB image; the output is a set of boxes, each with a class and a
> confidence. It matters because aquariums and marine surveys produce far more
> footage than anyone can annotate by hand — population counts and behaviour
> studies bottleneck on a human watching video.
>
> We used the task to compare two opposite detector designs: one we built from
> scratch, and one pretrained on COCO and fine-tuned. The question we set out to
> answer is what pretraining buys in accuracy and what it costs in latency."

**On screen:** notebook section 1, the system workflow diagram.

### Step 2 — Dataset, exploration, preprocessing, split *(~3 min)*

Lead with the measurements, not the description — they drive every later
decision:

| Finding | Value | Consequence |
|---|---|---|
| Objects per image | mean 7.4, median 4, **max 56** | Crowded scenes; NMS matters |
| Object scale | **76.1% below 32×32 px**, median 23×16 px | The decisive property; punishes coarse output grids |
| Class imbalance | **23 : 1** (fish 55.4% → starfish 2.4%) | Forces per-class AP reporting |
| Corrupted files | 0 of 638 | Every file fully decoded, not just opened |
| Duplicate images | 0, by content hash | Filename checks cannot catch re-encoded copies |
| Label defects | 5 degenerate boxes (<4 px²) | Found and reported, not silently dropped |

Split: 448 / 127 / 63 as published. Justify it: the training split has to stay
large because Model 1 learns from random initialisation; 127 validation images
is the minimum that gives a stable mAP signal for early stopping; the 63-image
test split is small and we say so as a limitation.

Leakage: three controls — filename disjointness, **content-hash disjointness**,
and the test split never entering any decision (threshold chosen on validation,
checkpoints selected on validation).

**On screen:** notebook sections 3 and 6.

### Step 3 — Both architectures and the training strategy *(~5 min, split between you)*

**Model 1 — the from-scratch grid detector.** Walk the data flow: 224×224 in,
five convolutional stages, four of them halving resolution, so 224 → 14 and the
grid size *falls out of the architecture*. Each of the 196 cells predicts
`[p_c, t_x, t_y, w, h, c₁…c₇]`. A cell owns an object when the object's centre
lands inside it.

Then the design decision that shows the work — **section 4**:

| Grid | Objects lost | Max achievable recall |
|---|---|---|
| S=7 | 790 / 3324 (23.8%) | 76.2% |
| **S=14** | 230 (6.9%) | **93.1%** |
| S=28 | 39 (1.2%) | 98.8% |

> "We measured the ceiling before choosing. At S=7 nearly a quarter of the
> objects cannot be encoded at all — recall is capped at 76% before training
> starts. That failure is easily mistaken for a bad architecture when it is
> actually broken supervision. S=14 raises the ceiling to 93% and costs nothing,
> because four poolings already take 224 to 14."

**Model 2 — Faster R-CNN, fine-tuned.** Two stages: backbone + FPN extracts
multi-scale features, the RPN proposes regions, the ROI heads classify and
refine. Contrast it with Model 1 explicitly: Model 1 makes 196 dense predictions
in a fixed geometry regardless of content; Faster R-CNN decides *where to look*
first, then looks carefully. That is what buys accuracy on small overlapping
objects and what costs latency.

The three adaptations (be specific — the guide requires pretrained weights to be
*meaningfully* adapted):
1. COCO's 91-class box predictor discarded, replaced with a randomly-initialised
   7-class + background head.
2. Only the top 3 ResNet stages unfrozen — with 448 images, tuning all of them
   destroys the pretrained features.
3. The internal transform pinned to 224×224. **Say this one unprompted:** left
   at torchvision's default it resizes to an 800-pixel minimum side, and the
   comparison would then be measuring 224 vs 800, not the architectures.

**On screen:** notebook sections 4 and 7.

### Step 4 — Results, quality against speed *(~4 min)*

Present in this order: mAP@0.5 → mAP@[.5:.95] → per-class AP → the speed
trade-off. Do not lead with a single number.

Say what each metric reveals:
- **mAP@0.5** — overall quality, averaged over classes so a rare class cannot
  hide.
- **mAP@[.5:.95]** — averaged over stricter IoU thresholds. The *gap* against
  mAP@0.5 is a direct read on localisation precision: a model that wins the
  first but not the second is finding objects, not localising them.
- **Per-class AP** — which species work. Read it next to the support column.
- **Confusion matrix** — with explicit background row and column, so misses and
  false positives are separated from class confusions.

**On screen:** notebook section 9, `results/comparison.md`.

### Step 5 — Run the system on new inputs *(~3 min)*

```bash
python scripts/predict.py
```

Runs both models on the 9 images in `real_world_samples/`, from saved weights,
no training. Already run — output is in `results/predictions/`.

Say: *"These images were never in train, validation or test. Six are NOAA
public-domain reef photographs, two are our own, and the subfolder names are the
difficulty categories."*

**What actually happened — detections per image:**

| Image | Category | Model 1 | Model 2 | What it shows |
|---|---|---|---|---|
| `pl23_reef2269` (two clear sharks) | normal | **0** | **2** | Model 1 missed both; Model 2 found both |
| `pl23_reef1133` | normal | 0 | 1 | |
| `pl23_reef2272` | normal | **15** | 1 | Model 1 floods a clean image |
| `pl23_reef1591` (dense school) | difficult | **67** | 29 | |
| `pl23_sanc1002` | difficult | 41 | 24 | |
| `pl23_reef0078` | difficult | 2 | **9** | |
| `pl23_reef1437` (empty reef) | failure | **10** | 3 | Every box here is a false positive |
| selfie (human) | failure | **1** | **0** | Model 2 correctly silent on out-of-domain |
| headphones (human) | failure | 0 | 0 | Both correctly silent |

**The three things to point out:**

1. **Model 1 is erratic, not merely weaker.** On the clear two-shark photo it
   produced **zero** detections, then produced **15** on another clean image and
   **67** on a school. It is silent when it should fire and floods when it
   should not — which is the same poor score calibration that forced its
   operating threshold up to 0.90.

2. **The empty reef is the cleanest measurement in the whole demo.** There is no
   animal in the frame, so the ground truth is unambiguous: Model 1 produced
   **10 false positives**, Model 2 produced **3**. Coral texture reads as
   "animal" to both, but three times more often to the scratch model.

3. **Out-of-domain behaviour differs meaningfully.** Neither model has a "none
   of the above" class, so a human face forces a choice. Model 1 emitted a
   detection on the selfie; **Model 2 emitted nothing on either human photo**.
   That is not luck — it reflects Faster R-CNN's RPN having learned a genuine
   objectness prior from COCO, so it declines to propose regions that do not
   look like the objects it was trained on. This is a concrete, defensible
   benefit of pretraining that the aggregate mAP number does not show.

### Step 6 — Failures, limitations, future work *(~3 min)*

Use the generated taxonomy, not impressions. Every prediction is classified as
one of six outcomes and aggregated in `results/failure_analysis.md`.

Then the limitations, stated before you are asked:
- Single seed — the numbers are point estimates with no error bars.
- 63 test images — a small test set, so per-class AP for rare classes is noisy.
- 224×224 input — small for Faster R-CNN, chosen deliberately for fairness.
- One object per cell in Model 1 — recall ceiling 93.1%, measured.
- Aspect ratio not preserved — applied to both, so it does not bias the
  comparison, but it costs absolute accuracy.

---

## 2. Questions you must be able to answer

The guide lists these explicitly. Short answers below; the code reference is
where to point.

### "Explain the data flow and what the main layers do."

224×224×3 → stage 1 (2 conv + pool) → 112 → stage 2 → 56 → stage 3 → 28 →
stage 4 → 14 → stage 5 (2 conv, no pool, 512 channels) → 1×1 conv head → 14×14×12
→ permute so the last axis is the prediction vector.

Each conv block is **Conv → BatchNorm → ReLU**. BatchNorm normalises each
channel to zero mean and unit variance over the batch, then rescales with two
learned parameters; without it, a network this deep from random init on 448
images needs a tiny learning rate and converges too slowly to be useful. The
conv bias is omitted because BatchNorm's shift parameter makes it redundant.

The head is a **1×1 convolution, not a dense layer** (YOLOv1 used dense). This
keeps the network fully convolutional: the same 12-output predictor is applied
at every cell with shared weights, cutting parameters by an order of magnitude
and making each cell's prediction depend only on its receptive field.

→ `src/models.py`

### "Why does the model output raw logits instead of probabilities?"

So training can use the fused `*_with_logits` losses. `BCEWithLogitsLoss`
applies the log-sum-exp trick internally; `sigmoid` followed by a separate
`BCELoss` can produce `log(0)` and return NaN gradients once the model becomes
confident. Activations are applied in exactly two places — inside the loss, and
inside `decode_grid_predictions` — so training and inference cannot disagree
about what the numbers mean.

→ `src/losses.py`, `src/data.py::decode_grid_predictions`

### "Explain your loss function."

$L = 5·L_{loc} + 1·L_{obj} + 0.5·L_{noobj} + 1·L_{cls}$

- $L_{loc}$: Smooth-L1 on $(t_x, t_y)$ and on $(\sqrt{w}, \sqrt{h})$ — **object
  cells only**.
- $L_{obj}$ / $L_{noobj}$: BCE-with-logits on objectness, split by the object
  mask so the two can be weighted differently.
- $L_{cls}$: cross-entropy — **object cells only**.

Three follow-ups you should pre-empt:

**Why do localisation and classification skip background cells?** A cell with no
object has no box to regress and no class to name; including it would train the
network towards an arbitrary target.

**Why √w and √h?** A 10-pixel error on a 20-pixel starfish matters far more than
the same error on a 200-pixel shark, but L1 on raw width treats them identically.
The square root compresses the large end and expands the small end, so small
objects — 76% of this dataset — get a proportionate share of the gradient.

**Why λ_noobj = 0.5?** ~96% of the 196 cells are background. Equal weighting
lets the background term dominate the gradient and drives the model to "predict
nothing anywhere", which scores well on loss and detects nothing.

→ `src/losses.py`

### "Why is the detection head initialised differently from the rest?"

The backbone gets He initialisation, which preserves activation variance through
a ReLU stack. The head is not a link in a ReLU chain — it is a linear output
layer, and He init over its 512 input channels produces logits around ±10, which
would swamp the bias prior and start training from a saturated sigmoid with
near-zero gradient. It gets normal(0, 0.01) instead, plus an objectness bias of
−4 (σ(−4) ≈ 0.018) so the model starts from "assume empty".

*This was a real bug caught by `tests/test_core.py::TestModel::test_objectness_bias_starts_negative`.*

→ `src/models.py::_init_weights`

### "What preprocessing did you apply, and why each step?"

Resize 224×224 (fixed tensor shape; aspect ratio not preserved, applied
identically to both models). Scale to [0,1]. **Normalisation happens inside each
model, not in the dataset** — Model 1 via a registered buffer, Model 2 via
torchvision's internal transform. Doing it in the dataset as well would
normalise Model 2 twice and corrupt its pretrained features.

Augmentation, training split only: horizontal flip (underwater scenes have no
meaningful left/right), colour jitter (water colour and turbidity are the
dominant nuisance variables), ±15% scale and ±8% translate (animals appear at
varying distances). Excluded deliberately: vertical flips and large rotations
(an upside-down penguin is not a sample the deployed system will see), and
mosaic (it fabricates object co-occurrences that do not exist here).

**Every geometric op maps the boxes too.** Off-the-shelf `torchvision.transforms`
transform only the image and would silently invalidate the annotations.

→ `src/data.py::BoxAwareAugment`, verified in `tests/test_core.py::TestAugmentation`

### "How did you prevent leakage, and why are the test results valid?"

Three independent controls:
1. **Filename disjointness** across the three splits — asserted, not assumed.
2. **Content-hash disjointness.** Each image is reduced to a 16×16 greyscale
   thumbnail thresholded at its own mean. This catches the same photograph saved
   under two names — the usual way duplicates enter a dataset, and invisible to
   a filename check. Result: 638 images, 638 unique contents, 0 cross-split.
3. **The test split enters no decision.** Checkpoints are selected on validation
   mAP; the confidence threshold is chosen on validation F1 and then frozen. The
   test split is read once, at the end.

→ `src/eda.py::leakage_report`, `src/data.py::find_duplicates`

### "Why did you select checkpoints on validation mAP rather than validation loss?"

The two losses are not comparable — one is a weighted multi-task sum over a
grid, the other a sum of RPN and ROI terms — and loss is a proxy anyway: the
lowest-loss epoch is not necessarily the best-detecting one. Selecting on the
metric we actually report also avoids the common failure where a model starts
overfitting at epoch 5 and the weights saved are from epoch 20.

→ `src/engine.py::train_model`

### "What do the reported curves and runtime measurements mean?"

Training curves: loss and validation mAP per epoch, with the selected epoch
marked. If validation mAP peaks well before the run ends, that gap is
overfitting caught by early stopping — and the saved weights are from the peak.

Latency: batch size 1, warm-up iterations discarded (the first calls pay for
CUDA context creation and kernel autotuning), every measurement bracketed by
`torch.cuda.synchronize()` because CUDA launches are asynchronous and an
unsynchronised clock measures queueing rather than computation. Median and p95
reported, not just mean, since one stall would dominate a mean over a few dozen
samples. Post-processing is included — it is part of the cost of getting a
detection out.

→ `src/engine.py::benchmark_latency`, `src/utils.py::Timer`

### "Which parts are yours and which are external?"

**Ours:** the grid architecture, target encoding and decoding, the multi-task
loss, IoU, class-aware NMS, the whole evaluation stack (per-class AP, mAP,
confusion matrix), the error taxonomy, the box-aware augmentation, the training
engine, and 59 unit tests.

**External, and cited:** torchvision's Faster R-CNN implementation and its
COCO-pretrained weights; PyTorch, NumPy, Matplotlib, Pillow; the Roboflow
Aquarium Combined dataset (CC BY 4.0). Design ideas from YOLOv1 (grid encoding,
λ weighting, √w/√h), Faster R-CNN, FPN, RetinaNet (the prior-bias trick), and
PASCAL VOC (the all-point interpolated AP definition).

**AI assistance:** Claude was used for code structuring, refactoring, debugging
and documentation drafting. Every design decision and result was reviewed and
executed by us. *(Say this plainly if asked — the guide explicitly permits it
and explicitly requires disclosure.)*

---

## 3. Hard questions, and honest answers

**"Your mAP is low. Isn't this a failed project?"**
The guide's own instruction is that the goal is not merely a working result. The
project's finding is the *comparison*: an identical pipeline with and without
COCO pretraining, on 448 images. Model 1's ceiling is set by data volume and by
the measured 93.1% encoding limit, both of which we quantified rather than
guessed. A low absolute number that we can fully explain is worth more than a
high one we cannot.

**"Why not just use YOLOv8 and get a better number?"**
That would measure library defaults, not understanding. The guide rules out
off-the-shelf execution without meaningful technical contribution. We wanted one
model whose every component we implemented and could defend.

**"Isn't 224×224 crippling Faster R-CNN?"**
Yes, and deliberately. Its default is an 800-pixel minimum side. Had we left it
there, the comparison would be 224 vs 800 and the accuracy and latency gaps
would mostly measure input resolution. We chose the fair comparison over the
flattering number, and flagged it as a limitation.

**"How do you know your mAP implementation is correct?"**
It is unit-tested against cases with hand-computable answers: perfect ranking →
AP 1.0, all false positives → 0.0, one of two objects found and ranked first →
exactly 0.5. There is also a specific regression test that two boxes on one
object yield one TP and one FP, so recall can never exceed 1.0 — the classic
double-counting bug.

**"Why do the two models use different optimisers? Isn't that unfair?"**
Each uses the recipe appropriate to its regime: Adam converges faster from
random init, SGD+momentum is the reference for fine-tuning pretrained detection
backbones and disturbs them less. Forcing one optimiser on both would handicap
whichever it suits less. Every non-identical condition is listed with its
justification in `results/comparison.md`; everything else is shared verbatim.

**"Could the comparison be biased by preprocessing?"**
No — both models are fed by the same base dataset class, so resize and
augmentation are provably identical, and both post-process through the same NMS
function at the same IoU. torchvision's internal NMS runs first at that same
threshold, so our pass is idempotent for Model 2 — tested.

---

## 4. Splitting the presentation between two students

Both of you must understand all of it — either can be asked about any part. A
workable split by section:

| Student A | Student B |
|---|---|
| 1. Problem and objective | 5. Model 2: Faster R-CNN and the adaptations |
| 2. Dataset and EDA findings | 6. Evaluation methodology and metrics |
| 3. Preprocessing, augmentation, split, leakage | 7. Results and the speed/accuracy trade-off |
| 4. Model 1: architecture, grid analysis, loss | 8. Failure analysis, live demo, future work |

Swap once in rehearsal so each of you has explained the other's half at least
once.

---

## 5. Before you walk in

Already done:

- [x] Both models trained; `checkpoints/*.pt` hold the best-epoch weights
- [x] `scripts/evaluate.py` run — `results/comparison.md`, `metrics.json`, 16 figures
- [x] `scripts/predict.py` run on all 9 unseen images — `results/predictions/`
- [x] `pytest tests/` passes — **59 tests**, a good answer to "did you test it?"
- [x] Notebook executed with stored outputs (31/31 cells, 20 figures, 0 errors)
- [x] `main_project.html` exported for viewing without Jupyter
- [x] 9 images in the three difficulty folders, sources recorded in `sources.md`
- [x] Both student names in `README.md` and notebook cell 0

Still yours to do:

- [ ] **Section number** — still `نسيت` in `README.md` and notebook cell 0
- [ ] Delete `main_project (1).ipynb` if it is still in the project folder
- [ ] Read `EXPLANATION_AR.md` once end to end
- [ ] Rehearse the split once, swapping halves

### Numbers to know without looking

| | |
|---|---|
| Dataset | 638 images, 4,817 objects, 7 classes |
| Scene density | 7.4 objects/image average, 56 max |
| **Object scale** | **76% smaller than 32×32 px** |
| Imbalance | 23 : 1 |
| **Grid ceiling** | S=7 → 76.2% recall · **S=14 → 93.1%** |
| **mAP@0.5** | **0.272 vs 0.536** |
| **mAP@[.5:.95]** | **0.082 vs 0.263** |
| **Latency** | **36 ms vs 329 ms** (9× ) |
| Parameters | 4.7 M vs 41.3 M (8.8×) |
| Small-object miss rate | 76.8% vs 56.3% |

If you remember only three: **0.272 vs 0.536**, **9× faster**, and
**76% of objects are small**. Every other number can be read off the screen.
