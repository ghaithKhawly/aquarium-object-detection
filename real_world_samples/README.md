# Real-world test images

Nine images that were **never** used in training, validation, or testing —
including the test split, which the development process touched indirectly
through design and hyperparameter choices.

This satisfies the project guide's requirement to test the final system on new
samples covering normal, difficult, and failure-oriented cases.

## Structure

The subfolder name **is** the difficulty category. `scripts/predict.py` reads it
and prints it alongside each result, so the required grouping is structural
rather than a note added afterwards.

```
normal/            3 images — clear, well-lit, identifiable subjects
difficult/         3 images — dense schools, murky water, low contrast
failure_oriented/  3 images — chosen to provoke a specific, explainable failure
```

## What each category tests

**`normal/`** — NOAA reef photographs with clear subjects. Failure here would
indicate something fundamentally wrong. *Result: Model 2 detected the sharks and
the jellyfish; Model 1 returned nothing on both.*

**`difficult/`** — dense schools and turbid water, in domain but hard. This is
where the two models separate most. *Result: Model 1 over-fires heavily (67 and
41 boxes) where Model 2 is more selective (29 and 24).*

**`failure_oriented/`** — three deliberate probes:

| Image | What it tests | Result |
|---|---|---|
| `pl23_reef1437.jpg` (empty reef) | No animal is present, so every box is unambiguously a false positive | Model 1: **10 FPs**, Model 2: **3** |
| `photo_..._01-52-22.jpg` (person) | Out of domain — neither model has a "none of the above" class | Model 1: **1 detection**, Model 2: **0** |
| `photo_..._01-52-36.jpg` (person) | Same, different scene | Both: **0** |

The out-of-domain pair is the most informative result in the demo: Faster
R-CNN's region proposal network learned a genuine objectness prior from COCO
and therefore declines to propose regions on a human face, while the
from-scratch detector does not have that prior and fires. This is a concrete
benefit of pretraining that the aggregate mAP number does not reveal.

## Provenance

Six images are NOAA Photo Library material (public domain); two are the team's
own photographs. Full attribution is in [`sources.md`](sources.md).

None of these images come from the Roboflow Aquarium Combined dataset — reusing
one would place a training or test image here and defeat the purpose.

## Reproducing the results

```bash
python scripts/predict.py
```

Runs both models from saved weights at their calibrated confidence thresholds.
Output lands in `results/predictions/`. Section 11 of `main_project.ipynb`
performs the same run inline with the annotated figure.
