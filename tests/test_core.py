"""Unit tests for the components this project implements from scratch.

The pieces tested here - IoU, NMS, grid encoding/decoding, the multi-task
loss, and the AP computation - are the ones the report makes claims about, so
they are verified against cases with hand-computable answers rather than
trusted because the training loss went down.

    pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.boxes import iou_matrix, iou_xyxy, non_max_suppression, xyxy_to_yolo, yolo_to_xyxy
from src.config import Config
from src.data import decode_grid_predictions, encode_grid_targets
from src.losses import GridDetectionLoss
from src.metrics import (
    ImageGroundTruth, ImagePrediction, compute_average_precision, evaluate_detections,
)
from src.models import ScratchGridDetector


# ==========================================================================
# IoU
# ==========================================================================
class TestIoU:
    def test_identical_boxes(self):
        box = [10, 10, 20, 20]
        assert iou_xyxy(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        assert iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_touching_edges_is_zero(self):
        # Sharing an edge means zero overlap area, not a sliver.
        assert iou_xyxy([0, 0, 10, 10], [10, 0, 20, 10]) == 0.0

    def test_known_quarter_overlap(self):
        # Two 10x10 boxes offset by 5 in both axes: intersection 5x5 = 25,
        # union 100 + 100 - 25 = 175, so IoU = 25/175 = 1/7.
        assert iou_xyxy([0, 0, 10, 10], [5, 5, 15, 15]) == pytest.approx(1 / 7)

    def test_contained_box(self):
        # 5x5 inside 10x10: intersection 25, union 100 -> 0.25
        assert iou_xyxy([0, 0, 10, 10], [2, 2, 7, 7]) == pytest.approx(0.25)

    def test_degenerate_box_returns_zero(self):
        assert iou_xyxy([5, 5, 5, 5], [0, 0, 10, 10]) == 0.0

    def test_matrix_matches_scalar(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 50, size=(7, 2))
        b = rng.uniform(0, 50, size=(5, 2))
        boxes_a = np.hstack([a, a + rng.uniform(1, 30, size=(7, 2))])
        boxes_b = np.hstack([b, b + rng.uniform(1, 30, size=(5, 2))])

        matrix = iou_matrix(boxes_a, boxes_b)
        for i in range(len(boxes_a)):
            for j in range(len(boxes_b)):
                assert matrix[i, j] == pytest.approx(iou_xyxy(boxes_a[i], boxes_b[j]))

    def test_matrix_handles_empty(self):
        assert iou_matrix(np.zeros((0, 4)), np.ones((3, 4))).shape == (0, 3)


# ==========================================================================
# Coordinate conversion
# ==========================================================================
class TestCoordinateConversion:
    def test_round_trip(self):
        boxes = np.array([[0.5, 0.5, 0.25, 0.4], [0.1, 0.9, 0.2, 0.1]])
        xyxy = yolo_to_xyxy(boxes, 224, 224)
        back = xyxy_to_yolo(xyxy, 224, 224)
        np.testing.assert_allclose(back, boxes, atol=1e-9)

    def test_centre_box_maps_to_centre(self):
        xyxy = yolo_to_xyxy(np.array([[0.5, 0.5, 1.0, 1.0]]), 100, 100)
        np.testing.assert_allclose(xyxy[0], [0, 0, 100, 100])

    def test_non_square_image(self):
        xyxy = yolo_to_xyxy(np.array([[0.5, 0.5, 0.5, 0.5]]), 200, 100)
        np.testing.assert_allclose(xyxy[0], [50, 25, 150, 75])


# ==========================================================================
# NMS
# ==========================================================================
class TestNMS:
    def test_suppresses_overlapping_same_class(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=float)
        keep = non_max_suppression(boxes, np.array([0.9, 0.8]), np.array([0, 0]),
                                   iou_threshold=0.5)
        assert list(keep) == [0]

    def test_keeps_overlapping_different_classes(self):
        # Class-aware: a shark box must not suppress an overlapping fish box.
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=float)
        keep = non_max_suppression(boxes, np.array([0.9, 0.8]), np.array([0, 1]),
                                   iou_threshold=0.5)
        assert sorted(keep) == [0, 1]

    def test_keeps_distant_boxes(self):
        boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=float)
        keep = non_max_suppression(boxes, np.array([0.9, 0.8]), np.array([0, 0]))
        assert sorted(keep) == [0, 1]

    def test_output_sorted_by_descending_score(self):
        boxes = np.array([[0, 0, 5, 5], [50, 50, 55, 55], [100, 100, 105, 105]], dtype=float)
        keep = non_max_suppression(boxes, np.array([0.3, 0.9, 0.6]), np.array([0, 0, 0]))
        assert list(keep) == [1, 2, 0]

    def test_score_threshold_filters(self):
        boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=float)
        keep = non_max_suppression(boxes, np.array([0.9, 0.2]), np.array([0, 0]),
                                   score_threshold=0.5)
        assert list(keep) == [0]

    def test_empty_input(self):
        keep = non_max_suppression(np.zeros((0, 4)), np.zeros(0), np.zeros(0))
        assert len(keep) == 0

    def test_max_detections(self):
        boxes = np.array([[i * 20, 0, i * 20 + 10, 10] for i in range(5)], dtype=float)
        keep = non_max_suppression(boxes, np.linspace(0.9, 0.5, 5), np.zeros(5, dtype=int),
                                   max_detections=3)
        assert len(keep) == 3

    def test_idempotent(self):
        """Running NMS on already-suppressed output must change nothing.

        The evaluation pipeline relies on this: Model 2's detections pass
        through torchvision's internal NMS and then ours at the same
        threshold, so the second pass has to be a no-op for the comparison to
        be fair.
        """
        rng = np.random.default_rng(3)
        xy = rng.uniform(0, 180, size=(30, 2))
        boxes = np.hstack([xy, xy + rng.uniform(10, 40, size=(30, 2))])
        scores = rng.uniform(0, 1, size=30)
        labels = rng.integers(0, 3, size=30)

        first = non_max_suppression(boxes, scores, labels, iou_threshold=0.45)
        second = non_max_suppression(boxes[first], scores[first], labels[first],
                                     iou_threshold=0.45)
        assert list(second) == list(range(len(first)))


# ==========================================================================
# Grid encoding / decoding
# ==========================================================================
class TestGridEncoding:
    cfg = Config()

    def test_single_box_lands_in_correct_cell(self):
        S, img = 14, 224
        # Centre at (0.25, 0.75) of the image -> col 3, row 10 at S=14
        boxes = yolo_to_xyxy(np.array([[0.25, 0.75, 0.2, 0.2]]), img, img)
        target, dropped = encode_grid_targets(boxes, np.array([2]), S, 7, img)

        assert dropped == 0
        assert target[..., 0].sum() == 1
        row, col = 10, 3
        assert target[row, col, 0] == 1.0
        assert target[row, col, 5 + 2] == 1.0

    def test_offsets_are_within_cell(self):
        S, img = 14, 224
        boxes = yolo_to_xyxy(np.array([[0.3333, 0.6666, 0.1, 0.1]]), img, img)
        target, _ = encode_grid_targets(boxes, np.array([0]), S, 7, img)
        occupied = torch.nonzero(target[..., 0])
        row, col = occupied[0].tolist()
        tx, ty = target[row, col, 1].item(), target[row, col, 2].item()
        assert 0.0 <= tx <= 1.0 and 0.0 <= ty <= 1.0

    def test_encode_decode_round_trip(self):
        """A perfectly-confident encoded target must decode to the input box.

        This is the single most important correctness check for Model 1: if
        encoding and decoding disagree, the model can train to a low loss and
        still produce nonsense boxes at inference.
        """
        S, img, C = 14, 224, 7
        original = np.array([[0.4, 0.6, 0.25, 0.35]])
        boxes = yolo_to_xyxy(original, img, img)
        target, _ = encode_grid_targets(boxes, np.array([3]), S, C, img)

        # Turn the [0,1] target into the logits that would produce it exactly.
        def inv_sigmoid(v, eps=1e-6):
            v = np.clip(v, eps, 1 - eps)
            return float(np.log(v / (1 - v)))

        raw = torch.zeros(S, S, 5 + C)
        raw[..., 0] = -10.0                       # everything else is background
        occupied = torch.nonzero(target[..., 0])
        row, col = occupied[0].tolist()

        raw[row, col, 0] = 10.0
        raw[row, col, 1] = inv_sigmoid(target[row, col, 1].item())
        raw[row, col, 2] = inv_sigmoid(target[row, col, 2].item())
        raw[row, col, 3] = inv_sigmoid(target[row, col, 3].item())
        raw[row, col, 4] = inv_sigmoid(target[row, col, 4].item())
        raw[row, col, 5 + 3] = 20.0

        out_boxes, scores, labels = decode_grid_predictions(raw, img, conf_threshold=0.5)

        assert len(out_boxes) == 1
        assert int(labels[0]) == 3
        assert scores[0] > 0.99
        np.testing.assert_allclose(out_boxes[0], boxes[0], atol=0.5)

    def test_collision_keeps_larger_object(self):
        S, img = 7, 224
        # Two centres inside the same cell; the larger must win.
        small = [0.51, 0.51, 0.05, 0.05]
        large = [0.52, 0.52, 0.30, 0.30]
        boxes = yolo_to_xyxy(np.array([small, large]), img, img)
        target, dropped = encode_grid_targets(boxes, np.array([0, 1]), S, 7, img)

        assert dropped == 1
        assert target[..., 0].sum() == 1
        occupied = torch.nonzero(target[..., 0])[0].tolist()
        kept_class = int(torch.argmax(target[occupied[0], occupied[1], 5:]))
        assert kept_class == 1, "the larger object should survive the collision"

    def test_finer_grid_drops_fewer_objects(self):
        img = 224
        rng = np.random.default_rng(7)
        centres = rng.uniform(0.05, 0.95, size=(40, 2))
        wh = rng.uniform(0.03, 0.12, size=(40, 2))
        boxes = yolo_to_xyxy(np.hstack([centres, wh]), img, img)
        labels = rng.integers(0, 7, size=40)

        _, dropped_coarse = encode_grid_targets(boxes, labels, 7, 7, img)
        _, dropped_fine = encode_grid_targets(boxes, labels, 14, 7, img)
        assert dropped_fine <= dropped_coarse

    def test_empty_input_gives_empty_grid(self):
        target, dropped = encode_grid_targets(np.zeros((0, 4)), np.zeros(0, dtype=int),
                                              14, 7, 224)
        assert target.shape == (14, 14, 12)
        assert target.sum() == 0 and dropped == 0

    def test_decode_respects_confidence_threshold(self):
        raw = torch.full((14, 14, 12), -10.0)
        boxes, scores, labels = decode_grid_predictions(raw, 224, conf_threshold=0.5)
        assert len(boxes) == 0


# ==========================================================================
# Model
# ==========================================================================
class TestModel:
    def test_output_shape(self):
        model = ScratchGridDetector(num_classes=7, grid_size=14)
        out = model(torch.rand(2, 3, 224, 224))
        assert out.shape == (2, 14, 14, 12)

    def test_grid_size_falls_out_of_architecture(self):
        """Four poolings take 224 -> 14 without the adaptive layer resizing."""
        model = ScratchGridDetector(num_classes=7, grid_size=14)
        x = torch.rand(1, 3, 224, 224)
        x = (x - model.pixel_mean) / model.pixel_std
        for stage in (model.stage1, model.stage2, model.stage3, model.stage4, model.stage5):
            x = stage(x)
        assert x.shape[-2:] == (14, 14)

    def test_objectness_bias_starts_negative(self):
        model = ScratchGridDetector()
        out = model(torch.rand(1, 3, 224, 224))
        assert torch.sigmoid(out[..., 0]).mean() < 0.25, (
            "the objectness prior should start low, since most cells are background"
        )

    def test_gradients_flow_to_first_layer(self):
        model = ScratchGridDetector()
        out = model(torch.rand(2, 3, 224, 224))
        out.sum().backward()
        first_conv = model.stage1[0][0]
        assert first_conv.weight.grad is not None
        assert torch.isfinite(first_conv.weight.grad).all()
        assert first_conv.weight.grad.abs().sum() > 0

    def test_normalisation_buffers_persist_in_state_dict(self):
        """The checkpoint must carry the normalisation constants."""
        model = ScratchGridDetector()
        keys = model.state_dict().keys()
        assert "pixel_mean" in keys and "pixel_std" in keys


# ==========================================================================
# Loss
# ==========================================================================
class TestLoss:
    def test_perfect_prediction_has_low_loss(self):
        S, C = 14, 7
        target = torch.zeros(1, S, S, 5 + C)
        target[0, 5, 5, 0] = 1.0
        target[0, 5, 5, 1:3] = 0.5
        target[0, 5, 5, 3:5] = 0.25
        target[0, 5, 5, 5 + 2] = 1.0

        pred = torch.full((1, S, S, 5 + C), -10.0)
        pred[0, 5, 5, 0] = 10.0
        pred[0, 5, 5, 1:3] = 0.0        # sigmoid(0) = 0.5, matching t_x, t_y
        pred[0, 5, 5, 3:5] = float(np.log(0.25 / 0.75))   # sigmoid -> 0.25
        pred[0, 5, 5, 5:] = -10.0
        pred[0, 5, 5, 5 + 2] = 10.0

        loss, parts = GridDetectionLoss()(pred, target)
        assert loss.item() < 0.05
        assert parts["loc"] < 1e-3 and parts["cls"] < 1e-3

    def test_wrong_prediction_costs_more_than_right_one(self):
        S, C = 14, 7
        target = torch.zeros(1, S, S, 5 + C)
        target[0, 3, 3, 0] = 1.0
        target[0, 3, 3, 1:5] = 0.5
        target[0, 3, 3, 5] = 1.0

        good = torch.full((1, S, S, 5 + C), -10.0)
        good[0, 3, 3, 0] = 10.0
        good[0, 3, 3, 1:5] = 0.0
        good[0, 3, 3, 5] = 10.0

        bad = torch.full((1, S, S, 5 + C), -10.0)
        bad[0, 3, 3, 0] = 10.0
        bad[0, 3, 3, 1:5] = 5.0
        bad[0, 3, 3, 6] = 10.0

        assert GridDetectionLoss()(bad, target)[0] > GridDetectionLoss()(good, target)[0]

    def test_all_background_batch_is_finite(self):
        target = torch.zeros(2, 14, 14, 12)
        pred = torch.randn(2, 14, 14, 12)
        loss, parts = GridDetectionLoss()(pred, target)
        assert torch.isfinite(loss)
        assert parts["loc"] == 0.0 and parts["cls"] == 0.0

    def test_loss_is_differentiable(self):
        target = torch.zeros(1, 14, 14, 12)
        target[0, 2, 2, 0] = 1.0
        target[0, 2, 2, 1:5] = 0.4
        target[0, 2, 2, 6] = 1.0

        pred = torch.randn(1, 14, 14, 12, requires_grad=True)
        loss, _ = GridDetectionLoss()(pred, target)
        loss.backward()
        assert torch.isfinite(pred.grad).all()

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            GridDetectionLoss()(torch.zeros(1, 14, 14, 12), torch.zeros(1, 7, 7, 12))


# ==========================================================================
# Metrics
# ==========================================================================
class TestAveragePrecision:
    def test_perfect_ranking_gives_ap_one(self):
        scores = np.array([0.9, 0.8, 0.7])
        is_tp = np.array([True, True, True])
        ap, _, _ = compute_average_precision(scores, is_tp, n_gt=3)
        assert ap == pytest.approx(1.0)

    def test_all_false_positives_give_ap_zero(self):
        ap, _, _ = compute_average_precision(
            np.array([0.9, 0.8]), np.array([False, False]), n_gt=2
        )
        assert ap == pytest.approx(0.0)

    def test_half_recall_perfect_precision(self):
        # One correct detection out of two objects, ranked first.
        ap, _, _ = compute_average_precision(
            np.array([0.9]), np.array([True]), n_gt=2
        )
        assert ap == pytest.approx(0.5)

    def test_no_ground_truth_is_nan(self):
        ap, _, _ = compute_average_precision(np.array([0.9]), np.array([False]), n_gt=0)
        assert np.isnan(ap)

    def test_recall_never_exceeds_one(self):
        """Regression guard for the classic double-counting bug.

        Two predictions on a single object must not both count as true
        positives: the second is a duplicate, and cumulative recall has to
        stay at or below 1.0.
        """
        gt = [ImageGroundTruth(boxes=np.array([[0, 0, 50, 50]], dtype=float),
                               labels=np.array([0]), image_id="a")]
        preds = [ImagePrediction(
            boxes=np.array([[0, 0, 50, 50], [1, 1, 51, 51]], dtype=float),
            scores=np.array([0.95, 0.90]),
            labels=np.array([0, 0]),
            image_id="a",
        )]
        result = evaluate_detections(preds, gt, ["fish"], compute_map_range=False)

        assert result.tp == 1, "only one of the two boxes may be a true positive"
        assert result.fp == 1, "the duplicate must be counted as a false positive"
        assert result.recall <= 1.0

        for rec, _ in result.pr_curves.values():
            assert rec.max() <= 1.0 + 1e-9


class TestEvaluateDetections:
    def _single(self, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
        preds = [ImagePrediction(np.asarray(pred_boxes, dtype=float),
                                 np.asarray(pred_scores, dtype=float),
                                 np.asarray(pred_labels), "img")]
        gts = [ImageGroundTruth(np.asarray(gt_boxes, dtype=float),
                                np.asarray(gt_labels), "img")]
        return preds, gts

    def test_perfect_detection(self):
        preds, gts = self._single([[0, 0, 50, 50]], [0], [0.99], [[0, 0, 50, 50]], [0])
        r = evaluate_detections(preds, gts, ["fish", "shark"], compute_map_range=False)
        assert r.tp == 1 and r.fp == 0 and r.fn == 0
        assert r.precision == pytest.approx(1.0)
        assert r.recall == pytest.approx(1.0)
        assert r.f1 == pytest.approx(1.0)
        assert r.per_class["fish"]["ap"] == pytest.approx(1.0)

    def test_wrong_class_is_not_a_true_positive(self):
        preds, gts = self._single([[0, 0, 50, 50]], [1], [0.99], [[0, 0, 50, 50]], [0])
        r = evaluate_detections(preds, gts, ["fish", "shark"], compute_map_range=False)
        assert r.tp == 0 and r.fp == 1 and r.fn == 1

    def test_poor_overlap_is_not_a_true_positive(self):
        preds, gts = self._single([[0, 0, 20, 20]], [0], [0.99], [[0, 0, 50, 50]], [0])
        r = evaluate_detections(preds, gts, ["fish"], iou_threshold=0.5,
                                compute_map_range=False)
        assert r.tp == 0 and r.fn == 1

    def test_missing_prediction_counts_as_fn(self):
        preds = [ImagePrediction(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int), "i")]
        gts = [ImageGroundTruth(np.array([[0, 0, 50, 50]], dtype=float),
                                np.array([0]), "i")]
        r = evaluate_detections(preds, gts, ["fish"], compute_map_range=False)
        assert r.tp == 0 and r.fn == 1 and r.recall == 0.0

    def test_map_ignores_classes_with_no_ground_truth(self):
        preds, gts = self._single([[0, 0, 50, 50]], [0], [0.99], [[0, 0, 50, 50]], [0])
        r = evaluate_detections(preds, gts, ["fish", "unseen"], compute_map_range=False)
        assert r.map_50 == pytest.approx(1.0)
        assert np.isnan(r.per_class["unseen"]["ap"])

    def test_confusion_matrix_shape_and_totals(self):
        preds, gts = self._single([[0, 0, 50, 50]], [1], [0.9], [[0, 0, 50, 50]], [0])
        r = evaluate_detections(preds, gts, ["fish", "shark"], compute_map_range=False)
        assert r.confusion_matrix.shape == (3, 3)
        # localised correctly but labelled shark -> row fish, column shark
        assert r.confusion_matrix[0, 1] == 1

    def test_mismatched_lengths_raise(self):
        preds = [ImagePrediction(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int))]
        with pytest.raises(ValueError):
            evaluate_detections(preds, [], ["fish"])


# ==========================================================================
# Failure analysis
# ==========================================================================
class TestFailureAnalysis:
    def test_taxonomy_assigns_each_case(self):
        from src.failure_analysis import classify_image_errors

        gt = ImageGroundTruth(
            boxes=np.array([[0, 0, 50, 50], [100, 100, 150, 150]], dtype=float),
            labels=np.array([0, 1]), image_id="img",
        )
        pred = ImagePrediction(
            boxes=np.array([
                [0, 0, 50, 50],          # exact match, class 0    -> true_positive
                [1, 1, 51, 51],          # same object again       -> duplicate_detection
                [300, 300, 350, 350],    # nothing there           -> background FP
            ], dtype=float),
            scores=np.array([0.95, 0.90, 0.85]),
            labels=np.array([0, 0, 0]),
            image_id="img",
        )
        kinds = [r.kind for r in classify_image_errors(pred, gt)]
        assert kinds[0] == "true_positive"
        assert kinds[1] == "duplicate_detection"
        assert kinds[2] == "background_false_positive"
        assert "missed_detection" in kinds       # the class-1 object was never found

    def test_wrong_class_distinguished_from_background(self):
        from src.failure_analysis import classify_image_errors

        gt = ImageGroundTruth(np.array([[0, 0, 50, 50]], dtype=float), np.array([0]), "i")
        pred = ImagePrediction(np.array([[0, 0, 50, 50]], dtype=float),
                               np.array([0.9]), np.array([3]), "i")
        kinds = [r.kind for r in classify_image_errors(pred, gt)]
        assert "wrong_class" in kinds
        assert "background_false_positive" not in kinds

    def test_size_buckets(self):
        from src.failure_analysis import size_bucket

        assert size_bucket(np.array([0, 0, 20, 20])) == "small"
        assert size_bucket(np.array([0, 0, 60, 60])) == "medium"
        assert size_bucket(np.array([0, 0, 150, 150])) == "large"


# ==========================================================================
# Reproducibility
# ==========================================================================
class TestReproducibility:
    def test_same_seed_gives_same_weights(self):
        from src.utils import set_seed

        set_seed(123)
        a = ScratchGridDetector()
        set_seed(123)
        b = ScratchGridDetector()

        for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
            assert torch.equal(pa, pb), f"{name} differs between identically seeded runs"

    def test_different_seeds_give_different_weights(self):
        from src.utils import set_seed

        set_seed(1)
        a = ScratchGridDetector()
        set_seed(2)
        b = ScratchGridDetector()
        assert not torch.equal(a.head.weight, b.head.weight)


# ==========================================================================
# Checkpoint round trip
# ==========================================================================
class TestCheckpoint:
    def test_save_and_load_preserves_predictions(self, tmp_path):
        from src.engine import load_checkpoint, save_checkpoint

        cfg = Config()
        model = ScratchGridDetector(num_classes=cfg.num_classes, grid_size=cfg.grid_size)
        model.eval()

        sample = torch.rand(1, 3, cfg.img_size, cfg.img_size)
        with torch.no_grad():
            before = model(sample)

        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, model, "scratch", cfg, epoch=7, metrics={"val_map50": 0.3})

        restored, restored_cfg, ckpt = load_checkpoint(path, torch.device("cpu"))
        with torch.no_grad():
            after = restored(sample)

        assert torch.allclose(before, after, atol=1e-6)
        assert ckpt["epoch"] == 7
        assert restored_cfg.grid_size == cfg.grid_size
        assert restored_cfg.class_names == cfg.class_names

    def test_missing_checkpoint_raises_helpful_error(self, tmp_path):
        from src.engine import load_checkpoint

        with pytest.raises(FileNotFoundError, match="train.py"):
            load_checkpoint(tmp_path / "nope.pt")


# ==========================================================================
# Augmentation
# ==========================================================================
class TestAugmentation:
    def test_horizontal_flip_moves_boxes_correctly(self):
        from PIL import Image
        from src.data import BoxAwareAugment

        cfg = Config()
        cfg.aug_hflip_prob = 1.0
        cfg.aug_color_jitter_prob = 0.0
        cfg.aug_scale_translate_prob = 0.0

        img = Image.new("RGB", (224, 224))
        boxes = np.array([[10.0, 20.0, 60.0, 80.0]])
        _, out_boxes, _ = BoxAwareAugment(cfg, enabled=True)(img, boxes, np.array([0]))

        # x1' = W - x2 = 164,  x2' = W - x1 = 214; y is untouched.
        np.testing.assert_allclose(out_boxes[0], [164.0, 20.0, 214.0, 80.0])

    def test_flip_twice_is_identity(self):
        from PIL import Image
        from src.data import BoxAwareAugment

        cfg = Config()
        cfg.aug_hflip_prob = 1.0
        cfg.aug_color_jitter_prob = 0.0
        cfg.aug_scale_translate_prob = 0.0
        aug = BoxAwareAugment(cfg, enabled=True)

        img = Image.new("RGB", (224, 224))
        boxes = np.array([[10.0, 20.0, 60.0, 80.0]])
        _, once, _ = aug(img, boxes, np.array([0]))
        _, twice, _ = aug(img, once, np.array([0]))
        np.testing.assert_allclose(twice[0], boxes[0])

    def test_disabled_augmenter_is_identity(self):
        from PIL import Image
        from src.data import BoxAwareAugment

        cfg = Config()
        img = Image.new("RGB", (224, 224))
        boxes = np.array([[10.0, 20.0, 60.0, 80.0]])
        out_img, out_boxes, out_labels = BoxAwareAugment(cfg, enabled=False)(
            img, boxes, np.array([0])
        )
        np.testing.assert_array_equal(out_boxes, boxes)
        assert out_img is img

    def test_boxes_stay_inside_frame_after_scaling(self):
        from PIL import Image
        from src.data import BoxAwareAugment

        cfg = Config()
        cfg.aug_hflip_prob = 0.0
        cfg.aug_color_jitter_prob = 0.0
        cfg.aug_scale_translate_prob = 1.0

        img = Image.new("RGB", (224, 224))
        rng = np.random.default_rng(11)
        for _ in range(25):
            xy = rng.uniform(0, 150, size=2)
            boxes = np.array([[xy[0], xy[1], xy[0] + 50, xy[1] + 50]])
            _, out_boxes, _ = BoxAwareAugment(cfg, enabled=True)(img, boxes, np.array([0]))
            for box in out_boxes:
                assert -1e-6 <= box[0] <= 224 and -1e-6 <= box[1] <= 224
                assert box[2] <= 224 + 1e-6 and box[3] <= 224 + 1e-6
                assert box[2] > box[0] and box[3] > box[1]
