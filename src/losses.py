"""Multi-task detection loss for the from-scratch grid detector.

    L = lambda_coord * L_loc
      + lambda_obj   * L_obj
      + lambda_noobj * L_noobj
      + lambda_cls   * L_cls

Detection is three problems at once - is anything here, where exactly, and
what is it - so the loss is a weighted sum of three different objectives
computed over different subsets of the grid.

Where each term applies
-----------------------
`L_loc` and `L_cls` are evaluated **only on cells that own an object**. A cell
containing background has no meaningful box to regress or class to name, so
including it would train the network towards an arbitrary target. `L_obj` and
`L_noobj` split the objectness term by the same mask, which is what lets the
two be weighted differently.

The class-imbalance problem
---------------------------
At S=14 the grid has 196 cells while a training image holds 7.4 objects on
average (median 4, max 56 - measured in notebook section 3), so roughly **96%
of cells are empty**. Weighting every cell equally would let the background
term dominate the gradient and drive the model to the trivial "predict nothing
anywhere" solution - which scores well on loss and detects nothing. Two
mechanisms counter this: `lambda_noobj` (0.5) down-weights the background
objectness term, and the detection head's bias is initialised negative
(see `ScratchGridDetector._init_weights`) so the model starts out already
predicting "empty" and does not have to spend early epochs getting there.

Note the two terms are normalised by their own counts (`n_obj` and `n_noobj`)
rather than by a shared total. Dividing the background term by ~188 cells and
the object term by ~7 keeps each a per-cell average, so the balance between
them is set by the lambdas alone and does not drift with scene density.

Reference: Redmon et al., "You Only Look Once" (CVPR 2016), whose
lambda_coord / lambda_noobj weighting scheme this follows.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GridDetectionLoss"]


class GridDetectionLoss(nn.Module):
    """Loss for [B, S, S, 5+C] raw logits against encoded targets.

    Predictions arrive as **logits**; every activation is applied inside this
    module using the fused `*_with_logits` forms. Doing it here rather than in
    the model keeps the computation numerically stable: `BCEWithLogitsLoss`
    uses the log-sum-exp trick internally, whereas `sigmoid` followed by a
    separate `BCELoss` can produce `log(0)` and hand back NaN gradients once
    the model becomes confident.
    """

    def __init__(
        self,
        lambda_coord: float = 5.0,
        lambda_obj: float = 1.0,
        lambda_noobj: float = 0.5,
        lambda_cls: float = 1.0,
    ):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_cls = lambda_cls

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Returns (total_loss, per-term breakdown for logging)."""
        if predictions.shape != targets.shape:
            raise ValueError(
                f"prediction shape {tuple(predictions.shape)} != "
                f"target shape {tuple(targets.shape)}"
            )

        obj_mask = targets[..., 0] > 0.5                   # [B, S, S]
        noobj_mask = ~obj_mask
        n_obj = obj_mask.sum().clamp(min=1).float()
        n_noobj = noobj_mask.sum().clamp(min=1).float()

        # ---- objectness: BCE over every cell, split by mask --------------
        obj_logits = predictions[..., 0]
        obj_target = targets[..., 0]
        bce = F.binary_cross_entropy_with_logits(
            obj_logits, obj_target, reduction="none"
        )
        loss_obj = (bce * obj_mask).sum() / n_obj
        loss_noobj = (bce * noobj_mask).sum() / n_noobj

        # ---- localisation: object cells only -----------------------------
        if obj_mask.any():
            pred_txy = torch.sigmoid(predictions[..., 1:3][obj_mask])
            true_txy = targets[..., 1:3][obj_mask]

            pred_wh = torch.sigmoid(predictions[..., 3:5][obj_mask])
            true_wh = targets[..., 3:5][obj_mask]

            # Regress the square root of width/height rather than the raw
            # value. A 10-pixel error on a 20-pixel starfish matters far more
            # than the same error on a 200-pixel shark, but an L1 loss on raw
            # w/h treats them identically. sqrt compresses the large end and
            # expands the small end, so small objects - most of this dataset -
            # get a proportionate share of the gradient. (YOLOv1, sec. 2.2)
            eps = 1e-6
            loss_wh = F.smooth_l1_loss(
                torch.sqrt(pred_wh + eps), torch.sqrt(true_wh + eps), reduction="sum"
            )
            loss_xy = F.smooth_l1_loss(pred_txy, true_txy, reduction="sum")
            loss_loc = (loss_xy + loss_wh) / n_obj

            # ---- classification: object cells only -----------------------
            cls_logits = predictions[..., 5:][obj_mask]
            cls_target = targets[..., 5:][obj_mask].argmax(dim=-1)
            loss_cls = F.cross_entropy(cls_logits, cls_target, reduction="sum") / n_obj
        else:
            # An all-background batch still trains the noobj term.
            zero = predictions.sum() * 0.0
            loss_loc, loss_cls = zero, zero

        total = (
            self.lambda_coord * loss_loc
            + self.lambda_obj * loss_obj
            + self.lambda_noobj * loss_noobj
            + self.lambda_cls * loss_cls
        )

        breakdown = {
            "loss": float(total.detach()),
            "loc": float(loss_loc.detach()),
            "obj": float(loss_obj.detach()),
            "noobj": float(loss_noobj.detach()),
            "cls": float(loss_cls.detach()),
            "n_obj": int(n_obj.item()),
        }
        return total, breakdown
