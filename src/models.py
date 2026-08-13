"""The two detectors compared in this project.

Model 1 - ScratchGridDetector
    A single-shot, anchor-free grid detector written from first principles.
    No pretrained weights, no detection library: the architecture, the target
    encoding, the loss, the decoding, IoU and NMS are all implemented in this
    repository.

Model 2 - Faster R-CNN (ResNet-50 + FPN), COCO-pretrained, fine-tuned
    A two-stage detector adapted to the 7 aquarium classes by replacing the
    box predictor and fine-tuning. This is the "modern course technique" half
    of the comparison (transfer learning / fine-tuning).

The two are deliberately opposite designs - dense single-pass regression
versus sparse propose-then-classify - which is what makes the accuracy /
latency trade-off in the results interpretable rather than arbitrary.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .config import IMAGENET_MEAN, IMAGENET_STD, Config

__all__ = ["ScratchGridDetector", "build_faster_rcnn", "build_model"]


# --------------------------------------------------------------------------
# Model 1
# --------------------------------------------------------------------------
def conv_bn_relu(in_ch: int, out_ch: int, kernel: int = 3) -> nn.Sequential:
    """Conv -> BatchNorm -> ReLU.

    BatchNorm normalises each channel over the batch to zero mean and unit
    variance, then rescales with two learned parameters. It matters a lot
    here: without it, a network this deep trained from random initialisation
    on only ~450 images needs a very small learning rate to stay stable, and
    converges far too slowly to reach a useful result in a feasible number of
    epochs. The convolution bias is omitted because BatchNorm's shift
    parameter makes it redundant.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ScratchGridDetector(nn.Module):
    """Single-shot grid detector, built from scratch.

    Architecture
    ------------
    Five convolutional stages, each doubling the channel count; the first four
    halve the spatial resolution with max-pooling:

        input      3 x 224 x 224
        stage 1   32 x 112 x 112     (2 conv + pool)
        stage 2   64 x  56 x  56     (2 conv + pool)
        stage 3  128 x  28 x  28     (2 conv + pool)
        stage 4  256 x  14 x  14     (2 conv + pool)
        stage 5  512 x  14 x  14     (2 conv, no pool)
        head    (5+C) x 14 x 14      (1x1 conv)

    The output grid size falls out of the architecture: four poolings take
    224 down to 14, so S = 14 by construction rather than by an arbitrary
    resize. Every cell is responsible for objects whose centre lands inside
    its 16x16-pixel footprint.

    The head is a 1x1 convolution rather than a flatten-plus-linear layer (as
    in the original YOLOv1). This keeps the detector fully convolutional: the
    same 5+C predictor is applied at every cell with shared weights, which
    cuts the parameter count by an order of magnitude and means a cell's
    prediction depends only on its own receptive field instead of on where it
    happens to sit in a flattened vector.

    Output
    ------
    Raw logits of shape [B, S, S, 5+C]. Activations are applied in
    `data.decode_grid_predictions` (inference) and inside the loss (training),
    so the numerically stable `*_with_logits` formulations can be used.
    """

    def __init__(self, num_classes: int = 7, grid_size: int = 14, in_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.out_per_cell = 5 + num_classes

        # Normalisation lives inside the model as a non-trainable buffer, so
        # it travels with the checkpoint. A saved model can therefore be
        # handed a plain [0,1] image tensor and will do the right thing -
        # there is no external preprocessing step to forget at demo time.
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        self.stage1 = nn.Sequential(conv_bn_relu(in_channels, 32), conv_bn_relu(32, 32), nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(conv_bn_relu(32, 64), conv_bn_relu(64, 64), nn.MaxPool2d(2))
        self.stage3 = nn.Sequential(conv_bn_relu(64, 128), conv_bn_relu(128, 128), nn.MaxPool2d(2))
        self.stage4 = nn.Sequential(conv_bn_relu(128, 256), conv_bn_relu(256, 256), nn.MaxPool2d(2))
        self.stage5 = nn.Sequential(conv_bn_relu(256, 512), conv_bn_relu(512, 512))

        self.dropout = nn.Dropout2d(p=0.1)
        self.head = nn.Conv2d(512, self.out_per_cell, kernel_size=1)

        # If the grid the config asks for is not what the convolutions
        # produce, adapt the spatial size explicitly rather than silently
        # emitting a differently-shaped grid than the targets expect.
        self.resize_to_grid = nn.AdaptiveAvgPool2d((grid_size, grid_size))

        self._init_weights()

    def _init_weights(self) -> None:
        """He initialisation for the ReLU stacks; a small-variance output head.

        Two different schemes, for two different jobs:

        * **Backbone convolutions** get He (Kaiming) initialisation, which sets
          the weight variance to 2/fan_out so that activation variance is
          preserved through a ReLU stack instead of shrinking or exploding
          layer by layer.

        * **The detection head** does *not*. It is a linear output layer, not a
          link in a ReLU chain, and He initialisation over its 512 input
          channels produces logits of magnitude ~10 at initialisation - which
          would completely swamp the bias prior set below and start training
          from a saturated sigmoid with near-zero gradient. A small normal
          (std 0.01) keeps the initial logits close to the bias.

        The objectness bias is then set to -4 (sigmoid ~ 0.018) because the
        grid is overwhelmingly background: at S=14 there are 196 cells while a
        training image averages 7.4 objects, so ~96% of cells are empty.
        Starting from "assume empty" prevents an initial flood of false
        positives whose gradients would otherwise dominate the first epochs -
        the prior-bias trick from RetinaNet's focal-loss paper.

        `tests/test_core.py::TestModel::test_objectness_bias_starts_negative`
        checks that the prior actually survives to the output.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.head.bias)
        nn.init.constant_(self.head.bias[0], -4.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] in [0,1]  ->  [B, S, S, 5+C] raw logits."""
        x = (x - self.pixel_mean) / self.pixel_std

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        x = self.resize_to_grid(x)
        x = self.dropout(x)
        x = self.head(x)

        # [B, 5+C, S, S] -> [B, S, S, 5+C] so the last axis is the prediction
        # vector, matching the target layout produced by encode_grid_targets.
        return x.permute(0, 2, 3, 1).contiguous()

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self(images)


# --------------------------------------------------------------------------
# Model 2
# --------------------------------------------------------------------------
def build_faster_rcnn(
    num_classes: int = 7,
    img_size: int = 224,
    trainable_backbone_layers: int = 3,
    pretrained: bool = True,
) -> torch.nn.Module:
    """Faster R-CNN with a COCO-pretrained ResNet-50 FPN backbone.

    Adaptation performed (the guide requires pretrained weights to be
    *meaningfully* adapted, not merely executed):

    1. The 91-class COCO box predictor is discarded and replaced with a fresh
       `FastRCNNPredictor` sized for our 7 classes + background. Its weights
       are randomly initialised - none of COCO's class semantics survive.
    2. Only the top `trainable_backbone_layers` ResNet stages are unfrozen.
       The early layers encode edges and textures that transfer as-is; with
       ~450 training images, fine-tuning all of them overfits quickly and
       destroys the pretrained features. Three is the torchvision default and
       a reasonable middle ground for a dataset this size.
    3. The internal transform is pinned to our fixed 224x224 input so that
       both models genuinely see the same resolution. Left at its defaults,
       torchvision would resize images to a 800-pixel minimum side - the
       comparison would then be 224px against 800px, and the latency and
       accuracy gap would mostly be measuring that, not the architectures.

    Note the model normalises internally with the ImageNet statistics we pass
    it, exactly matching Model 1's internal normalisation.
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None

    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        trainable_backbone_layers=trainable_backbone_layers if pretrained else 5,
        min_size=img_size,
        max_size=img_size,
        image_mean=list(IMAGENET_MEAN),
        image_std=list(IMAGENET_STD),
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

    return model


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------
def build_model(name: str, cfg: Config) -> torch.nn.Module:
    """Build either model from its short name."""
    name = name.lower().replace("-", "_")
    if name in {"scratch", "model1", "grid"}:
        return ScratchGridDetector(
            num_classes=cfg.num_classes, grid_size=cfg.grid_size
        )
    if name in {"fasterrcnn", "faster_rcnn", "model2"}:
        return build_faster_rcnn(
            num_classes=cfg.num_classes,
            img_size=cfg.img_size,
            trainable_backbone_layers=cfg.m2_trainable_backbone_layers,
        )
    raise ValueError(f"Unknown model '{name}'. Use 'scratch' or 'fasterrcnn'.")
