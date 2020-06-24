import torch.nn as nn

from .backbone.bulder import build_backbone
from .enhancemodule.builder import build_enhance
from .head.builder import build_head
from .enhancemodule.bricks import build_brick
from .registry import MODELS


@MODELS.register_module
class GModel(nn.Module):
    def __init__(self, backbone=None, enhance=None, collect=None, fusion=None, head=None):
        super(GModel, self).__init__()

        self.body = build_backbone(backbone) if backbone else None
        self.enhance = build_enhance(enhance) if enhance else None
        self.fusion = build_brick(fusion) if fusion else None
        self.collect = build_brick(collect) if collect else None
        self.head = build_head(head) if head else None

    def forward(self, img):
        feature = self.body(img)
        if self.enhance:
            feature = self.enhance(feature)
        if self.fusion:
            feature = self.fusion(feature)
        if self.head:
            feature = self.head(feature)

        return feature

@MODELS.register_module
class PseNet(nn.Module):
    def __init__(self, backbone=None, enhance=None, collect=None, fusion=None, head=None):
        super(PseNet, self).__init__()

        self.body = build_backbone(backbone) if backbone else None
        self.enhance = build_enhance(enhance) if enhance else None
        self.fusion = build_brick(fusion) if fusion else None
        self.collect = build_brick(collect) if collect else None
        self.head = build_head(head) if head else None

    def forward(self, img):
        feature = self.body(img)
        if self.enhance:
            feature = self.enhance(feature)
        if self.fusion:
            feature = self.fusion(feature)
        if self.head:
            feature = self.head(feature)

        return feature
