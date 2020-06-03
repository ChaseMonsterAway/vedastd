import torch.nn as nn

from .backbone.bulder import build_backbone
from .enhancemodule.builder import build_enhance
from .head.builder import build_head
from .registry import MODELS


@MODELS.register_module
class GModel(nn.Module):
    def __init__(self, backbone, enhance, head):
        super(GModel, self).__init__()

        self.body = build_backbone(backbone)
        self.enhance = build_decoder(enhance)
        self.head = build_head(head)

    def forward(self, img):
        x = self.body(img)

        out = self.head(x)

        return out
