"""Conv-then-patch visual encoder (Figure 4 + strided-conv ViT embed).

This module reconciles Figure 4 (two Conv2D layers producing feature maps on
the whole image) with Figure 5's patch-token grid by using two strided
convolutions whose combined stride is 16, yielding a 14×14 map (196 tokens).
It diverges from the Section 2.2.1 wording that patches the raw image first.

In ``NanoVLM_Experiments.txt`` this is Architecture A (image → CNN → tokens →
transformer). Code name remains ``arch_b`` per the scaffolding spec.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from microvlm.models.config import NanoVLMConfig
from microvlm.models.vision_encoder.base import BaseVisionEncoder


class ArchBEncoder(BaseVisionEncoder):
    """Two strided Conv2D layers on the full 224×224 image, then tokens.

    Conv1/Conv2: kernel 8, stride 4, padding 2 → 224→56→14. Channels 3→64→128.
    LayerNorm + ReLU sit between the two convs. A linear layer maps C=128 to
    ``n_embd`` if needed. Subsequent CLS / pos / transformer steps match arch_a.
    """

    def __init__(self, cfg: NanoVLMConfig) -> None:
        super().__init__(cfg)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=8, stride=4, padding=2)
        self.mid_ln = nn.LayerNorm(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=8, stride=4, padding=2)
        self.channel_fc = nn.Linear(128, cfg.n_embd)

    def encode(self, image_tensor: Tensor) -> Tensor:
        """Encode ``[B, 3, 224, 224]`` to ``[B, img_embd_dim]``."""

        h = self.conv1(image_tensor)
        h = F.relu(self.mid_ln(h.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        h = self.conv2(h)
        b, c, gh, gw = h.shape
        tokens = h.flatten(2).transpose(1, 2)
        tokens = self.channel_fc(tokens)
        return self.encode_tokens(tokens)
