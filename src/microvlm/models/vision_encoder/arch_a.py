"""Patch-then-conv visual encoder (literal Section 2.2.1).

This module matches the paper *text* in Section 2.2.1: split 224×224×3 into
196 non-overlapping 16×16 patches via unfold (not a conv), apply two Conv2D
layers independently to each patch, flatten, and project to one token per
patch. It necessarily diverges from Figure 4 (which shows convs on the whole
image) and from Figure 5 (standard ViT linear patch embed).

In ``NanoVLM_Experiments.txt`` this is Architecture B (patches → CNN on
patches → transformer). Code name remains ``arch_a`` per the scaffolding spec.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from microvlm.models.config import NanoVLMConfig
from microvlm.models.vision_encoder.base import BaseVisionEncoder


class ArchAEncoder(BaseVisionEncoder):
    """Per-patch Conv2D → Conv2D → LayerNorm → ReLU → FC token.

    Channel counts 3→32→64 with 3×3 kernels and padding 1 keep the 16×16
    spatial size (the spec forbids collapsing spatial dims to nothing). The
    flattened conv map is 64*16*16 = 16384 units, then a linear layer maps
    to ``n_embd``.
    """

    def __init__(self, cfg: NanoVLMConfig) -> None:
        super().__init__(cfg)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.patch_ln = nn.LayerNorm([64, cfg.patch_size, cfg.patch_size])
        flat = 64 * cfg.patch_size * cfg.patch_size
        self.patch_fc = nn.Linear(flat, cfg.n_embd)

    def encode(self, image_tensor: Tensor) -> Tensor:
        """Encode ``[B, 3, 224, 224]`` to ``[B, img_embd_dim]``."""

        b, c, h, w = image_tensor.shape
        ps = self.cfg.patch_size
        # unfold: [B, 3*ps*ps, 196] then [B*196, 3, ps, ps]
        patches = F.unfold(image_tensor, kernel_size=ps, stride=ps)
        n_patches = patches.size(-1)
        patches = patches.transpose(1, 2).reshape(b * n_patches, c, ps, ps)
        h_map = self.conv2(self.conv1(patches))
        h_map = F.relu(self.patch_ln(h_map))
        tokens = self.patch_fc(h_map.flatten(1)).view(b, n_patches, self.cfg.n_embd)
        return self.encode_tokens(tokens)
