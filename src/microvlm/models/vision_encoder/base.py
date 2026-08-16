"""Shared visual-encoder interface: image tensor → ``[batch, img_embd_dim]``."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from microvlm.models.blocks import TransformerBlock
from microvlm.models.config import NanoVLMConfig


class BaseVisionEncoder(nn.Module, ABC):
    """Common CLS-token transformer stack after patch/conv embedding."""

    def __init__(self, cfg: NanoVLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        n_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.n_patches = n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.n_embd))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, cfg.n_embd))
        self.pos_ln = nn.LayerNorm(cfg.n_embd)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    cfg.n_embd,
                    cfg.n_head,
                    cfg.head_size,
                    cfg.dropout,
                    causal=False,
                )
                for _ in range(cfg.n_blks)
            ]
        )
        self.cls_proj = nn.Linear(cfg.n_embd, cfg.img_embd_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def encode_tokens(self, patch_tokens: Tensor) -> Tensor:
        """Prepend CLS, add positions, run transformer, project CLS.

        Args:
            patch_tokens: ``[batch, 196, n_embd]``.

        Returns:
            ``[batch, img_embd_dim]`` image representation.
        """

        b = patch_tokens.size(0)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, patch_tokens], dim=1)
        x = self.pos_ln(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x, attn_mask=None)
        return self.cls_proj(x[:, 0])

    @abstractmethod
    def encode(self, image_tensor: Tensor) -> Tensor:
        """Encode a batch of images to ``[batch, img_embd_dim]``."""

    def forward(self, image_tensor: Tensor) -> Tensor:
        """Alias for :meth:`encode`."""

        return self.encode(image_tensor)
