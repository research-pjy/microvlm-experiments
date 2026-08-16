"""Visual-textual connector: Linear + GELU, then sequence concatenation."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from microvlm.models.config import NanoVLMConfig


class Connector(nn.Module):
    """Project visual embeddings into the text embedding space (Section 2.2.2)."""

    def __init__(self, cfg: NanoVLMConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(cfg.img_embd_dim, cfg.n_embd)
        self.act = nn.GELU()

    def project(self, visual: Tensor) -> Tensor:
        """Project ``[B, img_embd_dim]`` or ``[B, T_v, img_embd_dim]`` to ``n_embd``.

        A rank-2 encoder output is treated as a single visual token.
        """

        if visual.dim() == 2:
            visual = visual.unsqueeze(1)
        return self.act(self.proj(visual))

    def fuse(self, visual: Tensor, text_embeddings: Tensor) -> Tensor:
        """Concatenate projected visual tokens with text token embeddings."""

        return torch.cat([self.project(visual), text_embeddings], dim=1)
