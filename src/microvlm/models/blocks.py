"""Pre-norm Transformer block shared by encoder and decoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with an optional additive mask."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        head_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.head_size = head_size
        inner = n_head * head_size
        self.qkv = nn.Linear(n_embd, 3 * inner)
        self.proj = nn.Linear(inner, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """Apply attention.

        Args:
            x: ``[batch, seq, n_embd]``.
            attn_mask: Boolean mask ``[seq, seq]`` or ``[batch, 1, seq, seq]``
                where True means the location **may be attended**.
        """

        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_head, self.head_size)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = self.head_size**-0.5
        att = (q @ k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.view(1, 1, t, t)
            att = att.masked_fill(~attn_mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = torch.nan_to_num(att, nan=0.0)
        att = self.attn_drop(att)
        out = (
            (att @ v)
            .transpose(1, 2)
            .contiguous()
            .view(b, t, self.n_head * self.head_size)
        )
        return self.resid_drop(self.proj(out))


class TransformerBlock(nn.Module):
    """Pre-norm block: LN → MHA → residual → LN → MLP → residual.

    Used by both the visual encoder (non-causal) and the language decoder
    (causal / prefix-causal via ``attn_mask``).
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        head_size: int,
        dropout: float,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.causal = causal
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, head_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """Run one transformer block.

        If ``causal`` is True and no mask is passed, a lower-triangular mask is
        built. The decoder should pass a prefix-LM mask instead.
        """

        t = x.size(1)
        mask = attn_mask
        if mask is None and self.causal:
            mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
        x = x + self.attn(self.ln1(x), attn_mask=mask)
        x = x + self.mlp(self.ln2(x))
        return x
