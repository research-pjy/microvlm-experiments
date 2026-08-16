"""Causal language decoder over a multimodal token sequence (Section 2.2.3)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from microvlm.models.blocks import TransformerBlock
from microvlm.models.config import NanoVLMConfig


def prefix_causal_mask(n_visual: int, n_text: int, device: torch.device) -> Tensor:
    """Build a prefix-LM attention mask.

    Visual positions attend to all visual positions (context, not targets).
    Text positions attend to all visual positions and to text positions ≤ i.
    Text cannot attend to future text.
    """

    t = n_visual + n_text
    mask = torch.zeros(t, t, dtype=torch.bool, device=device)
    mask[:n_visual, :n_visual] = True
    mask[n_visual:, :n_visual] = True
    text_causal = torch.tril(
        torch.ones(n_text, n_text, dtype=torch.bool, device=device)
    )
    mask[n_visual:, n_visual:] = text_causal
    return mask


class Decoder(nn.Module):
    """Positional embeddings → causal transformer blocks → vocab logits."""

    def __init__(self, cfg: NanoVLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    cfg.n_embd,
                    cfg.n_head,
                    cfg.head_size,
                    cfg.dropout,
                    causal=True,
                )
                for _ in range(cfg.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def embed_text(self, token_ids: Tensor) -> Tensor:
        """Embed text token ids to ``[B, T, n_embd]``."""

        return self.tok_emb(token_ids)

    def forward(
        self,
        multimodal: Tensor,
        n_visual: int,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Decode a fused visual+text sequence.

        Args:
            multimodal: ``[B, T_v + T_text, n_embd]`` (already concatenated).
            n_visual: Number of leading visual tokens (loss is skipped there).
            targets: Optional text token ids ``[B, T_text]`` for teacher forcing.
                Loss is cross-entropy on text positions only.

        Returns:
            ``(logits, loss)`` where logits cover the full sequence and loss is
            None if ``targets`` is omitted.
        """

        b, t, _ = multimodal.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {t} exceeds max_seq_len={self.cfg.max_seq_len}"
            )
        pos = torch.arange(t, device=multimodal.device)
        x = self.drop(multimodal + self.pos_emb(pos)[None, :, :])
        n_text = t - n_visual
        mask = prefix_causal_mask(n_visual, n_text, multimodal.device)
        for blk in self.blocks:
            x = blk(x, attn_mask=mask)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            text_logits = logits[:, n_visual:, :]
            loss = nn.functional.cross_entropy(
                text_logits.reshape(-1, text_logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss
