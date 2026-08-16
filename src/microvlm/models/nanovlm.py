"""Assemble encoder + connector + decoder into a NanoVLM."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from microvlm.models.config import NanoVLMConfig
from microvlm.models.connector import Connector
from microvlm.models.decoder import Decoder
from microvlm.models.vision_encoder.arch_a import ArchAEncoder
from microvlm.models.vision_encoder.arch_b import ArchBEncoder
from microvlm.models.vision_encoder.base import BaseVisionEncoder


class NanoVLM(nn.Module):
    """Image + text-token ids → logits / optional LM loss."""

    def __init__(
        self,
        cfg: NanoVLMConfig,
        encoder: BaseVisionEncoder,
        connector: Connector,
        decoder: Decoder,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.connector = connector
        self.decoder = decoder

    def forward(
        self,
        images: Tensor,
        token_ids: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Run the full VLM.

        Args:
            images: ``[B, 3, H, W]``.
            token_ids: ``[B, T_text]`` (prompt or teacher-forced sequence).
            targets: Optional ``[B, T_text]`` labels aligned with ``token_ids``.
        """

        visual = self.encoder.encode(images)
        text_emb = self.decoder.embed_text(token_ids)
        fused = self.connector.fuse(visual, text_emb)
        n_visual = fused.size(1) - token_ids.size(1)
        return self.decoder(fused, n_visual=n_visual, targets=targets)

    @torch.no_grad()
    def generate(
        self,
        images: Tensor,
        prompt_ids: Tensor,
        max_new_tokens: int = 32,
        eos_id: int | None = None,
    ) -> Tensor:
        """Greedy decode text tokens given images and a prompt.

        Returns:
            Token ids including the prompt, shape ``[B, T_prompt + new]``.
        """

        self.eval()
        tokens = prompt_ids
        for _ in range(max_new_tokens):
            logits, _ = self.forward(images, tokens, targets=None)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_id], dim=1)
            if eos_id is not None and bool((next_id == eos_id).all()):
                break
            if tokens.size(1) + 1 >= self.cfg.max_seq_len:
                break
        return tokens


def build_encoder(cfg: NanoVLMConfig) -> BaseVisionEncoder:
    """Construct arch_a or arch_b from ``cfg.encoder_arch``."""

    if cfg.encoder_arch == "arch_a":
        return ArchAEncoder(cfg)
    if cfg.encoder_arch == "arch_b":
        return ArchBEncoder(cfg)
    raise ValueError(f"Unknown encoder_arch {cfg.encoder_arch!r}")


def build_nanovlm(config: NanoVLMConfig) -> NanoVLM:
    """Factory: encoder + connector + decoder per mini/base/large + arch."""

    encoder = build_encoder(config)
    connector = Connector(config)
    decoder = Decoder(config)
    return NanoVLM(config, encoder, connector, decoder)
