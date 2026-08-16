"""Unit tests for NanoVLM modules (random tensors, no network)."""

from __future__ import annotations

import math

import pytest
import torch

from microvlm.models.config import NanoVLMConfig, count_parameters
from microvlm.models.decoder import prefix_causal_mask
from microvlm.models.nanovlm import build_nanovlm


@pytest.mark.parametrize("arch", ["arch_a", "arch_b"])
def test_forward_shapes(arch: str) -> None:
    """Mini model produces finite logits and a scalar loss."""

    cfg = NanoVLMConfig(encoder_arch=arch, max_seq_len=64)
    model = build_nanovlm(cfg)
    model.eval()
    images = torch.rand(2, 3, 224, 224)
    tokens = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = tokens.clone()
    logits, loss = model(images, tokens, targets=targets)
    assert logits.shape[0] == 2
    assert logits.shape[-1] == cfg.vocab_size
    assert loss is not None and math.isfinite(float(loss))


def test_prefix_mask() -> None:
    """Text may see all visual tokens; text is causal among itself."""

    mask = prefix_causal_mask(2, 3, torch.device("cpu"))
    assert mask.shape == (5, 5)
    assert bool(mask[2, 0]) and bool(mask[2, 1])
    assert not bool(mask[2, 4])
    assert bool(mask[4, 2])


def test_mini_param_order_of_magnitude() -> None:
    """Mini should be nearer 5M than 50M (Table 1 is approximate)."""

    model = build_nanovlm(NanoVLMConfig(encoder_arch="arch_a"))
    n = count_parameters(model)
    assert 1_000_000 < n < 20_000_000
