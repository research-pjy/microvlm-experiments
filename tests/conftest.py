"""Shared pytest helpers."""

from __future__ import annotations

import pytest
import torch

from microvlm.models.config import NanoVLMConfig


@pytest.fixture
def mini_cfg() -> NanoVLMConfig:
    """Mini Table-1 config with arch_a."""

    return NanoVLMConfig(encoder_arch="arch_a")


@pytest.fixture
def dummy_image() -> torch.Tensor:
    """One random 224×224 RGB image batch."""

    torch.manual_seed(0)
    return torch.rand(1, 3, 224, 224)
