"""Training guardrail and one CPU step on random tensors."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from microvlm.models.config import NanoVLMConfig
from microvlm.models.nanovlm import build_nanovlm
from microvlm.training.device import guard_cpu_sample_limit
from microvlm.training.loop import train_one_epoch


def test_cpu_guard_raises() -> None:
    """More than 5 CPU samples must raise RuntimeError."""

    with pytest.raises(RuntimeError, match="dgx_full"):
        guard_cpu_sample_limit("cpu", 6)


def test_cpu_guard_allows_five() -> None:
    """Five samples on CPU is the local smoke limit."""

    guard_cpu_sample_limit("cpu", 5)


def test_one_train_step() -> None:
    """One Adam step on a tiny random batch produces a finite loss."""

    cfg = NanoVLMConfig(encoder_arch="arch_a", max_seq_len=32, vocab_size=128)
    model = build_nanovlm(cfg)
    images = torch.rand(1, 3, 224, 224)
    tokens = torch.randint(0, 128, (1, 6))
    targets = tokens.clone()

    class _DS(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, idx: int) -> dict:
            return {"image": images[0], "token_ids": tokens[0], "targets": targets[0]}

    loader = torch.utils.data.DataLoader(_DS(), batch_size=1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = train_one_epoch(model, loader, opt, torch.device("cpu"))
    assert losses and math_isfinite(losses[0])


def math_isfinite(x: float) -> bool:
    import math

    return math.isfinite(x)


def test_omega_training_cfg_parses() -> None:
    """Hydra-like training node is readable."""

    cfg = OmegaConf.create({"training": {"device": "cpu", "max_samples": 5}})
    assert cfg.training.max_samples == 5
