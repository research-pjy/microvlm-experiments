"""High-level trainer: device guard, optimizer, epochs, loss history."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from microvlm.models.nanovlm import NanoVLM
from microvlm.training.device import guard_cpu_sample_limit, resolve_device
from microvlm.training.loop import train_one_epoch


@dataclass
class TrainResult:
    """Loss curves from a training run."""

    losses: list[float] = field(default_factory=list)
    epoch_means: list[float] = field(default_factory=list)
    device: str = "cpu"


def train_model(
    model: NanoVLM,
    dataset: Dataset,
    cfg: DictConfig,
) -> TrainResult:
    """Train ``model`` according to ``cfg.training``.

    Applies the CPU ≤5-sample guard before the first step.
    """

    n = len(dataset)  # type: ignore[arg-type]
    device = resolve_device(str(cfg.training.device))
    guard_cpu_sample_limit(device, n)
    max_samples = cfg.training.get("max_samples")
    if max_samples is not None and n > int(max_samples):
        raise RuntimeError(
            f"dataset has {n} samples but training.max_samples={max_samples}"
        )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=int(cfg.training.num_workers),
    )
    lr = float(cfg.training.get("learning_rate", cfg.model.get("learning_rate", 1e-3)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to(device)
    all_losses: list[float] = []
    epoch_means: list[float] = []
    epochs = int(cfg.training.epochs)
    for _ in range(epochs):
        step_losses = train_one_epoch(model, loader, optimizer, device)
        all_losses.extend(step_losses)
        epoch_means.append(sum(step_losses) / max(len(step_losses), 1))
    return TrainResult(losses=all_losses, epoch_means=epoch_means, device=str(device))
