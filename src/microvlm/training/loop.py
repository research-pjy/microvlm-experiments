"""Single-epoch / multi-epoch PyTorch train loop."""

from __future__ import annotations

import logging

import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> list[float]:
    """Run one epoch; return per-step loss values."""

    model.train()
    losses: list[float] = []
    for batch in loader:
        images = batch["image"].to(device)
        token_ids = batch["token_ids"].to(device)
        targets = batch["targets"].to(device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(images, token_ids, targets=targets)
        if loss is None:
            raise RuntimeError("training step returned no loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        logger.debug("step loss=%s", losses[-1])
    return losses
