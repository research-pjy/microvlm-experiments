"""Checkpoint save/load."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def save_checkpoint(path: Path, model: nn.Module, extra: dict | None = None) -> None:
    """Write a state_dict plus optional metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "extra": extra or {}}
    torch.save(payload, path)


def load_checkpoint(path: Path, model: nn.Module, map_location: str = "cpu") -> dict:
    """Load weights into ``model`` and return the extra metadata dict."""

    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    return payload.get("extra", {})
