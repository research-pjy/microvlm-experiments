"""Device resolution and the CPU sample-count safety rail."""

from __future__ import annotations

import torch
from omegaconf import DictConfig


def resolve_device(requested: str) -> torch.device:
    """Return a torch device, falling back to CPU if CUDA is requested but absent."""

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "training.device=cuda but CUDA is not available. "
            "Use training=local_smoke on this Ubuntu machine."
        )
    return torch.device(requested)


def guard_cpu_sample_limit(device: torch.device | str, n_samples: int) -> None:
    """Refuse CPU training on more than 5 samples (deliberate safety rail).

    Raises:
        RuntimeError: If device is CPU and ``n_samples > 5``.
    """

    name = device if isinstance(device, str) else device.type
    if name == "cpu" and n_samples > 5:
        raise RuntimeError(
            f"Refusing to train on {n_samples} samples on CPU. "
            "Reduce the dataset to ≤5 images (training=local_smoke) or switch to "
            "training=dgx_full with a CUDA device."
        )


def guard_from_config(cfg: DictConfig, n_samples: int) -> torch.device:
    """Resolve device from Hydra training config and apply the CPU guard."""

    device = resolve_device(str(cfg.training.device))
    guard_cpu_sample_limit(device, n_samples)
    return device
