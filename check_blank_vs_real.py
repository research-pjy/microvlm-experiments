"""Decisive test: does the encoder's output depend on actual pixel content?

Compares a real image's embedding against a blank (all-zero) image and a
pure-noise image, using the same trained encoder. If blank/noise embeddings
are nearly as large and nearly as close to the real one as real images are
to each other, the encoder isn't meaningfully using pixel content at all.

Standalone, not part of the installed package. Run from the repo root with
the conda env active.
"""

from __future__ import annotations

import torch

from microvlm.experiments._support import dataset_for_cfg
from microvlm.models.config import NanoVLMConfig
from microvlm.models.nanovlm import build_nanovlm
from microvlm.training.checkpoint import load_checkpoint
from microvlm.utils.hydra_compose import compose_config
from microvlm.utils.paths import project_root


def main() -> None:
    cfg = compose_config(
        overrides=[
            "model=base",
            "data=coco",
            "data.root_path=/home/jayanth/Research/datasets/coco",
            "training=local_gpu",
            "training.max_samples=5000",
        ]
    )
    dataset = dataset_for_cfg(cfg)
    model_cfg = NanoVLMConfig.from_hydra(cfg)
    model = build_nanovlm(model_cfg)
    ckpt_path = project_root() / "outputs" / "checkpoints" / "last.pt"
    load_checkpoint(ckpt_path, model, map_location="cpu")
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    real_image = dataset[0]["image"].unsqueeze(0).to(device)
    blank_image = torch.zeros_like(real_image)
    noise_image = torch.rand_like(real_image)

    with torch.no_grad():
        v_real = model.encoder.encode(real_image)[0]
        v_blank = model.encoder.encode(blank_image)[0]
        v_noise = model.encoder.encode(noise_image)[0]

    print(f"real image ({dataset[0]['image_id']}) embedding norm: {v_real.norm().item():.4f}")
    print(f"blank (all-zero) image embedding norm:               {v_blank.norm().item():.4f}")
    print(f"pure-noise image embedding norm:                     {v_noise.norm().item():.4f}")
    print()
    print(f"distance real vs blank:  {(v_real - v_blank).norm().item():.4f}")
    print(f"distance real vs noise:  {(v_real - v_noise).norm().item():.4f}")
    print(f"distance blank vs noise: {(v_blank - v_noise).norm().item():.4f}")
    print()
    print("For reference, distances between two DIFFERENT REAL images were")
    print("roughly 0.3 to 1.9 in the last check_encoder_differentiation.py run.")


if __name__ == "__main__":
    main()