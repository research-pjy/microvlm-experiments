"""Check whether the visual encoder produces different embeddings for
different images - decoupled entirely from decoding/generation behavior.

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
            "training.max_samples=2000",
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

    n = 5
    imgs = torch.stack([dataset[i]["image"] for i in range(n)]).to(device)
    with torch.no_grad():
        visual = model.encoder.encode(imgs)  # [n, img_embd_dim]

    print("Per-image embedding norm (sanity check, should be nonzero):")
    for i in range(n):
        print(f"  image {i} ({dataset[i]['image_id']}): norm={visual[i].norm().item():.4f}")

    print("\nPairwise L2 distance between different images' embeddings:")
    for i in range(n):
        for j in range(i + 1, n):
            dist = (visual[i] - visual[j]).norm().item()
            print(f"  image {i} vs {j}: {dist:.4f}")


if __name__ == "__main__":
    main()