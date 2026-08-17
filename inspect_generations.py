"""One-off script: load the trained checkpoint and look at real generations.

Not part of the installed package - a throwaway inspection tool, run directly
with `python inspect_generations.py` from the repo root (conda env active).

Rebuilds the SAME dataset the training run used (same overrides) purely to
recover the tokenizer's vocabulary - SimpleTokenizer.fit() is deterministic
and the checkpoint itself only stores model weights, not the vocab mapping.
If you trained with different overrides than the ones below, update them to
match exactly, or this will decode against the wrong vocabulary.
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
    # Must match the training run's overrides exactly.
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
    tok = dataset.tokenizer

    model_cfg = NanoVLMConfig.from_hydra(cfg)
    model = build_nanovlm(model_cfg)
    ckpt_path = project_root() / "outputs" / "checkpoints" / "last.pt"
    load_checkpoint(ckpt_path, model, map_location="cpu")
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    n_show = 5
    for i in range(n_show):
        item = dataset[i]
        image = item["image"].unsqueeze(0).to(device)
        prompt_ids = torch.tensor([[tok.BOS]], dtype=torch.long, device=device)
        with torch.no_grad():
            out_ids = model.generate(
                image, prompt_ids, max_new_tokens=40, eos_id=tok.EOS
            )
        generated = tok.decode(out_ids[0].tolist())
        print(f"--- image_id={item['image_id']} ---")
        print("ground truth :", item["text"])
        print("generated    :", generated)
        print()


if __name__ == "__main__":
    main()