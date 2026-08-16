"""Train a NanoVLM from Hydra config (local_smoke or dgx_full)."""

from __future__ import annotations

import logging

from omegaconf import DictConfig

from microvlm.experiments._support import fixture_caption_dataset, require_fixture
from microvlm.models.config import NanoVLMConfig, count_parameters
from microvlm.models.nanovlm import build_nanovlm
from microvlm.training.checkpoint import save_checkpoint
from microvlm.training.trainer import train_model
from microvlm.utils.env_check import require_microvlm_env
from microvlm.utils.hydra_compose import compose_config
from microvlm.utils.logging_config import configure_logging
from microvlm.utils.paths import project_root
from microvlm.utils.seeding import seed_everything


def run(cfg: DictConfig) -> None:
    """Train on fixture captions (iteration 0 local path)."""

    require_microvlm_env()
    configure_logging()
    seed_everything(int(cfg.get("seed", 42)))
    fixtures = require_fixture()
    dataset = fixture_caption_dataset(
        fixtures,
        vocab_size=int(cfg.model.vocab_size),
        image_size=int(cfg.model.image_size),
        max_seq_len=min(64, int(cfg.model.max_seq_len)),
    )
    model = build_nanovlm(NanoVLMConfig.from_hydra(cfg))
    logging.getLogger(__name__).info("params=%s", count_parameters(model))
    result = train_model(model, dataset, cfg)
    ckpt = project_root() / str(cfg.training.checkpoint_dir) / "last.pt"
    save_checkpoint(ckpt, model, extra={"losses": result.losses})
    logging.getLogger(__name__).info(
        "checkpoint %s mean_loss=%s", ckpt, result.epoch_means
    )


def main() -> None:
    """CLI entry."""

    import sys

    cfg = compose_config(overrides=sys.argv[1:])
    run(cfg)


if __name__ == "__main__":
    main()
