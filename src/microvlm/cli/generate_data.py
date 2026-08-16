"""Generate teacher datasets from the local fixture or a COCO path in config."""

from __future__ import annotations

import logging

from omegaconf import DictConfig

from microvlm.data.generation.dataset_builder import build_dataset
from microvlm.data.teachers.factory import build_teacher
from microvlm.experiments._support import processed_path, require_fixture
from microvlm.utils.env_check import require_microvlm_env
from microvlm.utils.hydra_compose import compose_config
from microvlm.utils.logging_config import configure_logging


def run(cfg: DictConfig) -> None:
    """Generate JSONL for ``cfg.data.prompt_id`` using ``cfg.teacher``."""

    require_microvlm_env()
    configure_logging()
    fixtures = require_fixture()
    teacher = build_teacher(cfg.teacher)
    prompt_id = str(cfg.data.prompt_id)
    out = processed_path(f"{prompt_id}_{teacher.name}.jsonl")
    records = build_dataset(fixtures, teacher, prompt_id, out)
    logging.getLogger(__name__).info("wrote %s records to %s", len(records), out)


def main() -> None:
    """Hydra-free CLI using compose_config + optional argv overrides."""

    import sys

    overrides = sys.argv[1:]
    cfg = compose_config(overrides=overrides)
    run(cfg)


if __name__ == "__main__":
    main()
