"""Evaluate a completion with ROUGE-1 and optional local Ollama judge."""

from __future__ import annotations

import logging

from omegaconf import DictConfig

from microvlm.evaluation.metrics.rouge import rouge1_f1
from microvlm.utils.env_check import require_microvlm_env
from microvlm.utils.hydra_compose import compose_config
from microvlm.utils.logging_config import configure_logging


def run(cfg: DictConfig) -> None:
    """Smoke evaluation on placeholder strings (full eval is in experiments)."""

    require_microvlm_env()
    configure_logging()
    del cfg
    score = rouge1_f1("a cat on a mat", "a cat sits on a mat")
    logging.getLogger(__name__).info("demo rouge1=%s", score)


def main() -> None:
    """CLI entry."""

    import sys

    cfg = compose_config(overrides=sys.argv[1:])
    run(cfg)


if __name__ == "__main__":
    main()
