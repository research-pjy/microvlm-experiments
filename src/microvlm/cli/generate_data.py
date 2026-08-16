"""Generate teacher datasets from the local fixture or a real COCO path in config."""

from __future__ import annotations

import logging

from omegaconf import DictConfig

from microvlm.data.generation.dataset_builder import build_dataset
from microvlm.data.teachers.factory import build_teacher
from microvlm.experiments._support import (
    generated_jsonl_path,
    load_coco_records,
    require_fixture,
)
from microvlm.utils.env_check import require_microvlm_env
from microvlm.utils.hydra_compose import compose_config
from microvlm.utils.logging_config import configure_logging


def run(cfg: DictConfig) -> None:
    """Generate JSONL for ``cfg.data.prompt_id`` using ``cfg.teacher``.

    Source records come from real COCO (``cfg.data.root_path``/``split``)
    for any ``data`` group except ``fixture``, which still uses the 5-image
    fixture — same behavior as before, just explicit about it now.

    ``data=controlled_length`` writes all 5 candidate lengths
    (``cfg.data.lengths``) into the same output file; each length gets its
    own cache key (``{prompt_id}_{length}``) so reruns only fill gaps.
    """

    require_microvlm_env()
    configure_logging()
    log = logging.getLogger(__name__)

    name = str(cfg.data.name)
    records = require_fixture() if name == "fixture" else load_coco_records(cfg)
    teacher = build_teacher(cfg.teacher)
    prompt_id = str(cfg.data.prompt_id)
    out = generated_jsonl_path(cfg, prompt_id)

    if name == "controlled_length":
        for length_n in list(cfg.data.lengths):
            written = build_dataset(
                records, teacher, prompt_id, out, length_n=int(length_n)
            )
            log.info("length=%s total records in %s: %s", length_n, out, len(written))
        return

    written = build_dataset(records, teacher, prompt_id, out)
    log.info("wrote %s records to %s", len(written), out)


def main() -> None:
    """Hydra-free CLI using compose_config + optional argv overrides."""

    import sys

    overrides = sys.argv[1:]
    cfg = compose_config(overrides=overrides)
    run(cfg)


if __name__ == "__main__":
    main()