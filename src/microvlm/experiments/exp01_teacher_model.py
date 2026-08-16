"""Experiment 01 — effect of teacher model.

Generate equivalent ShortDesc datasets using different LLMs (GPT-4o, Claude,
Gemini, Llama, Qwen, and local Ollama), train identical NanoVLM architectures,
and compare convergence / grammar / creativity / consistency / generalization.

Research question: Does the choice of teacher model influence the performance
of tiny Vision-Language Models?
"""

from __future__ import annotations

import logging

import pandas as pd
from omegaconf import DictConfig

from microvlm.data.generation.dataset_builder import build_dataset
from microvlm.data.teachers.factory import build_teacher
from microvlm.experiments._support import (
    load_teacher_cfg,
    processed_path,
    require_fixture,
)
from microvlm.utils.api_keys import load_api_keys

logger = logging.getLogger(__name__)


def run_exp01_teacher_model(cfg: DictConfig) -> pd.DataFrame:
    """Iterate teacher configs; only local Ollama (and llama/qwen presets) run today.

    Other backends raise ``NotImplementedError``, which is recorded as a row
    status rather than aborting the whole experiment.
    """

    fixtures = require_fixture()
    keys = load_api_keys()
    names = list(cfg.get("teacher_names", ["local_ollama"]))
    rows: list[dict] = []
    for name in names:
        teacher_cfg = load_teacher_cfg(name)
        try:
            teacher = build_teacher(teacher_cfg, api_keys=keys)
            out = processed_path(f"exp01_{name}_shortdesc.jsonl")
            records = build_dataset(fixtures, teacher, "shortdesc", out)
            rows.append(
                {
                    "teacher": name,
                    "backend": str(teacher_cfg.get("backend", name)),
                    "n_records": len(records),
                    "mean_word_count": (
                        sum(r.word_count for r in records) / max(len(records), 1)
                    ),
                    "status": "ok",
                    "error": "",
                }
            )
        except NotImplementedError as exc:
            logger.info("teacher %s stubbed: %s", name, exc)
            rows.append(
                {
                    "teacher": name,
                    "backend": str(teacher_cfg.get("backend", name)),
                    "n_records": 0,
                    "mean_word_count": 0.0,
                    "status": "stub",
                    "error": str(exc),
                }
            )
        except (
            Exception
        ) as exc:  # noqa: BLE001 — experiment must not crash the notebook
            logger.warning("teacher %s failed: %s", name, exc)
            rows.append(
                {
                    "teacher": name,
                    "backend": str(teacher_cfg.get("backend", name)),
                    "n_records": 0,
                    "mean_word_count": 0.0,
                    "status": "error",
                    "error": str(exc),
                }
            )
    return pd.DataFrame(rows)
