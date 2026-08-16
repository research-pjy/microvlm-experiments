"""Experiment 04 — language complexity scaling.

Create datasets at 20/40/60/80/100 words and study convergence, coherence,
hallucination, grammatical correctness, and context retention.

Research question: At what language complexity do tiny VLMs begin to fail?
"""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from microvlm.data.prompts.controlled_length_prompt import LENGTHS
from microvlm.models.config import NanoVLMConfig, count_parameters
from microvlm.models.nanovlm import build_nanovlm


def run_exp04_language_complexity(cfg: DictConfig) -> pd.DataFrame:
    """Report planned length grid plus mini-model size (no full training locally)."""

    model_cfg = NanoVLMConfig.from_hydra(cfg)
    model = build_nanovlm(model_cfg)
    n_params = count_parameters(model)
    lengths = list(cfg.get("lengths", LENGTHS))
    rows = [
        {
            "target_words": int(n),
            "model": model_cfg.name,
            "n_params": n_params,
            "status": "planned",
        }
        for n in lengths
    ]
    # TODO(iteration-1): train one model per length on generated JSONL and log loss
    return pd.DataFrame(rows)
