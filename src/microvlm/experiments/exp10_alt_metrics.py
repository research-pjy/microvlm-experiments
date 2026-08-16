"""Experiment 10 — alternative evaluation metrics.

Complement GPT-4o grading with BLEU, METEOR, CIDEr, SPICE, CLIPScore, and
human preference. ROUGE-1 is implemented in iteration 0.

Research question: Do automatic metrics change model rankings vs GPT-4o grades?
"""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from microvlm.evaluation.metrics.rouge import rouge1_f1


def run_exp10_alt_metrics(cfg: DictConfig) -> pd.DataFrame:
    """Score a toy pair with ROUGE-1; other metrics remain stubs."""

    del cfg
    demo_h = "a small cat sits on a red mat"
    demo_r = "a cat is sitting on a mat"
    rouge = rouge1_f1(demo_h, demo_r)
    rows = [
        {"metric": "rouge1", "status": "ok", "demo_score": rouge},
        {"metric": "bleu", "status": "stub", "demo_score": None},
        {"metric": "meteor", "status": "stub", "demo_score": None},
        {"metric": "cider", "status": "stub", "demo_score": None},
        {"metric": "spice", "status": "stub", "demo_score": None},
        {"metric": "clipscore", "status": "stub", "demo_score": None},
        {"metric": "human", "status": "csv_io_ready", "demo_score": None},
    ]
    # TODO(iteration-1): implement stub metrics and run on model completions
    return pd.DataFrame(rows)
