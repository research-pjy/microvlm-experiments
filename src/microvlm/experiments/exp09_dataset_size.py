"""Experiment 09 — dataset size scaling.

The paper uses ~28k pairs. Scale 10k / 25k / 50k / 100k / full COCO and study
behaviour. Local runs stay on the 5-image fixture (CPU guard).

Research question: How does NanoVLM quality scale with dataset size?
"""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig


def run_exp09_dataset_size(cfg: DictConfig) -> pd.DataFrame:
    """Return the planned size grid; full COCO training is DGX-only."""

    sizes = list(cfg.get("sizes", [10000, 25000, 50000, 100000, None]))
    rows = []
    for n in sizes:
        rows.append(
            {
                "n_pairs": n if n is not None else "full_coco",
                "local_allowed": False if n is None or int(n) > 5 else True,
                "note": "CPU guard forbids >5 samples; use training=dgx_full",
            }
        )
    # TODO(iteration-1): subsample generated JSONL to each n on the DGX
    return pd.DataFrame(rows)
