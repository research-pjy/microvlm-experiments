"""Experiment 03 — prompt engineering vs description length.

ShortDesc and LongDesc change both prompt wording and target length. Use one
common prompt with target length in {20, 40, 60, 80, 100} so only length varies.

Research question: Is performance affected by description length or by prompt
engineering?
"""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from microvlm.data.prompts.controlled_length_prompt import LENGTHS, render


def run_exp03_prompt_vs_length(cfg: DictConfig) -> pd.DataFrame:
    """Show the controlled-length prompt grid; generation uses local Ollama when called.

    Full teacher generation over COCO is DGX-scale. Locally this returns the
    prompt table so notebooks can display the experimental design.
    """

    lengths = list(cfg.get("lengths", LENGTHS))
    rows = []
    for n in lengths:
        prompt = render(["example caption 1", "example caption 2"], n=int(n))
        rows.append(
            {
                "target_words": int(n),
                "prompt_preview": prompt.split("Captions:")[0].strip(),
            }
        )
    # TODO(iteration-1): generate datasets per length via build_dataset(..., length_n=n)
    return pd.DataFrame(rows)
