"""Experiment 06 — child speech vs child-directed language.

The paper does not distinguish language spoken BY children vs language spoken
TO children. Generate Dataset A (typical 3-year-old speech) and Dataset B
(preschool teacher talking to a 3-year-old) and compare.

Research question: Do these two distributions train tiny VLMs differently?
"""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from microvlm.data.prompts.child_directed_prompt import TEMPLATE as DIRECTED
from microvlm.data.prompts.child_speech_prompt import TEMPLATE as SPEECH


def run_exp06_child_speech_vs_directed(cfg: DictConfig) -> pd.DataFrame:
    """Return the two prompts; generation is local-Ollama-ready via dataset_builder."""

    del cfg
    return pd.DataFrame(
        [
            {
                "dataset": "A_child_speech",
                "prompt_id": "child_speech",
                "template": SPEECH,
            },
            {
                "dataset": "B_child_directed",
                "prompt_id": "child_directed",
                "template": DIRECTED,
            },
        ]
    )
    # TODO(iteration-1): build_dataset(..., prompt_id=child_speech/child_directed) then train
