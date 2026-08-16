"""Experiment 05 — validate the child-language assumption.

Train identical NanoVLMs on (A) child-like language, (B) simple adult language,
(C) original COCO captions, (D) teacher-style explanatory language.

Research question: Does child-like language actually improve learning, or is
simple syntax sufficient?
"""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from microvlm.data.prompts import (
    child_directed_prompt,
    child_speech_prompt,
    shortdesc_prompt,
)


def run_exp05_child_language_assumption(cfg: DictConfig) -> pd.DataFrame:
    """Return the four dataset variants and which prompt/source they use."""

    variants = [
        {
            "variant": "child_like",
            "source": "teacher + shortdesc_prompt",
            "prompt_id": shortdesc_prompt.PROMPT_ID,
        },
        {
            "variant": "simple_adult",
            "source": "teacher (simple adult wording)",
            "prompt_id": "simple_adult",
        },
        {
            "variant": "coco_original",
            "source": "human COCO captions (no teacher)",
            "prompt_id": "human",
        },
        {
            "variant": "teacher_style",
            "source": "child_directed_prompt",
            "prompt_id": child_directed_prompt.PROMPT_ID,
        },
    ]
    # TODO(iteration-1): generate A/B/D via teachers; C uses records_from_fixture_captions
    _ = child_speech_prompt.PROMPT_ID
    return pd.DataFrame(variants)
