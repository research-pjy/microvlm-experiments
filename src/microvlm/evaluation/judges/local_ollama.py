"""Ollama judge: generate then parse five 0–10 scores defensively."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from microvlm.data.prompts.grading_prompt import render as render_grading
from microvlm.data.teachers.local_ollama import LocalOllamaTeacher
from microvlm.evaluation.judges.base_judge import BaseJudge

logger = logging.getLogger(__name__)

SCORE_KEYS = ("grammar", "creativity", "consistency", "meaningfulness", "plot")


class LocalOllamaJudge(BaseJudge):
    """Local Ollama judge. Model name is constructor/config only."""

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        timeout_s: float = 120.0,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "judge_local_ollama")
        self._teacher = LocalOllamaTeacher(
            model=model,
            host=host,
            temperature=temperature,
            timeout_s=timeout_s,
            name=self.name,
        )

    def generate(self, prompt: str, image: Path | None = None) -> str:
        """Delegate to the Ollama REST teacher."""

        return self._teacher.generate(prompt, image=image)

    def score(self, beginning: str, completion: str) -> dict[str, float | None]:
        """Grade beginning+completion; return None scores on total parse failure."""

        raw = self.generate(render_grading(beginning, completion))
        parsed = parse_scores(raw)
        if parsed is None:
            logger.warning(
                "Judge parse failed; returning None scores. raw=%r", raw[:500]
            )
            return {k: None for k in SCORE_KEYS}
        return parsed


def parse_scores(text: str) -> dict[str, float] | None:
    """Parse five scores from JSON or a regex fallback.

    Returns:
        Mapping of SCORE_KEYS to floats in ``[0, 10]``, or None if unusable.
    """

    json_blob = _extract_json_object(text)
    if json_blob is not None:
        try:
            data = json.loads(json_blob)
            return _coerce_scores(data)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    found: dict[str, float] = {}
    for key in SCORE_KEYS:
        match = re.search(rf"{key}\s*[:=]\s*(\d+(?:\.\d+)?)", text, flags=re.I)
        if match:
            found[key] = float(match.group(1))
    if len(found) != len(SCORE_KEYS):
        nums = [float(x) for x in re.findall(r"\b(\d+(?:\.\d+)?)\b", text)]
        nums = [n for n in nums if 0 <= n <= 10]
        if len(nums) >= 5:
            found = dict(zip(SCORE_KEYS, nums[:5], strict=False))
    if len(found) != len(SCORE_KEYS):
        return None
    return {k: max(0.0, min(10.0, found[k])) for k in SCORE_KEYS}


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _coerce_scores(data: Any) -> dict[str, float]:
    if not isinstance(data, dict):
        raise ValueError("not a dict")
    out: dict[str, float] = {}
    for key in SCORE_KEYS:
        if key not in data:
            raise ValueError(f"missing {key}")
        out[key] = float(data[key])
    return out
