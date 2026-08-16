"""Gemini judge stub (iteration 0)."""

from __future__ import annotations

from pathlib import Path

from microvlm.evaluation.judges.base_judge import BaseJudge


class GeminiJudge(BaseJudge):
    """Google Gemini judge. Implement using ``api_key`` from ``utils.api_keys``."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        temperature: float = 0.0,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "judge_gemini")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature

    def generate(self, prompt: str, image: Path | None = None) -> str:
        """Not implemented in iteration 0."""

        raise NotImplementedError(
            "GeminiJudge.generate is a stub. Use self.api_key from utils.api_keys "
            "(GOOGLE_API_KEY) when implementing the Gemini call."
        )
