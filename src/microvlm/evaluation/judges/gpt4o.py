"""GPT-4o judge stub (iteration 0)."""

from __future__ import annotations

from pathlib import Path

from microvlm.evaluation.judges.base_judge import BaseJudge


class GPT4oJudge(BaseJudge):
    """OpenAI GPT-4o judge. Implement using ``api_key`` from ``utils.api_keys``."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        temperature: float = 0.0,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "judge_gpt4o")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature

    def generate(self, prompt: str, image: Path | None = None) -> str:
        """Not implemented in iteration 0."""

        raise NotImplementedError(
            "GPT4oJudge.generate is a stub. Use self.api_key from utils.api_keys "
            "(OPENAI_API_KEY) when implementing the OpenAI call."
        )
