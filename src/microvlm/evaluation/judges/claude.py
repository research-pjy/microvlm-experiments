"""Claude judge stub (iteration 0)."""

from __future__ import annotations

from pathlib import Path

from microvlm.evaluation.judges.base_judge import BaseJudge


class ClaudeJudge(BaseJudge):
    """Anthropic Claude judge. Implement using ``api_key`` from ``utils.api_keys``."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        temperature: float = 0.0,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "judge_claude")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature

    def generate(self, prompt: str, image: Path | None = None) -> str:
        """Not implemented in iteration 0."""

        raise NotImplementedError(
            "ClaudeJudge.generate is a stub. Use self.api_key from utils.api_keys "
            "(ANTHROPIC_API_KEY) when implementing the Anthropic call."
        )
