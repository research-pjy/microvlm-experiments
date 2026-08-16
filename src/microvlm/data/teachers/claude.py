"""Claude teacher stub (iteration 0). Fill in with the Anthropic API later."""

from __future__ import annotations

from pathlib import Path

from microvlm.data.teachers.base_teacher import BaseTeacher


class ClaudeTeacher(BaseTeacher):
    """Anthropic Claude teacher.

    Pass ``api_key`` from ``load_api_keys()['anthropic']`` when implementing.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        temperature: float = 0.7,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "claude")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature

    def generate(self, prompt: str, image: Path | None = None) -> str:
        """Not implemented in iteration 0."""

        raise NotImplementedError(
            "ClaudeTeacher.generate is a stub. Implement the Anthropic Messages "
            "API here using self.api_key from utils.api_keys."
        )
