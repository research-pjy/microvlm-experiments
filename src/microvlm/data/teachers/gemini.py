"""Gemini teacher stub (iteration 0). Fill in with the Google Generative AI API later."""

from __future__ import annotations

from pathlib import Path

from microvlm.data.teachers.base_teacher import BaseTeacher


class GeminiTeacher(BaseTeacher):
    """Google Gemini teacher.

    Pass ``api_key`` from ``load_api_keys()['google']`` when implementing.
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        temperature: float = 0.7,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "gemini")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature

    def generate(self, prompt: str, image: Path | None = None) -> str:
        """Not implemented in iteration 0."""

        raise NotImplementedError(
            "GeminiTeacher.generate is a stub. Implement the Google Generative AI "
            "call here using self.api_key from utils.api_keys."
        )
