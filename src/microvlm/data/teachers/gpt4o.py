"""GPT-4o teacher stub (iteration 0). Fill in with the OpenAI API later."""

from __future__ import annotations

from pathlib import Path

from microvlm.data.teachers.base_teacher import BaseTeacher


class GPT4oTeacher(BaseTeacher):
    """OpenAI GPT-4o teacher.

    Iteration 0 does not perform network calls. Pass ``api_key`` from
    ``microvlm.utils.api_keys.load_api_keys()['openai']`` when implementing.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        temperature: float = 0.7,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "gpt4o")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature

    def generate(self, prompt: str, image: Path | None = None) -> str:
        """Not implemented in iteration 0."""

        raise NotImplementedError(
            "GPT4oTeacher.generate is a stub. Implement the OpenAI Chat Completions "
            "call here using self.api_key (loaded in utils.api_keys, never from env "
            "inside this class)."
        )
