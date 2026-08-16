"""Abstract judge: grade a beginning + completion (optional image)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseJudge(ABC):
    """LLM-as-judge used for grammar/creativity/consistency/meaningfulness/plot."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def generate(self, prompt: str, image: Path | None = None) -> str:
        """Return raw judge text. ``image`` is optional for future multimodal judges."""
