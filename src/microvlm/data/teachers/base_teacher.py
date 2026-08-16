"""Abstract teacher: text generation with optional image for future multimodal use."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseTeacher(ABC):
    """Teacher LLM used to turn COCO captions into ShortDesc/LongDesc text."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def generate(self, prompt: str, image: Path | None = None) -> str:
        """Return generated text. ``image`` is unused in the paper's text-only pipeline."""
