"""Qwen teacher preset: same API as local Ollama, different default model tag.

Serve Qwen via Ollama and set ``model`` in ``configs/teacher/qwen.yaml``.
"""

from microvlm.data.teachers.local_ollama import LocalOllamaTeacher


class QwenTeacher(LocalOllamaTeacher):
    """Thin Ollama preset for Qwen tags."""

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        host: str = "http://localhost:11434",
        temperature: float = 0.7,
        timeout_s: float = 120.0,
        name: str | None = None,
    ) -> None:
        super().__init__(
            model=model,
            host=host,
            temperature=temperature,
            timeout_s=timeout_s,
            name=name or "qwen",
        )
