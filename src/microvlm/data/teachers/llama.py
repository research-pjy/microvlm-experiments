"""Llama teacher preset: same API as local Ollama, different default model tag.

A genuinely separate HTTP API is not required; change ``model`` in
``configs/teacher/llama.yaml`` (or pass it here) to select any Llama tag
served by Ollama.
"""

from microvlm.data.teachers.local_ollama import LocalOllamaTeacher


class LlamaTeacher(LocalOllamaTeacher):
    """Thin Ollama preset for Llama tags."""

    def __init__(
        self,
        model: str = "llama3.2:3b",
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
            name=name or "llama",
        )
