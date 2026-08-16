"""Factory: Hydra teacher config → teacher instance (keys passed in explicitly)."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from microvlm.data.teachers.base_teacher import BaseTeacher
from microvlm.data.teachers.claude import ClaudeTeacher
from microvlm.data.teachers.gemini import GeminiTeacher
from microvlm.data.teachers.gpt4o import GPT4oTeacher
from microvlm.data.teachers.llama import LlamaTeacher
from microvlm.data.teachers.local_ollama import LocalOllamaTeacher
from microvlm.data.teachers.qwen import QwenTeacher
from microvlm.utils.api_keys import load_api_keys


def build_teacher(
    cfg: DictConfig, api_keys: dict[str, str | None] | None = None
) -> BaseTeacher:
    """Instantiate a teacher from a Hydra teacher node."""

    keys = api_keys if api_keys is not None else load_api_keys()
    raw = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(raw, dict)
    backend = str(raw.get("backend", raw.get("name", "local_ollama")))
    name = str(raw.get("name", backend))
    if backend in {"local_ollama", "llama", "qwen"}:
        cls = {
            "local_ollama": LocalOllamaTeacher,
            "llama": LlamaTeacher,
            "qwen": QwenTeacher,
        }[backend]
        return cls(
            model=str(raw["model"]),
            host=str(raw.get("host", "http://localhost:11434")),
            temperature=float(raw.get("temperature", 0.7)),
            timeout_s=float(raw.get("timeout_s", 120)),
            name=name,
        )
    if backend == "gpt4o":
        return GPT4oTeacher(
            model=str(raw.get("model", "gpt-4o")),
            api_key=keys.get("openai"),
            temperature=float(raw.get("temperature", 0.7)),
            name=name,
        )
    if backend == "claude":
        return ClaudeTeacher(
            model=str(raw.get("model", "claude-sonnet-4-20250514")),
            api_key=keys.get("anthropic"),
            temperature=float(raw.get("temperature", 0.7)),
            name=name,
        )
    if backend == "gemini":
        return GeminiTeacher(
            model=str(raw.get("model", "gemini-2.0-flash")),
            api_key=keys.get("google"),
            temperature=float(raw.get("temperature", 0.7)),
            name=name,
        )
    raise ValueError(f"Unknown teacher backend {backend!r}")
