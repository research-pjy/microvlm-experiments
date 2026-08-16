"""Ollama REST teacher. Model name comes only from constructor/config."""

from __future__ import annotations

from pathlib import Path

import requests

from microvlm.data.teachers.base_teacher import BaseTeacher


class LocalOllamaTeacher(BaseTeacher):
    """Call a local Ollama server (``/api/generate``).

    Default host is ``http://localhost:11434``. The model tag is a constructor
    argument so the same class serves ``llama3.2:3b`` locally and a larger tag
    on the DGX.
    """

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        temperature: float = 0.7,
        timeout_s: float = 120.0,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "local_ollama")
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout_s = timeout_s

    def generate(self, prompt: str, image: Path | None = None) -> str:
        """POST ``/api/generate`` with ``stream=false``. ``image`` is ignored."""

        del image
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        response = requests.post(url, json=payload, timeout=self.timeout_s)
        response.raise_for_status()
        data = response.json()
        return str(data.get("response", "")).strip()


def ollama_reachable(
    host: str = "http://localhost:11434", timeout_s: float = 2.0
) -> bool:
    """Return True if the Ollama HTTP server answers ``/api/tags``."""

    try:
        response = requests.get(f"{host.rstrip('/')}/api/tags", timeout=timeout_s)
        return response.status_code == 200
    except (requests.RequestException, OSError):
        return False
