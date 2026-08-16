"""Unit tests for data loaders, prompts, cache, and teachers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from microvlm.data.coco import load_fixture
from microvlm.data.generation.cache import append_record, existing_keys
from microvlm.data.generation.dataset_builder import build_dataset
from microvlm.data.prompts.shortdesc_prompt import TEMPLATE, render
from microvlm.data.schemas import GeneratedRecord
from microvlm.data.teachers.gpt4o import GPT4oTeacher
from microvlm.data.teachers.local_ollama import LocalOllamaTeacher, ollama_reachable


def test_shortdesc_prompt_is_verbatim() -> None:
    """Figure 2 Prompt 1 must appear unchanged."""

    assert "4-5 year old kid" in TEMPLATE
    assert "20-25 word" in TEMPLATE
    text = render(["a", "b", "c", "d", "e"])
    assert "Captions:" in text


def test_load_fixture_schema(tmp_path: Path) -> None:
    """Loader reads the frozen captions.json schema."""

    images = tmp_path / "images"
    images.mkdir()
    img = images / "000000000139.jpg"
    img.write_bytes(b"not-a-real-jpeg")
    payload = {
        "139": {
            "file_name": "000000000139.jpg",
            "captions": ["c1", "c2", "c3", "c4", "c5"],
        }
    }
    (tmp_path / "captions.json").write_text(json.dumps(payload), encoding="utf-8")
    recs = load_fixture(tmp_path)
    assert recs[0].image_id == "139"
    assert recs[0].captions[0] == "c1"


def test_cache_skips_existing(tmp_path: Path) -> None:
    """Reruns must not rewrite a cached (image_id, prompt_id, teacher) key."""

    path = tmp_path / "out.jsonl"
    rec = GeneratedRecord(
        image_id="1",
        captions=["a"],
        prompt_id="shortdesc",
        teacher_name="local_ollama",
        generated_text="hello there",
        word_count=2,
    )
    append_record(path, rec)
    assert ("1", "shortdesc", "local_ollama") in existing_keys(path)


def test_gpt4o_teacher_is_stub() -> None:
    """Cloud teacher must raise NotImplementedError in iteration 0."""

    with pytest.raises(NotImplementedError):
        GPT4oTeacher(api_key="dummy").generate("hi")


def test_ollama_teacher_mocked() -> None:
    """HTTP call is mocked; no network in unit tests."""

    teacher = LocalOllamaTeacher(model="llama3.2:3b")

    class _Resp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"response": " a small dog plays "}

    with patch(
        "microvlm.data.teachers.local_ollama.requests.post", return_value=_Resp()
    ):
        assert teacher.generate("prompt") == "a small dog plays"


def test_build_dataset_uses_teacher(tmp_path: Path) -> None:
    """dataset_builder writes JSONL via the teacher.generate mock."""

    from microvlm.data.coco import FixtureRecord

    recs = [
        FixtureRecord(
            image_id="1",
            file_name="x.jpg",
            image_path=tmp_path / "x.jpg",
            captions=["one", "two", "three", "four", "five"],
        )
    ]

    class _T:
        name = "mock"

        def generate(self, prompt: str, image=None) -> str:
            assert "Captions:" in prompt
            return "the dog runs in the park today"

    out = tmp_path / "g.jsonl"
    built = build_dataset(recs, _T(), "shortdesc", out)  # type: ignore[arg-type]
    assert built[0].generated_text.startswith("the dog")
    built2 = build_dataset(recs, _T(), "shortdesc", out)  # type: ignore[arg-type]
    assert len(built2) == 1


def test_ollama_reachable_mocked_false() -> None:
    """Unreachable Ollama is a boolean, not an exception."""

    with patch(
        "microvlm.data.teachers.local_ollama.requests.get",
        side_effect=OSError("nope"),
    ):
        assert ollama_reachable() is False
