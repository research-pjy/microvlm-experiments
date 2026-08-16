"""Build ShortDesc/LongDesc JSONL from fixture records + a teacher backend."""

from __future__ import annotations

from pathlib import Path

from microvlm.data.coco import FixtureRecord
from microvlm.data.generation.cache import (
    append_record,
    cache_key,
    existing_keys,
    load_jsonl,
)
from microvlm.data.prompts import (
    child_directed_prompt,
    child_speech_prompt,
    controlled_length_prompt,
    longdesc_prompt,
    shortdesc_prompt,
)
from microvlm.data.schemas import GeneratedRecord
from microvlm.data.teachers.base_teacher import BaseTeacher


def _render(prompt_id: str, captions: list[str], n: int | None = None) -> str:
    if prompt_id == shortdesc_prompt.PROMPT_ID:
        return shortdesc_prompt.render(captions)
    if prompt_id == longdesc_prompt.PROMPT_ID:
        return longdesc_prompt.render(captions)
    if prompt_id == child_speech_prompt.PROMPT_ID:
        return child_speech_prompt.render(captions)
    if prompt_id == child_directed_prompt.PROMPT_ID:
        return child_directed_prompt.render(captions)
    if prompt_id == controlled_length_prompt.PROMPT_ID:
        if n is None:
            raise ValueError("controlled_length requires n")
        return controlled_length_prompt.render(captions, n)
    raise ValueError(f"Unknown prompt_id {prompt_id!r}")


def word_count(text: str) -> int:
    """Count whitespace-separated tokens."""

    return len(text.split())


def build_dataset(
    records: list[FixtureRecord],
    teacher: BaseTeacher,
    prompt_id: str,
    output_jsonl: Path,
    length_n: int | None = None,
) -> list[GeneratedRecord]:
    """Generate descriptions, skipping keys already present in ``output_jsonl``.

    The paper pipeline is text-only: the teacher sees captions, not the image.
    """

    effective_prompt_id = prompt_id if length_n is None else f"{prompt_id}_{length_n}"
    seen = existing_keys(output_jsonl)
    written: list[GeneratedRecord] = []
    for rec in records:
        key = cache_key(rec.image_id, effective_prompt_id, teacher.name)
        if key in seen:
            continue
        prompt = _render(prompt_id, rec.captions, n=length_n)
        text = teacher.generate(prompt, image=None)
        generated = GeneratedRecord(
            image_id=rec.image_id,
            captions=rec.captions,
            prompt_id=effective_prompt_id,
            teacher_name=teacher.name,
            generated_text=text,
            word_count=word_count(text),
        )
        append_record(output_jsonl, generated)
        written.append(generated)
        seen.add(key)
    return load_jsonl(output_jsonl)
