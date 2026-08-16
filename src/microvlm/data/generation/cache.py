"""JSONL cache keyed by (image_id, prompt_id, teacher_name)."""

from __future__ import annotations

from pathlib import Path

from microvlm.data.schemas import GeneratedRecord


def cache_key(image_id: str, prompt_id: str, teacher_name: str) -> tuple[str, str, str]:
    """Return the regeneration-skip tuple."""

    return (str(image_id), str(prompt_id), str(teacher_name))


def load_jsonl(path: Path) -> list[GeneratedRecord]:
    """Load generated records from a JSONL file (empty if missing)."""

    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(GeneratedRecord.model_validate_json(line))
    return records


def existing_keys(path: Path) -> set[tuple[str, str, str]]:
    """Set of cache keys already present in ``path``."""

    return {
        cache_key(r.image_id, r.prompt_id, r.teacher_name) for r in load_jsonl(path)
    }


def append_record(path: Path, record: GeneratedRecord) -> None:
    """Append one JSONL line, creating parent directories as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json() + "\n")
