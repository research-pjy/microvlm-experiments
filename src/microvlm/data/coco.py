"""Loader for the externally produced 5-image COCO fixture (fixed schema)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FixtureRecord:
    """One fixture image plus its raw human COCO captions."""

    image_id: str
    file_name: str
    image_path: Path
    captions: list[str]


def load_fixture(fixture_dir: Path) -> list[FixtureRecord]:
    """Read ``captions.json`` + ``images/`` using the frozen fixture schema.

    Args:
        fixture_dir: Directory containing ``images/`` and ``captions.json``.

    Returns:
        List of fixture records. Captions are raw human strings (not teacher
        outputs).

    Raises:
        FileNotFoundError: If ``captions.json`` is missing.
        ValueError: If a referenced image file is missing.
    """

    captions_path = fixture_dir / "captions.json"
    images_dir = fixture_dir / "images"
    payload = json.loads(captions_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("captions.json must be a dict keyed by image_id")
    records: list[FixtureRecord] = []
    for image_id, value in payload.items():
        file_name = value["file_name"]
        captions = list(value["captions"])
        image_path = images_dir / file_name
        if not image_path.is_file():
            raise ValueError(f"Missing fixture image: {image_path}")
        records.append(
            FixtureRecord(
                image_id=str(image_id),
                file_name=file_name,
                image_path=image_path,
                captions=captions,
            )
        )
    return records


def fixture_is_present(fixture_dir: Path) -> bool:
    """Return True if captions.json exists and images/ is a non-empty directory."""

    captions = fixture_dir / "captions.json"
    images = fixture_dir / "images"
    if not captions.is_file() or not images.is_dir():
        return False
    return any(images.iterdir())
