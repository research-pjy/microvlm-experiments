"""Loader for the externally produced 5-image COCO fixture (fixed schema)
and for real COCO 2017 captions on machines where COCO is actually present
(local dev machine, DGX)."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FixtureRecord:
    """One image plus its raw human COCO captions.

    Despite the name, this shape is shared by the 5-image fixture and by
    real COCO loaded via ``load_coco`` below — downstream code (tokenizer
    fitting, ``CaptionDataset``) does not need to know which source an
    image came from.
    """

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


# ---------------------------------------------------------------------------
# Real COCO 2017 (standard JSON annotation format — this repo never bundles
# or samples COCO itself; see configs/data/coco.yaml for root_path/split).
# ---------------------------------------------------------------------------


def coco_annotations_path(root_path: Path, split: str) -> Path:
    """Path to ``captions_{split}.json`` under the standard COCO 2017 layout."""

    return root_path / "annotations" / f"captions_{split}.json"


def coco_images_dir(root_path: Path, split: str) -> Path:
    """Path to the ``{split}/`` image directory."""

    return root_path / split


def coco_is_present(root_path: Path | None, split: str) -> bool:
    """Return True if ``root_path`` is set and the split's data exists.

    Mirrors ``fixture_is_present`` so callers can check availability the
    same way regardless of data source. ``root_path`` is ``None`` on any
    machine where ``configs/data/coco.yaml``'s ``root_path`` is unset
    (e.g. local dev, per this project's convention of never bundling COCO).
    """

    if root_path is None:
        return False
    ann = coco_annotations_path(root_path, split)
    imgs = coco_images_dir(root_path, split)
    return ann.is_file() and imgs.is_dir() and any(imgs.iterdir())


def load_coco(
    root_path: Path, split: str = "val2017", limit: int | None = None
) -> list[FixtureRecord]:
    """Read real COCO captions for ``split``.

    Expects the standard COCO 2017 download layout::

        root_path/annotations/captions_{split}.json
        root_path/{split}/*.jpg

    Groups the human captions per image (usually 5) and returns them in
    the same ``FixtureRecord`` shape as ``load_fixture``, so a single
    downstream conversion function (``records_from_fixture_captions``)
    works unchanged for both the fixture and full COCO.

    Args:
        root_path: Directory containing ``annotations/`` and ``{split}/``.
        split: COCO split name, e.g. ``val2017`` or ``train2017``.
        limit: If set, keep only the first ``limit`` images (sorted by
            numeric image_id for reproducibility). Useful for a fast smoke
            run before committing to the full split.

    Returns:
        List of records, one per image that has at least one caption.

    Raises:
        FileNotFoundError: If the annotations file is missing.
        ValueError: If a referenced image file is missing.
    """

    ann_path = coco_annotations_path(root_path, split)
    images_dir = coco_images_dir(root_path, split)
    if not ann_path.is_file():
        raise FileNotFoundError(f"Missing COCO annotations: {ann_path}")

    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    file_names: dict[int, str] = {
        img["id"]: img["file_name"] for img in payload["images"]
    }
    captions_by_image: dict[int, list[str]] = defaultdict(list)
    for ann in payload["annotations"]:
        captions_by_image[ann["image_id"]].append(str(ann["caption"]).strip())

    image_ids = sorted(iid for iid in file_names if captions_by_image.get(iid))
    if limit is not None:
        image_ids = image_ids[:limit]

    records: list[FixtureRecord] = []
    for image_id in image_ids:
        file_name = file_names[image_id]
        image_path = images_dir / file_name
        if not image_path.is_file():
            raise ValueError(f"Missing COCO image: {image_path}")
        records.append(
            FixtureRecord(
                image_id=str(image_id),
                file_name=file_name,
                image_path=image_path,
                captions=captions_by_image[image_id],
            )
        )
    return records