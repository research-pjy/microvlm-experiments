"""Shared experiment helpers (not imported by notebooks directly)."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from microvlm.data.coco import FixtureRecord, fixture_is_present, load_fixture
from microvlm.data.datasets import (
    CaptionDataset,
    SimpleTokenizer,
    records_from_fixture_captions,
)
from microvlm.data.schemas import GeneratedRecord
from microvlm.utils.paths import configs_dir, fixture_dir, project_root


def require_fixture() -> list[FixtureRecord]:
    """Load the 5-image fixture or raise a clear error."""

    path = fixture_dir()
    if not fixture_is_present(path):
        raise FileNotFoundError(
            "fixture not found — see tests/fixtures/README.md "
            "(populate externally with sample_coco_fixture.py; not in this repo)"
        )
    return load_fixture(path)


def load_teacher_cfg(name: str) -> DictConfig:
    """Load ``configs/teacher/{name}.yaml``."""

    path = configs_dir() / "teacher" / f"{name}.yaml"
    return OmegaConf.load(path)  # type: ignore[return-value]


def fixture_caption_dataset(
    fixtures: list[FixtureRecord] | None = None,
    vocab_size: int = 8192,
    image_size: int = 224,
    max_seq_len: int = 64,
    records: list[GeneratedRecord] | None = None,
) -> CaptionDataset:
    """Build a CaptionDataset from fixture human captions (no teacher)."""

    fixtures = fixtures if fixtures is not None else require_fixture()
    recs = records if records is not None else records_from_fixture_captions(fixtures)
    tok = SimpleTokenizer(vocab_size=vocab_size)
    tok.fit([r.generated_text for r in recs])
    index = {f.image_id: f for f in fixtures}
    return CaptionDataset(
        index, recs, tok, image_size=image_size, max_seq_len=max_seq_len
    )


def processed_path(name: str) -> Path:
    """JSONL path under ``data/processed``."""

    return project_root() / "data" / "processed" / name
