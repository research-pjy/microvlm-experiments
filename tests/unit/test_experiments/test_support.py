"""Unit tests for microvlm.experiments._support: dataset_for_cfg and friends.

Builds real DictConfigs via compose_config (same helper cli/train.py and the
integration smoke test use) rather than hand-rolled fakes, so these actually
exercise the Hydra override paths a real invocation would use
(``data=coco data.root_path=...``, etc.).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from microvlm.data.generation.cache import append_record
from microvlm.data.schemas import GeneratedRecord
from microvlm.experiments._support import (
    dataset_for_cfg,
    generated_jsonl_path,
    load_coco_records,
)
from microvlm.utils.hydra_compose import compose_config
from microvlm.utils.paths import project_root


def _write_coco_root(root: Path, image_ids: list[int]) -> None:
    """Minimal COCO-2017-shaped directory: one caption per id, empty jpgs.

    Empty files are fine here — nothing under test decodes image bytes,
    only path existence and record counts.
    """

    ann_dir = root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    images_dir = root / "val2017"
    images_dir.mkdir(parents=True, exist_ok=True)
    images = [{"id": i, "file_name": f"{i:012d}.jpg"} for i in image_ids]
    annotations = [{"image_id": i, "caption": "x"} for i in image_ids]
    (ann_dir / "captions_val2017.json").write_text(
        json.dumps({"images": images, "annotations": annotations}), encoding="utf-8"
    )
    for i in image_ids:
        (images_dir / f"{i:012d}.jpg").write_bytes(b"")


def _generated(image_id: str, prompt_id: str = "shortdesc") -> GeneratedRecord:
    return GeneratedRecord(
        image_id=image_id,
        captions=["a caption"],
        prompt_id=prompt_id,
        teacher_name="local_ollama",
        generated_text="a small dog plays",
        word_count=4,
    )


# --------------------------------------------------------------------------
# load_coco_records
# --------------------------------------------------------------------------


def test_load_coco_records_uncapped(tmp_path: Path) -> None:
    _write_coco_root(tmp_path, [1, 2, 3, 4])
    cfg = compose_config(
        overrides=[
            "model=mini",
            "data=coco",
            f"data.root_path={tmp_path}",
            "training=dgx_full",  # max_samples: null -> no cap
        ]
    )
    records = load_coco_records(cfg)
    assert [r.image_id for r in records] == ["1", "2", "3", "4"]


def test_load_coco_records_capped_by_local_smoke(tmp_path: Path) -> None:
    _write_coco_root(tmp_path, [1, 2, 3, 4, 5, 6, 7])
    cfg = compose_config(
        overrides=[
            "model=mini",
            "data=coco",
            f"data.root_path={tmp_path}",
            "training=local_smoke",  # max_samples: 5
        ]
    )
    records = load_coco_records(cfg)
    assert len(records) == 5


def test_load_coco_records_missing_root_path_raises() -> None:
    cfg = compose_config(overrides=["model=mini", "data=coco", "training=local_smoke"])
    with pytest.raises(FileNotFoundError, match="root_path is not set"):
        load_coco_records(cfg)


# --------------------------------------------------------------------------
# generated_jsonl_path
# --------------------------------------------------------------------------


def test_generated_jsonl_path_uses_teacher_name() -> None:
    cfg = OmegaConf.create({"teacher": {"name": "local_ollama"}})
    path = generated_jsonl_path(cfg, "shortdesc")
    assert path == project_root() / "data" / "processed" / "shortdesc_local_ollama.jsonl"


# --------------------------------------------------------------------------
# dataset_for_cfg
# --------------------------------------------------------------------------


def test_dataset_for_cfg_fixture_uses_real_fixture() -> None:
    """data=fixture needs no mocking — tests/fixtures/ is real and committed."""

    cfg = compose_config(overrides=["model=mini", "data=fixture", "training=local_smoke"])
    dataset = dataset_for_cfg(cfg)
    assert len(dataset) >= 1  # type: ignore[arg-type]


def test_dataset_for_cfg_coco_raw_captions(tmp_path: Path) -> None:
    _write_coco_root(tmp_path, [1, 2, 3])
    cfg = compose_config(
        overrides=[
            "model=mini",
            "data=coco",
            f"data.root_path={tmp_path}",
            "training=dgx_full",
        ]
    )
    dataset = dataset_for_cfg(cfg)
    assert len(dataset) == 3  # type: ignore[arg-type]


def test_dataset_for_cfg_shortdesc_filters_to_available_images(tmp_path: Path) -> None:
    _write_coco_root(tmp_path, [1, 2, 3, 4])
    jsonl = tmp_path / "generated.jsonl"
    for image_id in ("1", "2", "5"):  # "5" is not in this COCO root
        append_record(jsonl, _generated(image_id))

    cfg = compose_config(
        overrides=[
            "model=mini",
            "data=shortdesc",
            f"data.root_path={tmp_path}",
            "training=dgx_full",
        ]
    )
    with patch(
        "microvlm.experiments._support.generated_jsonl_path", return_value=jsonl
    ):
        dataset = dataset_for_cfg(cfg)
    assert len(dataset) == 2  # type: ignore[arg-type]


def test_dataset_for_cfg_shortdesc_missing_jsonl_raises(tmp_path: Path) -> None:
    _write_coco_root(tmp_path, [1, 2])
    cfg = compose_config(
        overrides=[
            "model=mini",
            "data=shortdesc",
            f"data.root_path={tmp_path}",
            "training=dgx_full",
        ]
    )
    missing = tmp_path / "does_not_exist.jsonl"
    with patch(
        "microvlm.experiments._support.generated_jsonl_path", return_value=missing
    ):
        with pytest.raises(FileNotFoundError, match="microvlm-generate"):
            dataset_for_cfg(cfg)


def test_dataset_for_cfg_shortdesc_no_overlap_raises(tmp_path: Path) -> None:
    _write_coco_root(tmp_path, [1, 2])
    jsonl = tmp_path / "generated.jsonl"
    append_record(jsonl, _generated("99"))  # not in this COCO root

    cfg = compose_config(
        overrides=[
            "model=mini",
            "data=shortdesc",
            f"data.root_path={tmp_path}",
            "training=dgx_full",
        ]
    )
    with patch(
        "microvlm.experiments._support.generated_jsonl_path", return_value=jsonl
    ):
        with pytest.raises(ValueError, match="none of the records"):
            dataset_for_cfg(cfg)


def test_dataset_for_cfg_controlled_length_not_implemented() -> None:
    cfg = compose_config(overrides=["model=mini", "data=controlled_length"])
    with pytest.raises(NotImplementedError, match="exp03_prompt_vs_length"):
        dataset_for_cfg(cfg)


def test_dataset_for_cfg_unknown_data_name_raises() -> None:
    """Hydra won't compose an undefined `data` group entry, so this cfg is
    built directly rather than through compose_config."""

    cfg = OmegaConf.create(
        {
            "data": {"name": "bogus"},
            "model": {"vocab_size": 8192, "image_size": 224, "max_seq_len": 256},
        }
    )
    with pytest.raises(ValueError, match="Unknown cfg.data.name"):
        dataset_for_cfg(cfg)