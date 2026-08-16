"""Shared experiment helpers (not imported by notebooks directly)."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from microvlm.data.coco import FixtureRecord, fixture_is_present, load_coco, load_fixture
from microvlm.data.datasets import (
    CaptionDataset,
    SimpleTokenizer,
    records_from_fixture_captions,
)
from microvlm.data.generation.cache import load_jsonl
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
    """Build a CaptionDataset from human captions (no teacher).

    Despite the name this is not fixture-specific: ``fixtures`` just needs
    to be a list of records shaped like ``FixtureRecord`` (image_id,
    image_path, captions) — the 5-image fixture and real COCO
    (``microvlm.data.coco.load_coco``) both qualify, since
    ``records_from_fixture_captions`` only reads those three fields.
    """

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


def load_coco_records(cfg: DictConfig) -> list[FixtureRecord]:
    """Load real COCO captions for the active ``cfg.data`` entry.

    ``root_path``/``split`` are read from ``cfg.data`` directly. Every
    coco-derived data config (coco/shortdesc/longdesc/controlled_length)
    carries its own copy of these two fields, so a single
    ``data.root_path=... data.split=...`` CLI override works no matter
    which one is active — see configs/data/*.yaml.

    ``training.max_samples`` (set to 5 for ``local_smoke``) is used as an
    optional cap so a local CPU run never tries to pull the full split.

    Raises:
        FileNotFoundError: If ``cfg.data.root_path`` is unset (the
            portable default — set it per machine, never commit a real
            path as the group default).
    """

    root_path = cfg.data.get("root_path")
    if root_path is None:
        raise FileNotFoundError(
            "cfg.data.root_path is not set — pass e.g. "
            "data.root_path=/scratch/<user>/datasets/coco on the DGX, "
            "or select data=fixture for the local 5-image smoke path."
        )
    max_samples = cfg.training.get("max_samples") if "training" in cfg else None
    return load_coco(
        Path(str(root_path)),
        split=str(cfg.data.get("split", "val2017")),
        limit=int(max_samples) if max_samples is not None else None,
    )


def generated_jsonl_path(cfg: DictConfig, prompt_id: str) -> Path:
    """Path a matching ``microvlm-generate`` run would have written.

    ``cli/generate_data.py`` names outputs
    ``data/processed/{prompt_id}_{teacher_name}.jsonl`` — not
    ``cfg.data.output_jsonl`` verbatim — so different teachers' outputs for
    the same prompt never collide (needed for the teacher-comparison
    experiment). Training reads back this same computed path so it always
    matches whatever generation actually produced, regardless of what the
    yaml's ``output_jsonl`` field says.
    """

    teacher_name = str(cfg.teacher.get("name", cfg.teacher.get("backend", "unknown")))
    return processed_path(f"{prompt_id}_{teacher_name}.jsonl")


def dataset_for_cfg(cfg: DictConfig) -> CaptionDataset:
    """Build the training CaptionDataset for whichever ``data`` group is active.

    - ``data.name == "fixture"``: the 5-image fixture, raw human captions —
      unchanged iteration-0 path.
    - ``data.name == "coco"``: real COCO images, raw human captions (first
      of the ~5 per image) — the "raw captions" arm of the
      raw-vs-ShortDesc-vs-LongDesc comparison.
    - ``data.name in {"shortdesc", "longdesc"}``: real COCO images, teacher-
      generated target text loaded from the matching ``microvlm-generate``
      output (see ``generated_jsonl_path``). Run generation first — this
      raises a clear error if it hasn't happened.
    - ``data.name == "controlled_length"``: not handled here yet — it needs
      one length selected out of the 5 candidates. See
      exp03_prompt_vs_length.py / exp04_language_complexity.py.
    """

    name = str(cfg.data.name)
    vocab_size = int(cfg.model.vocab_size)
    image_size = int(cfg.model.image_size)
    max_seq_len = min(64, int(cfg.model.max_seq_len))

    if name == "fixture":
        return fixture_caption_dataset(
            vocab_size=vocab_size, image_size=image_size, max_seq_len=max_seq_len
        )

    if name == "controlled_length":
        # data.lengths holds 5 candidate lengths (20/40/60/80/100) — training
        # needs one specific length selected, and generation writes all five
        # into the same jsonl. That selection scheme belongs to
        # exp03_prompt_vs_length.py / exp04_language_complexity.py, which
        # actually run the length sweep — not this generic single-dataset
        # entrypoint. Deferring rather than guessing a selection field here.
        raise NotImplementedError(
            "controlled_length has 5 candidate lengths and isn't wired into "
            "the generic cli/train.py path yet — see "
            "exp03_prompt_vs_length.py / exp04_language_complexity.py."
        )

    if name not in {"coco", "shortdesc", "longdesc"}:
        raise ValueError(f"Unknown cfg.data.name={name!r}")

    # Only reached for coco/shortdesc/longdesc — validate before paying for
    # a COCO annotations parse that an unsupported name would waste.
    coco_records = load_coco_records(cfg)

    if name == "coco":
        return fixture_caption_dataset(
            fixtures=coco_records,
            vocab_size=vocab_size,
            image_size=image_size,
            max_seq_len=max_seq_len,
        )

    # Only "shortdesc"/"longdesc" remain at this point (validated above).
    prompt_id = str(cfg.data.prompt_id)
    jsonl_path = generated_jsonl_path(cfg, prompt_id)
    generated = load_jsonl(jsonl_path)
    if not generated:
        teacher_name = str(cfg.teacher.get("name", "unknown"))
        raise FileNotFoundError(
            f"{jsonl_path} is empty or missing — run "
            f"`microvlm-generate data={name} data.root_path=... "
            f"teacher={teacher_name}` first."
        )
    # Generation and training may point at different root_path/split;
    # only train on records whose image is actually present here.
    available = {r.image_id for r in coco_records}
    generated = [r for r in generated if r.image_id in available]
    if not generated:
        raise ValueError(
            f"none of the records in {jsonl_path} match images under "
            f"data.root_path={cfg.data.get('root_path')} "
            f"split={cfg.data.get('split')}"
        )
    return fixture_caption_dataset(
        fixtures=coco_records,
        records=generated,
        vocab_size=vocab_size,
        image_size=image_size,
        max_seq_len=max_seq_len,
    )