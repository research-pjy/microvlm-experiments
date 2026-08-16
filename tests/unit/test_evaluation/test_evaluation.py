"""Evaluation unit tests (ROUGE, judge parser, human CSV)."""

from __future__ import annotations

from pathlib import Path

from microvlm.evaluation.human_eval import export_human_eval, import_human_eval
from microvlm.evaluation.judges.local_ollama import parse_scores
from microvlm.evaluation.metrics.rouge import rouge1_f1


def test_rouge1_range() -> None:
    """ROUGE-1 F1 is in [0, 1]."""

    s = rouge1_f1("the cat sat on the mat", "the cat sat on the mat")
    assert 0.0 <= s <= 1.0
    assert s == 1.0


def test_parse_scores_json() -> None:
    """JSON object from the judge is preferred."""

    text = '{"grammar": 8, "creativity": 7, "consistency": 9, "meaningfulness": 6, "plot": 5}'
    scores = parse_scores(text)
    assert scores is not None
    assert scores["grammar"] == 8


def test_parse_scores_regex_fallback() -> None:
    """Messy local-model prose still yields five numbers when labeled."""

    text = "grammar: 4 creativity: 5 consistency: 6 meaningfulness: 7 plot: 8"
    scores = parse_scores(text)
    assert scores is not None
    assert scores["plot"] == 8


def test_parse_scores_failure_returns_none() -> None:
    """Garbage text does not crash."""

    assert parse_scores("sorry I cannot") is None


def test_human_eval_roundtrip(tmp_path: Path) -> None:
    """Export blank scores and import filled values."""

    path = tmp_path / "h.csv"
    export_human_eval(
        [{"id": "1", "beginning": "A dog", "completion": "runs fast."}],
        path,
    )
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        rows_in = list(csv.DictReader(handle))
    rows_in[0].update(
        {
            "grammar": "1",
            "creativity": "2",
            "consistency": "3",
            "meaningfulness": "4",
            "plot": "5",
        }
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_in[0].keys()))
        writer.writeheader()
        writer.writerows(rows_in)
    rows = import_human_eval(path)
    assert rows[0]["grammar"] == 1.0
