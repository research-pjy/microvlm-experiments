"""ROUGE-1 overlap used in the paper (Figure 8) as a memorization check."""

from __future__ import annotations

from rouge_score import rouge_scorer


def rouge1_f1(hypothesis: str, reference: str) -> float:
    """Return ROUGE-1 F1 in ``[0, 1]``."""

    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return float(scores["rouge1"].fmeasure)
