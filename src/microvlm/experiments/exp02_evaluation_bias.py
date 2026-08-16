"""Experiment 02 — evaluation bias.

GPT-4o is used both as teacher and as evaluator in the paper. Keep the dataset
fixed and evaluate with GPT-4o, Claude, Gemini, and human scores; compare
scores, rankings, and inter-evaluator agreement.

Research question: Does GPT-4o favour outputs similar to its own writing style?
"""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from microvlm.evaluation.human_eval import SCORE_COLUMNS, export_human_eval
from microvlm.evaluation.judges.local_ollama import LocalOllamaJudge, parse_scores
from microvlm.utils.paths import project_root


def run_exp02_evaluation_bias(cfg: DictConfig) -> pd.DataFrame:
    """Orchestrate multi-judge scoring; cloud judges are stubs.

    Local Ollama is wired. Human eval is CSV export. GPT-4o/Claude/Gemini
    rows are marked stub until those backends are implemented.
    """

    judges = list(cfg.get("judge_names", ["judge_local_ollama", "human"]))
    rows = []
    for name in judges:
        if name == "human":
            path = project_root() / "outputs" / "results" / "exp02_human.csv"
            export_human_eval([], path)
            rows.append({"judge": name, "status": "csv_exported", "path": str(path)})
            continue
        if name == "judge_local_ollama":
            _ = LocalOllamaJudge
            _ = parse_scores
            rows.append(
                {
                    "judge": name,
                    "status": "ready",
                    "note": "call LocalOllamaJudge.score(beginning, completion)",
                    **{k: None for k in SCORE_COLUMNS},
                }
            )
            continue
        # TODO(iteration-1): wire GPT4oJudge / ClaudeJudge / GeminiJudge
        rows.append(
            {"judge": name, "status": "stub", "note": "NotImplementedError backend"}
        )
    return pd.DataFrame(rows)
