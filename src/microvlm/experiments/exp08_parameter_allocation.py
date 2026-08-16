"""Experiment 08 — parameter allocation study.

The paper allocates ~70–80% of parameters to the visual encoder without an
ablation. Keep total parameter count conceptually fixed and compare encoder
vs decoder splits 80:20, 70:30, 60:40, 50:50.

Research question: What is the optimal parameter allocation for tiny VLMs?
"""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from microvlm.models.config import TABLE2_SHARES, NanoVLMConfig, parameter_allocation
from microvlm.models.nanovlm import build_nanovlm


def run_exp08_parameter_allocation(cfg: DictConfig) -> pd.DataFrame:
    """Measure actual module shares for mini/base/large vs Table 2 and listed splits."""

    rows = []
    for size in ("mini", "base", "large"):
        # Use current cfg dims only for the composed size; skip rebuilding all three
        # unless this cfg already is that size.
        if str(cfg.model.get("name", "")) != size:
            continue
        model_cfg = NanoVLMConfig.from_hydra(cfg)
        model = build_nanovlm(model_cfg)
        alloc = parameter_allocation(model.encoder, model.connector, model.decoder)
        ref = TABLE2_SHARES[size]
        rows.append(
            {
                "size": size,
                "total": alloc["total"],
                "encoder_frac": alloc["encoder_frac"],
                "connector_frac": alloc["connector_frac"],
                "decoder_frac": alloc["decoder_frac"],
                "table2_encoder": ref[0],
                "table2_connector": ref[1],
                "table2_decoder": ref[2],
            }
        )
    if not rows:
        model_cfg = NanoVLMConfig.from_hydra(cfg)
        model = build_nanovlm(model_cfg)
        alloc = parameter_allocation(model.encoder, model.connector, model.decoder)
        rows.append({"size": model_cfg.name, **alloc})
    splits = [
        tuple(s) for s in cfg.get("splits", [[80, 20], [70, 30], [60, 40], [50, 50]])
    ]
    plan = pd.DataFrame(
        [
            {"encoder_pct": a, "decoder_pct": b, "status": "planned_reallocation"}
            for a, b in splits
        ]
    )
    measured = pd.DataFrame(rows)
    measured["block"] = "measured"
    plan["block"] = "target_splits"
    # TODO(iteration-1): retune n_blks/n_layer to hit each split at fixed total params
    return pd.concat([measured, plan], ignore_index=True)
