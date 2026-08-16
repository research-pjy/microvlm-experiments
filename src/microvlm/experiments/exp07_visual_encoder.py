"""Experiment 07 — visual encoder ambiguity.

Section 2.2.1, Figure 4, and Figure 5 disagree. Train arch_a (patch-then-conv)
and arch_b (conv-then-patch) on the same 5-image fixture and compare loss
curves. No external LLM is required.

Research question: Which implementation matches the reported results?
"""

from __future__ import annotations

import copy

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from microvlm.experiments._support import fixture_caption_dataset, require_fixture
from microvlm.models.config import NanoVLMConfig, count_parameters, parameter_allocation
from microvlm.models.nanovlm import build_nanovlm
from microvlm.training.trainer import train_model
from microvlm.utils.seeding import seed_everything


def run_exp07_visual_encoder(cfg: DictConfig) -> pd.DataFrame:
    """Train mini NanoVLM with arch_a and arch_b on the fixture captions."""

    seed_everything(int(cfg.get("seed", 42)))
    fixtures = require_fixture()
    dataset = fixture_caption_dataset(
        fixtures,
        vocab_size=int(cfg.model.vocab_size),
        image_size=int(cfg.model.image_size),
    )
    rows: list[dict] = []
    for arch in list(cfg.get("encoder_arches", ["arch_a", "arch_b"])):
        cfg_i = copy.deepcopy(cfg)
        OmegaConf.set_struct(cfg_i, False)
        cfg_i.encoder = {"arch": arch, "name": arch}
        model_cfg = NanoVLMConfig.from_hydra(cfg_i)
        model = build_nanovlm(model_cfg)
        alloc = parameter_allocation(model.encoder, model.connector, model.decoder)
        result = train_model(model, dataset, cfg_i)
        for step, loss in enumerate(result.losses):
            rows.append(
                {
                    "arch": arch,
                    "step": step,
                    "loss": loss,
                    "n_params": count_parameters(model),
                    "encoder_frac": alloc["encoder_frac"],
                    "final_epoch_mean": (
                        result.epoch_means[-1] if result.epoch_means else None
                    ),
                }
            )
    return pd.DataFrame(rows)
