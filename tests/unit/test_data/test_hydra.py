"""Hydra composition smoke test."""

from microvlm.models.config import NanoVLMConfig
from microvlm.utils.hydra_compose import compose_config


def test_compose_default_config() -> None:
    """Default config exposes model mini and encoder arch_a."""

    cfg = compose_config()
    assert cfg.model.name == "mini"
    assert cfg.encoder.arch == "arch_a"
    NanoVLMConfig.from_hydra(cfg)


def test_compose_exp07() -> None:
    """Experiment 07 YAML is a global package with training.local_smoke."""

    cfg = compose_config(config_name="experiment/exp07")
    assert cfg.experiment_id == "exp07"
    assert cfg.training.device == "cpu"
    assert cfg.training.max_samples == 5
