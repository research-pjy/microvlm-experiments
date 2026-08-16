"""Hydra compose helper used by notebooks and tests."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from microvlm.utils.paths import configs_dir


def compose_config(
    config_name: str = "config",
    overrides: list[str] | None = None,
) -> DictConfig:
    """Compose a Hydra config from ``configs/`` without a global Hydra job.

    Args:
        config_name: YAML stem under ``configs/`` (usually ``config``).
        overrides: Optional Hydra override strings.

    Returns:
        Composed ``DictConfig``.
    """

    from hydra import compose, initialize_config_dir

    cfg_dir = str(configs_dir())
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=list(overrides or []))
    return cfg


def to_container(cfg: DictConfig) -> dict:
    """Convert a DictConfig to a plain dict."""

    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
