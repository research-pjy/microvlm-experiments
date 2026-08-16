"""NanoVLM hyperparameters and parameter-allocation helpers."""

from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig, OmegaConf


@dataclass
class NanoVLMConfig:
    """Flat model config matching Table 1 plus encoder architecture name.

    Transformer blocks use ``n_embd = n_head * head_size``. The encoder's CLS
    state is then linearly mapped to ``img_embd_dim`` so the connector can
    project ``img_embd_dim -> n_embd`` as specified in Section 2.2.2.
    """

    name: str = "mini"
    n_blks: int = 1
    n_layer: int = 4
    n_head: int = 8
    head_size: int = 12
    n_embd: int = 96
    img_embd_dim: int = 400
    dropout: float = 0.1
    image_size: int = 224
    patch_size: int = 16
    learning_rate: float = 1e-3
    vocab_size: int = 8192
    max_seq_len: int = 256
    encoder_arch: str = "arch_a"

    def __post_init__(self) -> None:
        expected = self.n_head * self.head_size
        if expected != self.n_embd:
            raise ValueError(
                f"n_head * head_size ({expected}) must equal n_embd ({self.n_embd})"
            )

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> NanoVLMConfig:
        """Build from a composed Hydra config with ``model`` and ``encoder``."""

        model = cfg.model if "model" in cfg else cfg
        encoder_arch = "arch_a"
        if "encoder" in cfg and cfg.encoder is not None:
            encoder_arch = str(cfg.encoder.get("arch", "arch_a"))
        raw = OmegaConf.to_container(model, resolve=True)
        assert isinstance(raw, dict)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in raw.items() if k in known}
        kwargs["encoder_arch"] = encoder_arch
        return cls(**kwargs)


# Table 2 reference shares (encoder, connector, decoder); not an exact constraint.
TABLE2_SHARES: dict[str, tuple[float, float, float]] = {
    "mini": (0.69, 0.14, 0.17),
    "base": (0.78, 0.08, 0.16),
    "large": (0.73, 0.06, 0.21),
}


def count_parameters(module) -> int:
    """Return the number of trainable parameters in ``module``."""

    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def parameter_allocation(encoder, connector, decoder) -> dict[str, float]:
    """Return module parameter counts and shares (for exp08 / validation).

    Args:
        encoder: Visual encoder module.
        connector: Connector module.
        decoder: Decoder module.

    Returns:
        Dict with counts and fractions for encoder, connector, and decoder.
    """

    e = count_parameters(encoder)
    c = count_parameters(connector)
    d = count_parameters(decoder)
    total = max(e + c + d, 1)
    return {
        "encoder": e,
        "connector": c,
        "decoder": d,
        "total": e + c + d,
        "encoder_frac": e / total,
        "connector_frac": c / total,
        "decoder_frac": d / total,
    }
