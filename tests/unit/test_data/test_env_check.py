"""Env guard unit test."""

from __future__ import annotations

import pytest

from microvlm.utils.env_check import require_microvlm_env


def test_require_microvlm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrong conda env must fail loudly."""

    monkeypatch.setenv("CONDA_DEFAULT_ENV", "base")
    with pytest.raises(RuntimeError, match="microvlm"):
        require_microvlm_env()
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "microvlm")
    require_microvlm_env()
