"""Fail loudly unless the active conda environment is named ``microvlm``."""

from __future__ import annotations

import os


def require_microvlm_env() -> None:
    """Raise ``RuntimeError`` if ``CONDA_DEFAULT_ENV`` is not ``microvlm``.

    Makefile targets and the smoke notebook call this before any other work.
    """

    active = os.environ.get("CONDA_DEFAULT_ENV", "")
    if active != "microvlm":
        raise RuntimeError(
            "This project must run inside the existing conda environment "
            f"'microvlm'. Active env is {active!r}. "
            "Run `conda activate microvlm` and retry. "
            "Do not create a new conda env or a venv."
        )


def is_microvlm_env() -> bool:
    """Return True if the active conda env is named microvlm."""

    return os.environ.get("CONDA_DEFAULT_ENV", "") == "microvlm"
