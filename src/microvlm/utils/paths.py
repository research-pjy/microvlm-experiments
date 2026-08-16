"""Project path helpers (Ubuntu POSIX paths only)."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root (parent of ``src/``)."""

    return Path(__file__).resolve().parents[3]


def configs_dir() -> Path:
    """Return the Hydra configs directory."""

    return project_root() / "configs"


def data_dir() -> Path:
    """Return the gitignored ``data/`` directory."""

    return project_root() / "data"


def outputs_dir() -> Path:
    """Return the gitignored ``outputs/`` directory."""

    return project_root() / "outputs"


def fixture_dir() -> Path:
    """Return ``tests/fixtures`` (populated externally)."""

    return project_root() / "tests" / "fixtures"
