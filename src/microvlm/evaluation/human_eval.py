"""Human evaluation CSV export/import (no ML/API dependency)."""

from __future__ import annotations

import csv
from pathlib import Path

SCORE_COLUMNS = (
    "grammar",
    "creativity",
    "consistency",
    "meaningfulness",
    "plot",
)


def export_human_eval(rows: list[dict[str, str]], path: Path) -> None:
    """Write generated completions with blank score columns.

    Args:
        rows: Each dict must include ``id``, ``beginning``, ``completion``.
        path: Destination CSV path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "beginning", "completion", *SCORE_COLUMNS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in fieldnames}
            for col in SCORE_COLUMNS:
                out.setdefault(col, "")
            writer.writerow(out)


def import_human_eval(path: Path) -> list[dict[str, str | float | None]]:
    """Read a filled-in human eval CSV into the judge-output schema."""

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        results: list[dict[str, str | float | None]] = []
        for row in reader:
            item: dict[str, str | float | None] = {
                "id": row.get("id", ""),
                "beginning": row.get("beginning", ""),
                "completion": row.get("completion", ""),
            }
            for col in SCORE_COLUMNS:
                raw = (row.get(col) or "").strip()
                item[col] = float(raw) if raw else None
            results.append(item)
        return results
