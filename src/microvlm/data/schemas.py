"""Pydantic schema for teacher-generated image descriptions."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class GeneratedRecord(BaseModel):
    """One cached teacher output written as JSONL under ``data/processed/``."""

    image_id: str
    captions: list[str]
    prompt_id: str
    teacher_name: str
    generated_text: str
    word_count: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
