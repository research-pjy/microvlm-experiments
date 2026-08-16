"""Child-speech prompt (research notes exp06 — language spoken BY a child)."""

PROMPT_ID = "child_speech"

TEMPLATE = "Write exactly as a typical 3-year-old would speak..."


def render(captions: list[str]) -> str:
    """Render the child-speech prompt with captions as context."""

    joined = "\n".join(f"- {c}" for c in captions)
    return f"{TEMPLATE}\n\nCaptions:\n{joined}"
