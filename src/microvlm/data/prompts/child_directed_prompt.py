"""Child-directed prompt (research notes exp06 — language spoken TO a child)."""

PROMPT_ID = "child_directed"

TEMPLATE = "Write as a preschool teacher describing the image to a 3-year-old..."


def render(captions: list[str]) -> str:
    """Render the child-directed prompt with captions as context."""

    joined = "\n".join(f"- {c}" for c in captions)
    return f"{TEMPLATE}\n\nCaptions:\n{joined}"
