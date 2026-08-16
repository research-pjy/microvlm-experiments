"""Controlled-length prompt (research notes exp03/exp04)."""

PROMPT_ID = "controlled_length"

TEMPLATE = (
    "Generate an image description suitable for a 3-4-year-old child. Target length: "
    "{n} words."
)

LENGTHS = [20, 40, 60, 80, 100]


def render(captions: list[str], n: int) -> str:
    """Render the single parameterized length template plus captions."""

    joined = "\n".join(f"- {c}" for c in captions)
    return f"{TEMPLATE.format(n=n)}\n\nCaptions:\n{joined}"
