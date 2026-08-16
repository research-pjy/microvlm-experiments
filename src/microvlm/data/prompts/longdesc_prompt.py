"""LongDesc teacher prompt (paper Figure 2, Prompt 2) — verbatim except length."""

PROMPT_ID = "longdesc"

TEMPLATE = (
    "You are a 4-5 year old kid who has to write a short description using very simple "
    "english words. Based on the 5 captions given, create a short 60-70 word (5-6 "
    "sentences) description that feels very simple to a kid, like you're sharing a "
    "description about a scene. Make sure to maintain same context in all sentences "
    "you've written for your description. Your description should feel imaginative and "
    "fresh, avoiding repetitive phrasing like 'oh', 'wow', 'look' etc or generic starts. "
    "Keep your tone fresh and let your curiosity shine."
)


def render(captions: list[str]) -> str:
    """Fill the LongDesc prompt with the five human captions."""

    joined = "\n".join(f"- {c}" for c in captions)
    return f"{TEMPLATE}\n\nCaptions:\n{joined}"
