"""Grading prompt (paper Figure 2, Prompt 3) — verbatim, text-only."""

PROMPT_ID = "grading"

TEMPLATE = (
    "In the following exercise, the student is given a beginning of a text. The student "
    "needs to complete the partially completed text. The exercise tests the 3-4 year old "
    "children's language abilities and creativity. Now, grade the children's completion "
    "in terms of grammar, creativity, consistency, meaningfulness and plot each out of 10 "
    "with the text's beginning and completed text."
)


def render(beginning: str, completion: str) -> str:
    """Ask a judge to score beginning + completion on the five paper axes."""

    return (
        f"{TEMPLATE}\n\n"
        "Return a JSON object with keys grammar, creativity, consistency, "
        "meaningfulness, plot (each an integer 0-10).\n\n"
        f"Beginning:\n{beginning}\n\nCompleted text:\n{completion}"
    )
