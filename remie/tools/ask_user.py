"""Interactive user-question tool."""

from typing import Any

def ask_user_tool(
    question: str, options: list[str] | None = None
) -> dict[str, Any]:
    """
    Asks the user a question and waits for their answer. Use this when you need
    a decision or clarification from the user instead of guessing.
    :param question: The question to ask the user.
    :param options: Optional list of predefined choices to offer.
    :return: The user's answer.
    """
    return {"question": question, "options": options or []}


