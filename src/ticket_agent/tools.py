import re

from langchain_core.tools import tool


ARTICLES: list[tuple[set[str], str]] = [
    (
        {"billing", "duplicate", "double", "twice", "charged", "refund"},
        (
            "[KB-001] Duplicate subscription charge\n"
            "Ask the customer for the dates and transaction IDs of both charges. "
            "A billing specialist must verify whether both payments settled. "
            "Do not promise a refund before verification. "
            "Never request a full card number or CVV."
        ),
    ),
    (
        {"account", "password", "login", "reset", "access"},
        (
            "[KB-002] Password reset\n"
            "Ask the customer to use the Forgot password link on the login page. "
            "If the email does not arrive, ask them to check the spam folder. "
            "If they cannot access their email, refer the case to support "
            "for identity verification. Never request their password."
        ),
    ),
]


@tool
def search_knowledge_base(query: str) -> str:
    """Find support instructions using short English keywords.

    Returns matching articles with source IDs, or a no-results message.
    """
    words = set(re.findall(r"[a-z]+", query.lower()))

    matches = [
        text
        for keywords, text in ARTICLES
        if words & keywords
    ]

    return "\n\n".join(matches) if matches else "No matching articles found."