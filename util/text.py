"""Short string helpers shared by news and trading."""


def truncate_text(text: str, limit: int = 3000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n… [truncated]"
