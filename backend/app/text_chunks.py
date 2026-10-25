"""Split long documents on paragraph (then line, then sentence) boundaries."""
import re


def chunk_text(text: str, max_chars: int, overlap: int = 0) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            window = text[start:end]
            for pattern in (r"\n\s*\n", r"\n", r"(?<=[.!?])\s"):
                breaks = [match.end() for match in re.finditer(pattern, window)]
                breaks = [position for position in breaks if position > max_chars * 0.5]
                if breaks:
                    end = start + breaks[-1]
                    break
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks
