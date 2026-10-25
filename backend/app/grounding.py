"""Locate extracted values in canonical document text so every solution gets hit highlighting.

Offsets are Python string indices (Unicode code points), matching the viewer's Array.from indexing.
"""
import bisect
from dataclasses import dataclass
from datetime import date
from typing import Literal

from pydantic import JsonValue
from rapidfuzz import fuzz

from app.schema import MONTHS, SchemaField, parse_date


TRANSLATE = {"‘": "'", "’": "'", "‛": "'", "′": "'", "“": '"', "”": '"',
             "–": "-", "—": "-", "−": "-", " ": " ", "­": ""}
MAX_OCCURRENCES = 500


@dataclass(frozen=True)
class Match:
    start: int
    end: int
    kind: Literal["exact", "fuzzy"]
    score: float


def normalise(text: str) -> tuple[str, list[int], list[int]]:
    """Lower-case, unify quotes/dashes and collapse whitespace, keeping a map to source offsets."""
    characters: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for offset, character in enumerate(text):
        character = TRANSLATE.get(character, character)
        if not character:
            continue
        if character.isspace():
            if characters and characters[-1] == " ":
                ends[-1] = offset + 1
                continue
            character = " "
        lowered = character.lower()
        characters.append(lowered if len(lowered) == 1 else character)
        starts.append(offset)
        ends.append(offset + 1)
    return "".join(characters), starts, ends


def _normalised_needle(needle: str) -> str:
    return normalise(needle)[0].strip()


def _is_word(character: str) -> bool:
    return character.isalnum() or character == "_"


class Grounder:
    def __init__(self, text: str):
        self.text = text
        self._norm, self._starts, self._ends = normalise(text)

    def _occurrences(self, haystack: str, needle: str) -> list[int]:
        found = []
        boundary = len(needle) < 5
        index = haystack.find(needle)
        while index >= 0 and len(found) < MAX_OCCURRENCES:
            before = haystack[index - 1] if index else " "
            after = haystack[index + len(needle)] if index + len(needle) < len(haystack) else " "
            if not boundary or not ((_is_word(needle[0]) and _is_word(before))
                                    or (_is_word(needle[-1]) and _is_word(after))):
                found.append(index)
            index = haystack.find(needle, index + 1)
        return found

    @staticmethod
    def _choose(candidates: list[tuple[int, int]], near: int | None) -> tuple[int, int] | None:
        if not candidates:
            return None
        if near is None:
            return candidates[0]
        return min(candidates, key=lambda item: (abs(item[0] - near), item[0]))

    def _norm_bounds(self, window: tuple[int, int] | None) -> tuple[int, int]:
        if window is None:
            return 0, len(self._norm)
        return (bisect.bisect_left(self._starts, max(0, window[0])),
                bisect.bisect_left(self._starts, min(len(self.text), window[1])))

    def _fuzzy(self, target: str, bounds: tuple[int, int], min_score: float) -> Match | None:
        haystack = self._norm[bounds[0]:bounds[1]]
        if len(haystack) < len(target) * 0.6:
            return None
        alignment = fuzz.partial_ratio_alignment(target, haystack, score_cutoff=min_score)
        if alignment is None:
            return None
        start, end = bounds[0] + alignment.dest_start, bounds[0] + alignment.dest_end
        while start < end and self._norm[start] == " ":
            start += 1
        while end > start and self._norm[end - 1] == " ":
            end -= 1
        if end <= start:
            return None
        return Match(self._starts[start], self._ends[end - 1], "fuzzy", round(alignment.score, 1))

    def find(self, needle: str, near: int | None = None, window: tuple[int, int] | None = None,
             fuzzy: bool = True, min_score: float = 87.0) -> Match | None:
        """Find needle: exact, then normalised (case/space/quotes), then fuzzy; inside window first."""
        if not isinstance(needle, str) or not needle.strip():
            return None
        stripped = needle.strip()
        target = _normalised_needle(stripped)
        exact = [(index, index + len(stripped)) for index in self._occurrences(self.text, stripped)]
        loose = [(self._starts[index], self._ends[index + len(target) - 1])
                 for index in self._occurrences(self._norm, target)] if target else []
        scopes = ([window] if window else []) + [None]
        for scope in scopes:
            def inside(items):
                return [item for item in items if scope is None or (item[0] >= scope[0] and item[1] <= scope[1])]
            chosen = self._choose(inside(exact), near)
            if chosen:
                return Match(chosen[0], chosen[1], "exact", 100.0)
            chosen = self._choose(inside(loose), near)
            if chosen:
                kind = "exact" if self.text[chosen[0]:chosen[1]] == stripped else "fuzzy"
                return Match(chosen[0], chosen[1], kind, 99.0)
            if fuzzy and len(target) >= 6:
                match = self._fuzzy(target[:1_000], self._norm_bounds(scope), min_score)
                if match:
                    return match
        return None

    def find_value(self, value: JsonValue, field: SchemaField, near: int | None = None,
                   window: tuple[int, int] | None = None) -> Match | None:
        """Ground a normalised value (no quote available) using type-aware renderings."""
        if value is None or isinstance(value, (dict, list, bool)):
            return None
        if field.type in ("text", "verbatim"):
            return self.find(str(value), near, window)
        if field.type == "date":
            parsed = parse_date(str(value))
            renderings = [str(value)] + (date_renderings(parsed) if parsed else [])
            return self._first(renderings, near, window)
        if field.type in ("number", "integer"):
            renderings = [str(value)]
            if isinstance(value, (int, float)):
                renderings += [f"{value:,}", f"{value:,.2f}"] if not float(value).is_integer() else [f"{int(value):,}"]
            # Bare numbers are ambiguous; only ground them near their item.
            return self._first(renderings, near, window, fuzzy=False) if window or len(str(value)) > 3 else None
        if field.type == "enum":
            return self._first([str(value)], near, window, fuzzy=False) if window else None
        return None

    def _first(self, renderings: list[str], near: int | None, window: tuple[int, int] | None,
               fuzzy: bool = False) -> Match | None:
        seen = set()
        for rendering in renderings:
            if rendering in seen:
                continue
            seen.add(rendering)
            match = self.find(rendering, near, window, fuzzy=fuzzy)
            if match:
                return match
        return None


def date_renderings(value: date) -> list[str]:
    month = MONTHS[value.month - 1].capitalize()
    suffix = "th" if 11 <= value.day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(value.day % 10, "th")
    return [
        f"{value.day} {month} {value.year}", f"{value.day}{suffix} {month} {value.year}",
        f"{month} {value.day}, {value.year}", f"{month} {value.day} {value.year}",
        f"{value.day} {month[:3]} {value.year}", f"{value.day:02d}/{value.month:02d}/{value.year}",
        f"{value.day}/{value.month}/{value.year}", f"{value.day:02d}.{value.month:02d}.{value.year}",
        f"{value.day:02d}-{value.month:02d}-{value.year}", value.isoformat(),
        f"{value.day:02d}/{value.month:02d}/{value.year % 100:02d}",
        f"{value.day} {month}",
    ]


def sentence_window(text: str, start: int, end: int, radius: int = 1_500) -> tuple[int, int]:
    return max(0, start - radius), min(len(text), end + radius)


def word_boundary_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Expand a fuzzy span to whole words for nicer highlights."""
    while start > 0 and _is_word(text[start - 1]) and _is_word(text[start]):
        start -= 1
    while end < len(text) and _is_word(text[end]) and _is_word(text[end - 1]):
        end += 1
    return start, end


