"""Solution-neutral extraction results: schema-shaped data plus grounded, scored leaf values."""
from dataclasses import dataclass
import json
from typing import Literal

from pydantic import Field, JsonValue

from app.grounding import Grounder, Match, sentence_window
from app.models import StrictModel
from app.schema import ExtractionSchema, SchemaField, coerce, field_path, is_empty


Grounding = Literal["exact", "fuzzy", "none"]
PathPart = str | int


class FieldValue(StrictModel):
    path: list[PathPart]
    field: str
    value: JsonValue
    quote: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    grounding: Grounding = "none"
    start: int | None = None
    end: int | None = None


class Usage(StrictModel):
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, other: "Usage") -> "Usage":
        return Usage(calls=self.calls + other.calls, prompt_tokens=self.prompt_tokens + other.prompt_tokens,
                     completion_tokens=self.completion_tokens + other.completion_tokens)


DocumentStatus = Literal["queued", "running", "done", "error", "cancelled"]
RunStatus = Literal["queued", "running", "done", "error", "cancelled"]


class DocumentResult(StrictModel):
    document_id: str
    document_name: str
    document_sha256: str
    status: DocumentStatus = "queued"
    error: str | None = None
    data: dict[str, JsonValue] | None = None
    values: list[FieldValue] = Field(default_factory=list)
    elapsed_seconds: float | None = None
    usage: Usage = Field(default_factory=Usage)
    notes: list[str] = Field(default_factory=list)
    raw: JsonValue = None
    started_at: str | None = None
    finished_at: str | None = None


class RunCreate(StrictModel):
    solution: str = Field(min_length=1, max_length=50)
    label: str = Field(default="", max_length=200)
    schema_id: str = Field(min_length=1, max_length=100)
    schema_version: int | None = Field(default=None, ge=1)
    settings: dict[str, JsonValue] = Field(default_factory=dict)
    document_ids: list[str] = Field(min_length=1, max_length=500)


class ImportedRecord(StrictModel):
    document_id: str
    data: dict[str, JsonValue] | None = None
    error: str | None = None
    raw: JsonValue = None


class RunImport(StrictModel):
    """Results produced outside the workbench (e.g. a packaged tool's UI), collated for comparison."""
    solution: str = Field(min_length=1, max_length=50)
    label: str = Field(default="", max_length=200)
    model: str = Field(default="", max_length=200)
    schema_id: str = Field(min_length=1, max_length=100)
    schema_version: int | None = Field(default=None, ge=1)
    records: list[ImportedRecord] = Field(min_length=1, max_length=500)


class Run(StrictModel):
    id: str
    solution: str
    solution_name: str
    label: str
    model: str
    confidence_method: str
    status: RunStatus = "queued"
    error: str | None = None
    extraction_schema: ExtractionSchema
    settings: dict[str, JsonValue]
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    heartbeat_at: str | None = None  # refreshed while a runner owns the run
    documents: list[DocumentResult]

    def summary(self) -> dict:
        done = [document for document in self.documents if document.status == "done"]
        values = [value for document in done for value in document.values]
        confidences = [value.confidence for value in values if value.confidence is not None]
        usage = Usage()
        for document in self.documents:
            usage = usage.add(document.usage)
        return {
            "id": self.id, "solution": self.solution, "solution_name": self.solution_name,
            "label": self.label, "model": self.model, "status": self.status, "error": self.error,
            "schema_id": self.extraction_schema.id, "schema_version": self.extraction_schema.version,
            "schema_name": self.extraction_schema.name,
            "created_at": self.created_at, "started_at": self.started_at, "finished_at": self.finished_at,
            "documents_total": len(self.documents), "documents_done": len(done),
            "documents_failed": sum(document.status == "error" for document in self.documents),
            "document_ids": [document.document_id for document in self.documents],
            "elapsed_seconds": round(sum(document.elapsed_seconds or 0 for document in self.documents), 1),
            "values": len(values),
            "grounded": sum(value.grounding != "none" for value in values),
            "mean_confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
            "usage": usage.model_dump(), "confidence_method": self.confidence_method,
        }


@dataclass
class Leaf:
    """A scalar value annotated with whatever evidence the solution produced."""
    value: JsonValue
    quote: str | None = None
    confidence: float | None = None
    start: int | None = None
    end: int | None = None


def annotate_plain(data: JsonValue, fields: list[SchemaField]) -> dict:
    """Wrap a plain schema-shaped record (e.g. NuExtract output) in Leaf objects."""
    def wrap(value, field):
        if field.array:
            single = field.model_copy(update={"array": False})
            return [wrap(item, single) for item in (value or [])]
        if field.type == "object":
            source = value if isinstance(value, dict) else {}
            return {child.name: wrap(source.get(child.name), child) for child in field.fields}
        return value if isinstance(value, Leaf) else Leaf(value)
    plain = {field.name: coerce((data or {}).get(field.name) if isinstance(data, dict) else None, field)
             for field in fields}
    return {field.name: wrap(plain[field.name], field) for field in fields}


def plain(value):
    if isinstance(value, Leaf):
        return value.value
    if isinstance(value, list):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    return value


def _fingerprint(value) -> str:
    return json.dumps(plain(value), sort_keys=True, ensure_ascii=False).casefold()


def merge_annotated(fields: list[SchemaField], records: list[dict]) -> dict:
    """Merge chunk-level records: best-confidence scalars, concatenated de-duplicated arrays."""
    def merge(field, values):
        if field.array:
            # Drop items repeated from an earlier chunk (overlap); repeats within one chunk are real.
            merged, earlier = [], set()
            for items in values:
                current = set()
                for item in items or []:
                    key = _fingerprint(item)
                    if key not in earlier and not is_empty(plain(item)):
                        current.add(key)
                        merged.append(item)
                earlier |= current
            return merged
        if field.type == "object":
            return {child.name: merge(child, [(value or {}).get(child.name) for value in values])
                    for child in field.fields}
        candidates = [value for value in values if isinstance(value, Leaf) and value.value is not None]
        if not candidates:
            return Leaf(None)
        return max(candidates, key=lambda leaf: leaf.confidence if leaf.confidence is not None else -1)
    return {field.name: merge(field, [record.get(field.name) for record in records]) for field in fields}


class ValueBuilder:
    """Ground and flatten an annotated record into FieldValues."""

    def __init__(self, text: str, grounder: Grounder | None = None):
        self.text = text
        self.grounder = grounder or Grounder(text)

    def build(self, fields: list[SchemaField], record: dict) -> tuple[dict, list[FieldValue]]:
        values: list[FieldValue] = []
        cursor = None
        for field in fields:
            anchor = self._field(field, record.get(field.name), [field.name], "", None, None, values)
            cursor = anchor if anchor is not None else cursor
        return {field.name: plain(record.get(field.name)) for field in fields}, values

    def _field(self, field, value, path, prefix, near, window, values) -> int | None:
        display = field_path(prefix, field)
        if field.array:
            single = field.model_copy(update={"array": False})
            first = None
            cursor = near
            for index, item in enumerate(value or []):
                anchor = self._item(single, item, path + [index], display, cursor, window, values)
                if anchor is not None:
                    cursor = anchor
                    first = anchor if first is None else first
            return first
        return self._item(field, value, path, display, near, window, values)

    def _item(self, field, value, path, display, near, window, values) -> int | None:
        if field.type != "object":
            return self._leaf(field, value, path, display, near, window, values)
        source = value if isinstance(value, dict) else {}
        scalars = [child for child in field.fields if child.type != "object" and not child.array]
        # Pass 1: children carrying explicit evidence locate the item. The most distinctive one goes
        # first (a span, then a quote occurring once), and the rest are searched near it, so a
        # name that also appears elsewhere is highlighted where this item is.
        evidence = [child for child in scalars if isinstance(source.get(child.name), Leaf)
                    and (source[child.name].quote or source[child.name].start is not None)]
        if not evidence:
            evidence = [child for child in scalars if child.type in ("verbatim", "text")][:1]

        def distinctiveness(child):
            leaf = source.get(child.name)
            if isinstance(leaf, Leaf) and leaf.start is not None:
                return 0
            needle = leaf.quote if isinstance(leaf, Leaf) and leaf.quote else plain(leaf)
            return 1 if isinstance(needle, str) and self.grounder.count(needle) == 1 else 2

        anchor = None
        local = window
        for child in sorted(evidence, key=distinctiveness):
            start = self._leaf(child, source.get(child.name), path + [child.name], f"{display}.{child.name}",
                               anchor if anchor is not None else near, local, values)
            if start is not None and anchor is None:
                anchor = start
                local = sentence_window(self.text, anchor, anchor)
        for child in field.fields:
            if child in evidence:
                continue
            self._field(child, source.get(child.name), path + [child.name], display,
                        anchor if anchor is not None else near, local, values)
        return anchor

    def _leaf(self, field, leaf, path, display, near, window, values) -> int | None:
        if not isinstance(leaf, Leaf):
            leaf = Leaf(leaf)
        if leaf.value is None or is_empty(leaf.value):
            return None
        match: Match | None = None
        if leaf.start is not None and leaf.end is not None and 0 <= leaf.start < leaf.end <= len(self.text):
            exact = leaf.quote is None or self.text[leaf.start:leaf.end] == leaf.quote
            match = Match(leaf.start, leaf.end, "exact" if exact else "fuzzy", 100.0 if exact else 90.0)
        if match is None and leaf.quote:
            match = self.grounder.find(leaf.quote, near, window)
        if match is None:
            match = self.grounder.find_value(leaf.value, field, near, window)
        values.append(FieldValue(
            path=path, field=display, value=leaf.value, quote=leaf.quote,
            confidence=None if leaf.confidence is None else min(1.0, max(0.0, float(leaf.confidence))),
            grounding=match.kind if match else "none",
            start=match.start if match else None, end=match.end if match else None,
        ))
        return match.start if match else None


def finalise(text: str, extraction_schema: ExtractionSchema, record: dict) -> tuple[dict, list[FieldValue]]:
    return ValueBuilder(text).build(extraction_schema.fields, record)
