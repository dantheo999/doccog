"""Shared extraction schema: the agreed property set every solution extracts against."""
from collections.abc import Iterator
from datetime import date, datetime
from decimal import Decimal
import re
from typing import Annotated, Literal

from pydantic import Field, JsonValue, StringConstraints, model_validator

from app.models import StrictModel


FieldType = Literal["text", "verbatim", "date", "number", "integer", "boolean", "enum", "object"]
FieldName = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
MAX_DEPTH = 3
TYPE_LABELS = {
    "text": "text (may be lightly normalised or summarised)",
    "verbatim": "verbatim text copied exactly from the document",
    "date": "date, normalised to ISO 8601 (YYYY-MM-DD)",
    "number": "number",
    "integer": "whole number",
    "boolean": "true/false",
    "enum": "one of the listed options",
    "object": "object",
}


class SchemaField(StrictModel):
    name: FieldName
    type: FieldType = "text"
    description: str = Field(default="", max_length=1_000)
    array: bool = False
    options: list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]] = Field(
        default_factory=list, max_length=50)
    fields: list["SchemaField"] = Field(default_factory=list, max_length=40)

    @model_validator(mode="after")
    def check_shape(self):
        if self.type == "object":
            if not self.fields:
                raise ValueError(f"Object field '{self.name}' needs at least one child field.")
            names = [field.name for field in self.fields]
            if len(set(names)) != len(names):
                raise ValueError(f"Object field '{self.name}' has duplicate child names.")
        elif self.fields:
            raise ValueError(f"Only object fields can have child fields ('{self.name}').")
        if self.type == "enum":
            if not self.options:
                raise ValueError(f"Enum field '{self.name}' needs options.")
            if len({option.casefold() for option in self.options}) != len(self.options):
                raise ValueError(f"Enum field '{self.name}' has duplicate options.")
        elif self.options:
            raise ValueError(f"Only enum fields can have options ('{self.name}').")
        return self


class SchemaDraft(StrictModel):
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
    description: str = Field(default="", max_length=4_000)
    fields: list[SchemaField] = Field(min_length=1, max_length=60)

    @model_validator(mode="after")
    def check_fields(self):
        names = [field.name for field in self.fields]
        if len(set(names)) != len(names):
            raise ValueError("Top-level field names must be unique.")
        for field in self.fields:
            if depth(field) > MAX_DEPTH:
                raise ValueError(f"Field '{field.name}' nests deeper than {MAX_DEPTH} levels.")
        return self


class ExtractionSchema(SchemaDraft):
    id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
    version: int = Field(ge=1)
    created_at: str = Field(max_length=100)
    exemplars: list[str] = Field(default_factory=list, max_length=20)


def depth(field: SchemaField) -> int:
    return 1 + max((depth(child) for child in field.fields), default=0)


def field_path(prefix: str, field: SchemaField) -> str:
    """Display path of a field, e.g. attendees[].name."""
    name = f"{prefix}.{field.name}" if prefix else field.name
    return f"{name}[]" if field.array else name


def iter_fields(fields: list[SchemaField], prefix: str = "") -> Iterator[tuple[str, SchemaField]]:
    for field in fields:
        path = field_path(prefix, field)
        yield path, field
        if field.type == "object":
            yield from iter_fields(field.fields, path)


def field_guide(fields: list[SchemaField], prefix: str = "", indent: str = "") -> str:
    """Human/LLM readable description of every field path, type and guidance."""
    lines = []
    for field in fields:
        path = field_path(prefix, field)
        kind = TYPE_LABELS[field.type]
        if field.type == "enum":
            kind += ": " + ", ".join(f'"{option}"' for option in field.options)
        if field.array:
            kind = f"list of {'objects' if field.type == 'object' else kind}"
        line = f"{indent}- {path} ({kind})"
        if field.description:
            line += f": {field.description}"
        lines.append(line)
        if field.type == "object":
            lines.append(field_guide(field.fields, path, indent + "  "))
    return "\n".join(lines)


def empty_value(field: SchemaField) -> JsonValue:
    if field.array:
        return []
    if field.type == "object":
        return {child.name: empty_value(child) for child in field.fields}
    return None


MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]


def parse_date(value: str) -> date | None:
    text = value.strip()
    try:
        return datetime.fromisoformat(text[:10] if re.match(r"^\d{4}-\d{2}-\d{2}", text) else text).date()
    except ValueError:
        pass
    cleaned = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text.casefold())
    cleaned = re.sub(r"^(mon|tue|tues|wed|wednes|thu|thur|thurs|fri|sat|satur|sun)(day)?\.?,?\s+", "", cleaned)
    # "22 October 2019", "22 of October 2019", "22-Oct-2019", "22/Oct/2019"
    match = re.match(r"^(\d{1,2})(?:\s+of\s+|\s+|[-/])([a-z]+)\.?(?:,?\s+|[-/])(\d{4})$", cleaned)
    if match:
        day, month, year = match.groups()
    else:
        match = re.match(r"^([a-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})$", cleaned)
        if match:
            month, day, year = match.groups()
        else:
            match = re.match(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})$", cleaned)
            if not match:
                return None
            day, month_number, year = match.groups()
            try:
                return date(int(year), int(month_number), int(day))
            except ValueError:
                return None
    month_index = next((index for index, name in enumerate(MONTHS, 1)
                        if len(month) >= 3 and name.startswith(month)), None)
    if month_index is None:
        return None
    try:
        return date(int(year), month_index, int(day))
    except ValueError:
        return None


DASHES = str.maketrans({"\u2212": "-", "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
                        "\u2014": "-", "\ufe63": "-", "\uff0d": "-"})
MAGNITUDES = {"k": 3, "thousand": 3, "m": 6, "mn": 6, "million": 6, "bn": 9, "billion": 9, "tn": 12, "trillion": 12}
NUMBER = re.compile(r"(-?)[^\d-]*?(-?[\d,]*\.?\d+)(?:([km])|\s*(thousand|million|billion|trillion|mn|bn|tn))?\b([^\d]*)",
                    re.IGNORECASE)


def parse_number(text: str) -> int | float | None:
    """A number written in text: currency/units around it, a typographic minus, a magnitude word."""
    match = NUMBER.fullmatch(text.translate(DASHES))
    if not match:
        return None
    sign, digits, letter, word, _ = match.groups()
    number = Decimal(("" if digits.startswith("-") else sign) + digits.replace(",", ""))
    number = number.scaleb(MAGNITUDES[(letter or word).lower()] if letter or word else 0)
    return int(number) if number == number.to_integral_value() else float(number)


def coerce_scalar(value: JsonValue, field: SchemaField) -> JsonValue:
    """Normalise a model-produced scalar to the field type; unparseable values stay as strings."""
    if value is None or isinstance(value, (dict, list)):
        return None if isinstance(value, (dict, list)) else value
    if isinstance(value, str):
        value = value.strip()
        if not value or value.casefold() in ("null", "none", "n/a"):
            return None
    if field.type in ("text", "verbatim"):
        return str(value)
    if field.type == "date":
        parsed = parse_date(str(value))
        return parsed.isoformat() if parsed else str(value)
    if field.type in ("number", "integer"):
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, (int, float)):
            return int(value) if field.type == "integer" and float(value).is_integer() else value
        # A non-integral value for an integer field stays as written (as for JSON numbers).
        number = parse_number(str(value))
        return str(value) if number is None else number
    if field.type == "boolean":
        if isinstance(value, bool):
            return value
        text = str(value).casefold()
        return True if text in ("true", "yes", "y") else False if text in ("false", "no", "n") else str(value)
    if field.type == "enum":
        text = str(value)
        return next((option for option in field.options if option.casefold() == text.casefold()), text)
    return value


def coerce(value: JsonValue, field: SchemaField) -> JsonValue:
    """Shape arbitrary model JSON to the schema: missing keys become null/[], unknown keys are dropped."""
    if field.array:
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        single = field.model_copy(update={"array": False})
        coerced = [coerce(item, single) for item in items]
        return [item for item in coerced if not is_empty(item)]
    if field.type == "object":
        source = value if isinstance(value, dict) else {}
        return {child.name: coerce(source.get(child.name), child) for child in field.fields}
    return coerce_scalar(value, field)


def coerce_record(data: JsonValue, fields: list[SchemaField]) -> dict:
    source = data if isinstance(data, dict) else {}
    return {field.name: coerce(source.get(field.name), field) for field in fields}


def is_empty(value: JsonValue) -> bool:
    if value is None or value == [] or value == "":
        return True
    if isinstance(value, dict):
        return all(is_empty(item) for item in value.values())
    return False


def meeting_minutes_schema() -> dict:
    """Starter schema for the meeting-minutes test corpus (also used by tests)."""
    return {
        "name": "Meeting minutes",
        "description": "Formal minutes of board, committee or management meetings.",
        "fields": [
            {"name": "meeting_title", "type": "verbatim", "description": "Title or heading of the meeting as written."},
            {"name": "organisation", "type": "text", "description": "Organisation or body holding the meeting."},
            {"name": "meeting_date", "type": "date", "description": "Date the meeting was held."},
            {"name": "location", "type": "verbatim", "description": "Venue or location, if stated."},
            {"name": "chair", "type": "verbatim", "description": "Name of the person chairing the meeting."},
            {"name": "attendees", "type": "object", "array": True,
             "description": "Every person listed as present, in attendance or sending apologies.",
             "fields": [
                 {"name": "name", "type": "verbatim", "description": "Person's name as written."},
                 {"name": "role", "type": "verbatim", "description": "Role, title or organisation, if stated."},
                 {"name": "attendance", "type": "enum", "options": ["present", "in attendance", "apologies", "absent"],
                  "description": "Attendance category the person is listed under."},
             ]},
            {"name": "agenda_items", "type": "object", "array": True,
             "description": "Numbered or titled agenda items discussed.",
             "fields": [
                 {"name": "number", "type": "verbatim", "description": "Item number, if any."},
                 {"name": "title", "type": "verbatim", "description": "Item title."},
                 {"name": "decisions", "type": "text", "array": True,
                  "description": "Decisions, approvals or agreements recorded under the item."},
             ]},
            {"name": "actions", "type": "object", "array": True,
             "description": "Action points assigned to people.",
             "fields": [
                 {"name": "description", "type": "text", "description": "What must be done."},
                 {"name": "owner", "type": "verbatim", "description": "Person or team responsible."},
                 {"name": "due_date", "type": "date", "description": "Deadline, if stated."},
             ]},
            {"name": "next_meeting_date", "type": "date", "description": "Date of the next meeting, if stated."},
        ],
    }
