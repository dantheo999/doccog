"""Contract every extraction solution implements so it can be run, compared and listed."""
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import threading
from typing import Any, Literal

from pydantic import JsonValue

from app.azure import ServiceError
from app.documents import DocumentStore, ParsedDocument
from app.results import Usage
from app.schema import ExtractionSchema, SchemaDraft


@dataclass(frozen=True)
class SettingSpec:
    key: str
    label: str
    type: Literal["integer", "number", "boolean", "select", "text"]
    default: Any
    help: str = ""
    min: float | None = None
    max: float | None = None
    options: list[str] | None = None

    def validate(self, value: Any) -> Any:
        if value is None:
            return self.default
        if self.type == "boolean":
            if not isinstance(value, bool):
                raise ServiceError(f"{self.label} must be true or false.", 422)
        elif self.type in ("integer", "number"):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or (
                    self.type == "integer" and not float(value).is_integer()):
                raise ServiceError(f"{self.label} must be a {'whole ' if self.type == 'integer' else ''}number.", 422)
            value = int(value) if self.type == "integer" else float(value)
            if (self.min is not None and value < self.min) or (self.max is not None and value > self.max):
                raise ServiceError(f"{self.label} must be between {self.min:g} and {self.max:g}.", 422)
        elif self.type == "select":
            if value not in (self.options or []):
                raise ServiceError(f"{self.label} must be one of: {', '.join(self.options or [])}.", 422)
        elif not isinstance(value, str) or len(value) > 8_000:
            raise ServiceError(f"{self.label} must be text of at most 8,000 characters.", 422)
        return value


@dataclass
class Outcome:
    """Annotated (Leaf-valued) schema record plus run metadata for one document."""
    record: dict
    usage: Usage = field(default_factory=Usage)
    notes: list[str] = field(default_factory=list)
    raw: JsonValue = None


@dataclass
class SuggestionOutcome:
    """A schema proposed by a solution from exemplar documents, with notes and spend."""
    draft: SchemaDraft
    notes: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    raw: JsonValue = None


def resolve(specs: list[SettingSpec], settings: dict) -> dict:
    """Defaults overlaid with any valid known keys; unknown keys and invalid values are ignored."""
    resolved = {}
    for spec in specs:
        try:
            resolved[spec.key] = spec.validate(settings.get(spec.key))
        except ServiceError:
            resolved[spec.key] = spec.default
    return resolved


@dataclass
class RunContext:
    store: DocumentStore
    cancel: threading.Event = field(default_factory=threading.Event)

    def check(self):
        if self.cancel.is_set():
            raise ServiceError("Run cancelled.", 409)


class Solution(ABC):
    id: str
    name: str
    tagline: str
    description: str
    technology: str
    runtime: Literal["cloud", "local"]
    capabilities: list[str]
    confidence_method: str
    route: str
    accent: str = "#b4a2ff"
    document_concurrency: int = 1
    external_url: str | None = None  # a packaged third-party UI the landing page can launch
    schema_suggestion_method: str | None = None  # how the solution proposes a schema; None = unsupported
    schema_suggestion_unsupported: str = "This solution does not suggest schemas."

    def settings_spec(self) -> list[SettingSpec]:
        return []

    def resolve_settings(self, settings: dict) -> dict:
        """Defaults overlaid with any valid known keys (for previews and pre-processing)."""
        return resolve(self.settings_spec(), settings)

    def preview(self, schema: ExtractionSchema, settings: dict) -> dict[str, str]:
        """Named text blocks showing what the solution sends for this schema."""
        return {}

    def validate_settings(self, settings: dict) -> dict:
        specs = {spec.key: spec for spec in self.settings_spec()}
        unknown = set(settings) - set(specs)
        if unknown:
            raise ServiceError(f"Unknown setting(s): {', '.join(sorted(unknown))}.", 422)
        return {key: spec.validate(settings.get(key)) for key, spec in specs.items()}

    @abstractmethod
    def status(self) -> dict:
        """{"ready": bool, "detail": str, "model": str}. Must be fast and never raise."""

    def model_label(self, settings: dict) -> str:
        return self.status().get("model") or ""

    def prepare(self, document: ParsedDocument, settings: dict, context: RunContext) -> list[str]:
        """Solution-specific pre-processing (cached by the document store). Returns notes."""
        return []

    @abstractmethod
    def extract(self, document: ParsedDocument, schema: ExtractionSchema, settings: dict,
                context: RunContext) -> Outcome:
        """Extract one document. Raise ServiceError with a user-facing message on failure."""

    # -- optional capability: propose a schema from exemplar documents --------------------

    def suggestion_settings_spec(self) -> list[SettingSpec]:
        return []

    def resolve_suggestion_settings(self, settings: dict) -> dict:
        return resolve(self.suggestion_settings_spec(), settings)

    def suggest_schema(self, documents: list[ParsedDocument], guidance: str, settings: dict,
                       context: RunContext) -> SuggestionOutcome:
        """Propose a schema from 1-5 exemplars. Only called when schema_suggestion_method is set."""
        raise ServiceError(self.schema_suggestion_unsupported, 422)

    def info(self) -> dict:
        return {
            "id": self.id, "name": self.name, "tagline": self.tagline, "description": self.description,
            "technology": self.technology, "runtime": self.runtime, "capabilities": self.capabilities,
            "confidence_method": self.confidence_method, "route": self.route, "accent": self.accent,
            "external_url": self.external_url,
            "settings": [asdict(spec) for spec in self.settings_spec()], "status": self.status(),
            "schema_suggestion": {
                "supported": self.schema_suggestion_method is not None,
                "method": self.schema_suggestion_method or self.schema_suggestion_unsupported,
                "settings": [asdict(spec) for spec in self.suggestion_settings_spec()],
            },
        }
