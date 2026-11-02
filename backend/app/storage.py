"""Small JSON-file stores for collections, schemas, schema suggestions and runs under the data directory."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import threading
from uuid import uuid4

from pydantic import Field, JsonValue

from app.documents import DocumentError, atomic_write
from app.models import StrictModel
from app.results import Run, Usage
from app.schema import ExtractionSchema, SchemaDraft


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
        raise DocumentError("Unknown identifier.", 404)
    return value


class CollectionDocument(StrictModel):
    id: str
    path: str = Field(max_length=2_048)
    name: str
    characters: int
    pages: int | None = None
    sha256: str


class Collection(StrictModel):
    id: str
    name: str = Field(max_length=500)
    created_at: str
    updated_at: str
    documents: list[CollectionDocument] = Field(default_factory=list)


class CollectionCreate(StrictModel):
    name: str = Field(min_length=1, max_length=500)


class JsonFiles:
    def __init__(self, root: Path | None, folder: str):
        self.root = root / folder if root else None
        self._memory: dict[str, dict] = {}
        self.lock = threading.RLock()

    def write(self, key: str, value: dict):
        key = _safe_id(key)
        with self.lock:
            if self.root is None:
                self._memory[key] = json.loads(json.dumps(value))
            else:
                atomic_write(self.root / f"{key}.json", json.dumps(value, ensure_ascii=False).encode())

    def read(self, key: str) -> dict | None:
        key = _safe_id(key)
        with self.lock:
            if self.root is None:
                value = self._memory.get(key)
                return json.loads(json.dumps(value)) if value is not None else None
            try:
                return json.loads((self.root / f"{key}.json").read_bytes())
            except FileNotFoundError:
                return None

    def keys(self) -> list[str]:
        with self.lock:
            if self.root is None:
                return list(self._memory)
            if not self.root.is_dir():
                return []
            return [path.stem for path in self.root.glob("*.json")]

    def delete(self, key: str):
        key = _safe_id(key)
        with self.lock:
            if self.root is None:
                self._memory.pop(key, None)
            else:
                (self.root / f"{key}.json").unlink(missing_ok=True)

    def clear(self):
        with self.lock:
            self._memory.clear()
            if self.root is not None:
                shutil.rmtree(self.root, ignore_errors=True)


class CollectionStore:
    def __init__(self, root: Path | None):
        self.files = JsonFiles(root, "collections")

    def create(self, name: str) -> Collection:
        collection = Collection(id=uuid4().hex[:16], name=name, created_at=now(), updated_at=now())
        self.files.write(collection.id, collection.model_dump())
        return collection

    def get(self, collection_id: str) -> Collection:
        data = self.files.read(collection_id)
        if data is None:
            raise DocumentError("Document set not found. Choose the folder again.", 404)
        return Collection.model_validate(data)

    def add_document(self, collection_id: str, document: CollectionDocument) -> Collection:
        with self.files.lock:
            collection = self.get(collection_id)
            documents = [item for item in collection.documents if item.path != document.path]
            documents.append(document)
            documents.sort(key=lambda item: item.path.casefold())
            collection = collection.model_copy(update={"documents": documents, "updated_at": now()})
            self.files.write(collection.id, collection.model_dump())
            return collection

    def list(self) -> list[dict]:
        items = []
        for key in self.files.keys():
            data = self.files.read(key)
            if data:
                items.append({**{k: v for k, v in data.items() if k != "documents"},
                              "documents": len(data.get("documents", []))})
        return sorted(items, key=lambda item: item["updated_at"], reverse=True)


class SchemaStore:
    """Versioned schemas: saving changed fields under an existing id creates the next version."""

    def __init__(self, root: Path | None):
        self.files = JsonFiles(root, "schemas")

    def save(self, draft: SchemaDraft, schema_id: str | None = None, exemplars: list[str] | None = None
             ) -> ExtractionSchema:
        with self.files.lock:
            history = self.files.read(schema_id) if schema_id else None
            versions = (history or {}).get("versions", [])
            latest = ExtractionSchema.model_validate(versions[-1]) if versions else None
            if latest and SchemaDraft.model_validate(latest.model_dump(include={"name", "description", "fields"})) == draft:
                return latest
            schema = ExtractionSchema(
                **draft.model_dump(), id=schema_id or uuid4().hex[:16],
                version=(latest.version + 1) if latest else 1, created_at=now(),
                exemplars=exemplars if exemplars is not None else (latest.exemplars if latest else []),
            )
            self.files.write(schema.id, {"id": schema.id, "versions": versions + [schema.model_dump()]})
            return schema

    def get(self, schema_id: str, version: int | None = None) -> ExtractionSchema:
        history = self.files.read(schema_id)
        if not history or not history.get("versions"):
            raise DocumentError("Schema not found.", 404)
        versions = history["versions"]
        if version is None:
            return ExtractionSchema.model_validate(versions[-1])
        for item in versions:
            if item["version"] == version:
                return ExtractionSchema.model_validate(item)
        raise DocumentError(f"Schema version {version} not found.", 404)

    def list(self) -> list[dict]:
        items = []
        for key in self.files.keys():
            history = self.files.read(key)
            if history and history.get("versions"):
                latest = history["versions"][-1]
                items.append({"id": latest["id"], "name": latest["name"], "version": latest["version"],
                              "created_at": latest["created_at"], "fields": len(latest["fields"])})
        return sorted(items, key=lambda item: item["created_at"], reverse=True)

    def delete(self, schema_id: str):
        self.files.delete(schema_id)


class SchemaSuggestion(StrictModel):
    """A proposed schema, from the shared suggestion (solution "shared") or one solution's own method."""
    id: str
    solution: str
    solution_name: str
    method: str
    draft: SchemaDraft
    notes: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    elapsed_seconds: float = 0
    exemplars: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    guidance: str = ""
    settings: dict[str, JsonValue] = Field(default_factory=dict)
    created_at: str
    raw: JsonValue = None


class SuggestionStore:
    def __init__(self, root: Path | None):
        self.files = JsonFiles(root, "suggestions")

    def save(self, suggestion: SchemaSuggestion) -> SchemaSuggestion:
        self.files.write(suggestion.id, suggestion.model_dump(mode="json"))
        return suggestion

    def get(self, suggestion_id: str) -> SchemaSuggestion:
        data = self.files.read(suggestion_id)
        if data is None:
            raise DocumentError("Schema suggestion not found.", 404)
        return SchemaSuggestion.model_validate(data)

    def list(self, limit: int = 100) -> list[dict]:
        """Newest first, without the (possibly large) raw output."""
        items = []
        for key in self.files.keys():
            data = self.files.read(key)
            if data and "created_at" in data:
                items.append({k: v for k, v in data.items() if k != "raw"})
        return sorted(items, key=lambda item: item["created_at"], reverse=True)[:limit]

    def delete(self, suggestion_id: str):
        self.files.delete(suggestion_id)


class RunStore:
    def __init__(self, root: Path | None):
        self.files = JsonFiles(root, "runs")

    def save(self, run: Run):
        self.files.write(run.id, run.model_dump(mode="json"))

    def get(self, run_id: str) -> Run:
        data = self.files.read(run_id)
        if data is None:
            raise DocumentError("Run not found.", 404)
        return Run.model_validate(data)

    def list(self) -> list[dict]:
        summaries = []
        for key in self.files.keys():
            try:
                summaries.append(self.get(key).summary())
            except (DocumentError, ValueError):
                continue
        return sorted(summaries, key=lambda item: item["created_at"], reverse=True)

    def delete(self, run_id: str):
        self.files.delete(run_id)
