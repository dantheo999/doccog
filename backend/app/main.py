import logging
import threading
from uuid import uuid4

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import Field

from app.azure import REASONING_EFFORTS, ServiceError
from app.cognition import CognitionError, extract_document, generate_recipe, optimize_recipe
from app.compare import compare_runs
from app.documents import DocumentError, DocumentStore, MAX_UPLOAD_BYTES, parse_document
from app.jobs import JobRunner
from app.models import ExtractRequest, GenerateRequest, OptimizationProposal, OptimizeRequest, Recipe, StrictModel
from app.results import DocumentResult, Run, RunCreate, RunImport, annotate_plain, finalise
from app.schema import SchemaDraft, meeting_minutes_schema
from app.schema_suggest import suggest_schema
from app.settings import data_dir, load_settings
from app.solutions import build_registry
from app.solutions.base import RunContext
from app.solutions.langextract import class_guide
from app.storage import CollectionCreate, CollectionDocument, CollectionStore, RunStore, SchemaStore, now


app = FastAPI(title="Document Cognition", docs_url="/api/docs", openapi_url="/api/openapi.json")
root = data_dir()
store = DocumentStore(root)
collections = CollectionStore(root)
schemas = SchemaStore(root)
runs = RunStore(root)
runner = JobRunner(runs, store)
solutions = build_registry()
model_lock = threading.Lock()
runner.recover()


@app.exception_handler(DocumentError)
@app.exception_handler(CognitionError)
@app.exception_handler(ServiceError)
async def known_error_handler(request, error):
    return JSONResponse(status_code=error.status, content={"detail": str(error)})


def settings_or_error():
    try:
        return load_settings()
    except ValueError as error:
        raise CognitionError(str(error), 503) from error


def schema_kwargs(schema_id: str | None, reasoning_effort: str | None = None) -> dict:
    """Optional schema/model arguments, omitted when unset so legacy calls are unchanged."""
    kwargs = {"class_guide": class_guide(schemas.get(schema_id))} if schema_id else {}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    return kwargs


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def config():
    try:
        settings = load_settings()
        return {
            "configured": settings.configured, "deployment": settings.deployment,
            "endpoint": settings.endpoint, "error": None,
            "max_upload_mib": 20, "max_characters": 500_000,
        }
    except ValueError as error:
        return {"configured": False, "deployment": "", "endpoint": "", "error": str(error),
                "max_upload_mib": 20, "max_characters": 500_000}


# --- Solutions ---------------------------------------------------------------------------

def solution_or_404(solution_id: str):
    solution = solutions.get(solution_id)
    if solution is None:
        raise HTTPException(404, f"Unknown solution '{solution_id}'.")
    return solution


@app.get("/api/solutions")
def list_solutions():
    return [solution.info() for solution in solutions.values()]


@app.get("/api/solutions/{solution_id}")
def get_solution(solution_id: str):
    return solution_or_404(solution_id).info()


class PreviewRequest(StrictModel):
    schema_id: str
    schema_version: int | None = None
    settings: dict = Field(default_factory=dict)


@app.post("/api/solutions/{solution_id}/preview")
def preview_solution(solution_id: str, request: PreviewRequest):
    """What the solution will send for this schema (prompts, templates, class guides)."""
    solution = solution_or_404(solution_id)
    schema = schemas.get(request.schema_id, request.schema_version)
    return solution.preview(schema, solution.resolve_settings(request.settings))


class PrepareRequest(StrictModel):
    document_id: str
    settings: dict = Field(default_factory=dict)


@app.post("/api/solutions/{solution_id}/prepare")
def prepare_document(solution_id: str, request: PrepareRequest):
    """Run a solution's one-off pre-processing (cached) for a document, e.g. page rendering."""
    solution = solution_or_404(solution_id)
    notes = solution.prepare(store.get(request.document_id), solution.resolve_settings(request.settings),
                             RunContext(store=store))
    return {"document_id": request.document_id, "notes": notes}


# --- Documents and collections -----------------------------------------------------------

@app.post("/api/documents")
def upload_document(file: UploadFile, collection_id: str | None = Form(default=None),
                    path: str | None = Form(default=None)):
    try:
        content = file.file.read(MAX_UPLOAD_BYTES + 1)
        document = store.add(parse_document(file.filename or "", content), content)
    finally:
        file.file.close()
    if collection_id:
        collections.add_document(collection_id, CollectionDocument(
            id=document.id, path=(path or document.name)[:2_048], name=document.name,
            characters=len(document.text), pages=document.pages, sha256=document.sha256))
    return document.response()


@app.get("/api/documents/{document_id}")
def get_document(document_id: str):
    return store.get(document_id).response()


@app.post("/api/collections")
def create_collection(request: CollectionCreate):
    return collections.create(request.name)


@app.get("/api/collections")
def list_collections():
    return collections.list()


@app.get("/api/collections/{collection_id}")
def get_collection(collection_id: str):
    return collections.get(collection_id)


# --- Schemas -----------------------------------------------------------------------------

class SchemaSave(StrictModel):
    schema_id: str | None = None
    draft: SchemaDraft
    exemplars: list[str] | None = Field(default=None, max_length=20)


class SuggestRequest(StrictModel):
    document_ids: list[str] = Field(min_length=1, max_length=5)
    guidance: str = Field(default="", max_length=4_000)
    reasoning_effort: str = "medium"


@app.get("/api/schemas")
def list_schemas():
    return schemas.list()


@app.get("/api/schemas/starter")
def starter_schema():
    return SchemaDraft.model_validate(meeting_minutes_schema())


@app.get("/api/schemas/{schema_id}")
def get_schema(schema_id: str, version: int | None = None):
    return schemas.get(schema_id, version)


@app.post("/api/schemas")
def save_schema(request: SchemaSave):
    return schemas.save(request.draft, request.schema_id, request.exemplars)


@app.delete("/api/schemas/{schema_id}")
def delete_schema(schema_id: str):
    schemas.delete(schema_id)
    return {"deleted": schema_id}


@app.post("/api/schemas/suggest")
def suggest(request: SuggestRequest):
    if request.reasoning_effort not in REASONING_EFFORTS:
        raise HTTPException(422, f"reasoning_effort must be one of {', '.join(REASONING_EFFORTS)}.")
    documents = [store.get(document_id) for document_id in dict.fromkeys(request.document_ids)]
    draft, notes, usage = suggest_schema(documents, request.guidance, settings_or_error(), request.reasoning_effort)
    return {"draft": draft, "notes": notes, "usage": usage, "exemplars": [document.name for document in documents]}


# --- Runs and comparison -----------------------------------------------------------------

@app.post("/api/runs")
def create_run(request: RunCreate):
    solution = solution_or_404(request.solution)
    schema = schemas.get(request.schema_id, request.schema_version)
    settings = solution.validate_settings(request.settings)
    status = solution.status()
    if not status.get("ready"):
        raise HTTPException(503, status.get("detail") or f"{solution.name} is not ready.")
    documents = [store.get(document_id) for document_id in dict.fromkeys(request.document_ids)]
    run = Run(
        id=uuid4().hex[:16], solution=solution.id, solution_name=solution.name,
        label=request.label.strip() or f"{solution.name} · {len(documents)} doc{'s' if len(documents) != 1 else ''}",
        model=solution.model_label(settings), confidence_method=solution.confidence_method,
        extraction_schema=schema, settings=settings, created_at=now(),
        documents=[DocumentResult(document_id=document.id, document_name=document.name,
                                  document_sha256=document.sha256) for document in documents],
    )
    runner.start(run, solution)
    return run.summary()


@app.post("/api/runs/import")
def import_run(request: RunImport):
    """Collate externally produced results into a completed run, grounded like any other."""
    solution = solutions.get(request.solution)
    schema = schemas.get(request.schema_id, request.schema_version)
    results = []
    for record in request.records:
        document = store.get(record.document_id)
        result = DocumentResult(document_id=document.id, document_name=document.name,
                                document_sha256=document.sha256, raw=record.raw,
                                started_at=now(), finished_at=now())
        if record.error or record.data is None:
            result.status, result.error = "error", record.error or "No result for this document."
        else:
            result.data, result.values = finalise(document.text, schema, annotate_plain(record.data, schema.fields))
            result.status = "done"
            ungrounded = sum(value.grounding == "none" for value in result.values)
            if ungrounded:
                result.notes.append(f"{ungrounded} of {len(result.values)} values could not be located in the text.")
        results.append(result)
    run = Run(
        id=uuid4().hex[:16], solution=request.solution,
        solution_name=solution.name if solution else request.solution,
        label=request.label.strip() or f"{solution.name if solution else request.solution} · imported",
        model=request.model or "imported results",
        confidence_method=solution.confidence_method if solution else "Not provided.",
        status="done" if any(item.status == "done" for item in results) else "error",
        extraction_schema=schema, settings={"imported": True}, created_at=now(), started_at=now(),
        finished_at=now(), documents=results,
    )
    runs.save(run)
    return run.summary()


@app.get("/api/runs")
def list_runs():
    return [{**summary, "active": runner.active(summary["id"])} for summary in runs.list()]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run = runs.get(run_id)
    return {**run.model_dump(mode="json"), "summary": run.summary(), "active": runner.active(run_id)}


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str):
    runs.get(run_id)
    return {"cancelled": runner.cancel(run_id)}


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str):
    if runner.active(run_id):
        raise HTTPException(409, "Cancel the run before deleting it.")
    runs.delete(run_id)
    return {"deleted": run_id}


class CompareRequest(StrictModel):
    run_ids: list[str] = Field(min_length=1, max_length=12)
    reference_run_id: str | None = None


@app.post("/api/compare")
def compare(request: CompareRequest):
    selected = [runs.get(run_id) for run_id in dict.fromkeys(request.run_ids)]
    return compare_runs(selected, request.reference_run_id)


# --- LangExtract recipe studio (interactive, one model operation at a time) ---------------

@app.post("/api/recipes/generate", response_model=Recipe)
def generate(request: GenerateRequest):
    documents = [store.get(document_id) for document_id in request.source_ids]
    extra = schema_kwargs(request.schema_id)
    if not model_lock.acquire(blocking=False):
        raise HTTPException(409, "Another model operation is in progress. Wait for it to finish.")
    try:
        return generate_recipe(documents, request.goal, settings_or_error(), **extra)
    finally:
        model_lock.release()


@app.post("/api/recipes/optimize", response_model=OptimizationProposal)
def optimize(request: OptimizeRequest):
    document = store.get(request.document_id)
    extra = schema_kwargs(request.schema_id)
    if not model_lock.acquire(blocking=False):
        raise HTTPException(409, "Another model operation is in progress. Wait for it to finish.")
    try:
        return optimize_recipe(document, request.recipe, request.result, settings_or_error(), **extra)
    finally:
        model_lock.release()


@app.post("/api/extractions")
def extract(request: ExtractRequest):
    document = store.get(request.document_id)
    extra = schema_kwargs(request.schema_id, request.reasoning_effort)
    if not model_lock.acquire(blocking=False):
        raise HTTPException(409, "Another model operation is in progress. Wait for it to finish.")
    try:
        return extract_document(document, request.recipe, settings_or_error(), request.options, **extra)
    finally:
        model_lock.release()
