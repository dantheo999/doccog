import logging
import threading

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.cognition import CognitionError, extract_document, generate_recipe, optimize_recipe
from app.documents import DocumentError, DocumentStore, MAX_UPLOAD_BYTES, parse_document
from app.models import ExtractRequest, GenerateRequest, OptimizationProposal, OptimizeRequest, Recipe
from app.settings import load_settings


app = FastAPI(title="Document Cognition", docs_url="/api/docs", openapi_url="/api/openapi.json")
store = DocumentStore()
model_lock = threading.Lock()


@app.exception_handler(DocumentError)
@app.exception_handler(CognitionError)
async def known_error_handler(request, error):
    return JSONResponse(status_code=error.status, content={"detail": str(error)})


def settings_or_error():
    try:
        return load_settings()
    except ValueError as error:
        raise CognitionError(str(error), 503) from error


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


@app.post("/api/documents")
def upload_document(file: UploadFile):
    try:
        content = file.file.read(MAX_UPLOAD_BYTES + 1)
        return store.add(parse_document(file.filename or "", content)).response()
    finally:
        file.file.close()


@app.post("/api/recipes/generate", response_model=Recipe)
def generate(request: GenerateRequest):
    documents = [store.get(document_id) for document_id in request.source_ids]
    if not model_lock.acquire(blocking=False):
        raise HTTPException(409, "Another model operation is in progress. Wait for it to finish.")
    try:
        return generate_recipe(documents, request.goal, settings_or_error())
    finally:
        model_lock.release()


@app.post("/api/recipes/optimize", response_model=OptimizationProposal)
def optimize(request: OptimizeRequest):
    document = store.get(request.document_id)
    if not model_lock.acquire(blocking=False):
        raise HTTPException(409, "Another model operation is in progress. Wait for it to finish.")
    try:
        return optimize_recipe(document, request.recipe, request.result, settings_or_error())
    finally:
        model_lock.release()


@app.post("/api/extractions")
def extract(request: ExtractRequest):
    document = store.get(request.document_id)
    if not model_lock.acquire(blocking=False):
        raise HTTPException(409, "Another model operation is in progress. Wait for it to finish.")
    try:
        return extract_document(document, request.recipe, settings_or_error(), request.options)
    finally:
        model_lock.release()
