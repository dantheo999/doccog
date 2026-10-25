import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import ExtractionResult, ExtractRequest


client = TestClient(app)


def test_health_does_not_require_model_credentials():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_answers_only_local_host_headers():
    """A DNS-rebinding page reaches the API with its own host name in the Host header."""
    assert client.get("/api/health", headers={"host": "rebind.attacker.example"}).status_code == 400
    assert client.get("/api/collections", headers={"host": "rebind.attacker.example:8088"}).status_code == 400
    for host in ("localhost:8088", "127.0.0.1:8010", "[::1]:5173", "backend:8000"):
        assert client.get("/api/health", headers={"host": host}).status_code == 200


def test_upload_returns_canonical_text():
    response = client.post("/api/documents", files={"file": ("a.txt", b"Alice signed.", "text/plain")})
    assert response.status_code == 200
    assert response.json()["text"] == "Alice signed."
    assert response.json()["characters"] == 13


def test_bad_upload_has_actionable_error():
    response = client.post("/api/documents", files={"file": ("a.docx", b"bad", "application/octet-stream")})
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], str)


def test_unknown_document_is_explicitly_expired():
    response = client.post("/api/recipes/generate", json={"document_id": "missing", "goal": "Find names"})
    assert response.status_code == 410
    assert "expired" in response.json()["detail"]


def test_config_never_returns_api_key():
    response = client.get("/api/config")
    assert "api_key" not in response.json()
    assert "configured" in response.json()


def extraction_body():
    return {
        "document_id": "missing",
        "recipe": {
            "id": "test", "version": 1, "instructions": "Extract people.",
            "goal": "Find people", "source_name": "sample.txt", "created_at": "2026-10-12",
            "examples": [{"text": "Alice signed.", "extractions": [
                {"extraction_class": "person", "extraction_text": "Alice"},
            ]}],
        },
    }


def test_legacy_extraction_requests_get_existing_processing_defaults():
    request = ExtractRequest.model_validate(extraction_body())
    assert hasattr(request, "options"), "Extraction request needs default options"
    assert request.options.model_dump() == {
        "chunk_size": 6000, "max_workers": 2,
        "extraction_passes": 1, "context_window_chars": 0,
    }


@pytest.mark.parametrize("options", [
    {"max_workers": 17}, {"chunk_size": 100.5}, {"extraction_passes": True},
    {"context_window_chars": -1}, {"max_workers": "2"}, {"unknown": 1},
])
def test_invalid_options_are_rejected_before_model_processing(options):
    response = client.post("/api/extractions", json={**extraction_body(), "options": options})
    assert response.status_code == 422


def test_endpoint_forwards_valid_options_and_returns_the_snapshot(monkeypatch):
    from app.documents import parse_document
    from app.settings import Settings

    requested = {"chunk_size": 1200, "max_workers": 8, "extraction_passes": 3, "context_window_chars": 300}
    monkeypatch.setattr("app.main.store.get", lambda _: parse_document("a.txt", b"Alice signed."))
    monkeypatch.setattr("app.main.settings_or_error", lambda: Settings(
        endpoint="https://example.openai.azure.com", deployment="test", api_key="test"))

    def extract(document, recipe, settings, options):
        return ExtractionResult(document_id=document.id, text=document.text,
                                recipe_id=recipe.id, recipe_version=recipe.version,
                                options=options, extractions=[], elapsed_seconds=0.1)

    monkeypatch.setattr("app.main.extract_document", extract)
    response = client.post("/api/extractions", json={**extraction_body(), "options": requested})
    assert response.status_code == 200
    assert response.json()["options"] == requested
    assert response.json()["recipe_id"] == "test"


def test_generation_accepts_multiple_sources_and_rejects_duplicates(monkeypatch):
    from app.documents import parse_document
    monkeypatch.setattr("app.main.store.get", lambda key: parse_document(key + ".txt", b"Alice signed."))
    monkeypatch.setattr("app.main.settings_or_error", lambda: None)
    monkeypatch.setattr("app.main.generate_recipe", lambda documents, goal, settings: {
        **extraction_body()["recipe"], "goal": goal,
        "source_name": ", ".join(document.name for document in documents),
    })
    response = client.post("/api/recipes/generate", json={"document_ids": ["a", "b"], "goal": "Find people"})
    assert response.status_code == 200
    assert response.json()["source_name"] == "a.txt, b.txt"
    for ids in ([], ["a", "a"], ["a", "b", "c", "d", "e", "f"]):
        assert client.post("/api/recipes/generate", json={"document_ids": ids, "goal": "Find people"}).status_code == 422


def optimization_body():
    body = extraction_body()
    return {**body, "result": {
        "document_id": body["document_id"], "text": "Alice signed.",
        "recipe_id": "test", "recipe_version": 1, "extractions": [], "elapsed_seconds": 1,
        "options": {"max_workers": 8},
    }}


def test_optimization_endpoint_returns_a_candidate_and_preserves_run_options(monkeypatch):
    from app.documents import parse_document
    from app.models import OptimizationProposal
    monkeypatch.setattr("app.main.store.get", lambda _: parse_document("a.txt", b"Alice signed."))
    monkeypatch.setattr("app.main.settings_or_error", lambda: None)
    def optimize(document, recipe, result, settings):
        assert result.options.max_workers == 8
        return OptimizationProposal(
            base_recipe_id=recipe.id, base_recipe_version=recipe.version,
            candidate=recipe.model_copy(update={"version": 2}), reasons=["Clarify scope."],
        )
    monkeypatch.setattr("app.main.optimize_recipe", optimize)
    response = client.post("/api/recipes/optimize", json=optimization_body())
    assert response.status_code == 200
    assert response.json()["base_recipe_version"] == 1
    assert response.json()["candidate"]["version"] == 2
    assert response.json()["reasons"] == ["Clarify scope."]


def test_optimization_uses_and_releases_the_shared_model_lock(monkeypatch):
    from app.cognition import CognitionError
    from app.documents import parse_document
    from app.main import model_lock
    monkeypatch.setattr("app.main.store.get", lambda _: parse_document("a.txt", b"Alice signed."))
    monkeypatch.setattr("app.main.settings_or_error", lambda: None)
    def fail(*args):
        raise CognitionError("Model quota exhausted.", 429)
    monkeypatch.setattr("app.main.optimize_recipe", fail)
    assert model_lock.acquire(blocking=False)
    try:
        assert client.post("/api/recipes/optimize", json=optimization_body()).status_code == 409
    finally:
        model_lock.release()
    response = client.post("/api/recipes/optimize", json=optimization_body())
    assert response.status_code == 429
    assert response.json()["detail"] == "Model quota exhausted."
    assert model_lock.acquire(blocking=False)
    model_lock.release()


def test_recipe_endpoints_document_their_response_models():
    schema = client.get("/api/openapi.json").json()
    for route, model in (("/api/recipes/generate", "Recipe"), ("/api/recipes/optimize", "OptimizationProposal")):
        response = schema["paths"][route]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
        assert response.get("$ref") == f"#/components/schemas/{model}"
