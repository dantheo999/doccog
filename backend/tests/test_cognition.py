import json
import threading
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI
from pydantic import ValidationError
from langextract.core import types
from langextract.providers.openai import OpenAILanguageModel

from app import models
from app.cognition import CognitionError, build_model_config, extract_document, generate_recipe
from app.documents import parse_document
from app.models import Example, Recipe
from app.settings import Settings


def recipe():
    return Recipe(
        id="test-recipe", version=1, instructions="Extract people using exact source text.",
        goal="Find people", source_name="sample.txt", created_at="2026-10-12T00:00:00Z",
        examples=[Example(text="Alice signed.", extractions=[
            {"extraction_class": "person", "extraction_text": "Alice", "attributes": {}},
        ])],
    )


def settings():
    return Settings(endpoint="https://example.openai.azure.com/", deployment="my-deployment", api_key="test-key")


def test_requires_exact_example_spans():
    with pytest.raises(ValidationError, match="exact"):
        Example(text="Alice signed.", extractions=[
            {"extraction_class": "person", "extraction_text": "Bob", "attributes": {}},
        ])


def test_whole_document_example_text_retains_the_document_unicode_limit():
    text = "\U0001f600" * 500_000
    example = Example(text=text, extractions=[
        {"extraction_class": "symbol", "extraction_text": "\U0001f600"},
    ])
    assert example.text == text
    with pytest.raises(ValidationError):
        Example(text=text + "\U0001f600", extractions=example.extractions)
    with pytest.raises(ValidationError):
        Example(text=text, extractions=[
            {"extraction_class": "symbol", "extraction_text": "\U0001f600" * 12_001},
        ])


def test_rejects_empty_extraction_classes():
    with pytest.raises(ValidationError):
        Example(text="Alice", extractions=[
            {"extraction_class": " ", "extraction_text": "Alice", "attributes": {}},
        ])


def test_routes_arbitrary_azure_deployment_to_openai():
    config = build_model_config(settings())
    assert config.provider == "openai"
    assert config.model_id == "my-deployment"
    assert config.provider_kwargs["base_url"] == "https://example.openai.azure.com/openai/v1/"


def test_rejects_wrong_endpoint_path():
    with pytest.raises(ValueError, match="endpoint"):
        Settings(endpoint="https://example.openai.azure.com/api/projects/demo", deployment="test", api_key="key")


def test_generation_validates_real_openai_response():
    def respond(request):
        body = json.loads(request.content)
        assert body["model"] == "my-deployment"
        assert body["response_format"] == {"type": "json_object"}
        assert "Alice signed." in body["messages"][1]["content"]
        payload = json.loads(body["messages"][1]["content"])
        assert "source_documents" in payload, "Generation must offer real source passages"
        passage = payload["source_documents"][0]["passages"][0]
        generated = {
            "instructions": "Extract people in order of appearance, using exact text.",
            "examples": [{"passage_ids": [passage["passage_id"]], "extractions": [
                {"extraction_class": "person", "extraction_text": "Alice", "attributes": {"role": "signer"}},
            ]}],
        }
        return httpx.Response(200, json={
            "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
            "model": "my-deployment", "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(generated)},
            }],
        })

    client = OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    result = generate_recipe(parse_document("source.txt", b"Alice signed."), "Find people", settings(), client=client)
    assert result.examples[0].extractions[0].extraction_text == "Alice"
    assert result.examples[0].text == "Alice signed."
    assert result.examples[0].source.name == "source.txt"
    assert result.examples[0].source.start == 0
    assert result.examples[0].source.end == 13
    assert result.examples[0].source.verified
    assert result.source_name == "source.txt"
    assert result.goal == "Find people"


def test_rejects_bad_model_examples():
    def respond(request):
        return httpx.Response(200, json={
            "id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": "test",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps({
                    "instructions": "Extract names", "examples": [{"text": "Alice", "extractions": [
                        {"extraction_class": "person", "extraction_text": "Bob"},
                    ]}],
                }),
            }}],
        })
    client = OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    with pytest.raises(CognitionError, match="valid"):
        generate_recipe(parse_document("a.txt", b"Alice"), "Find names", settings(), client=client)


def test_real_langextract_returns_unicode_source_offsets(monkeypatch):
    def deterministic_output(self, prompt, config):
        return types.ScoredOutput(score=1.0, output=json.dumps({
            "extractions": [{"person": "Alice", "person_attributes": {"role": "signer"}}],
        }))
    monkeypatch.setattr(OpenAILanguageModel, "_process_single_prompt", deterministic_output)
    document = parse_document("unicode.txt", "Hi \U0001f600 Alice signed.".encode())
    result = extract_document(document, recipe(), settings())
    assert result.text == "Hi \U0001f600 Alice signed."
    assert result.extractions[0].start == 5
    assert result.extractions[0].end == 10
    assert result.extractions[0].grounding == "exact"
    assert result.recipe_id == "test-recipe"
    assert result.model_dump()["options"] == {
        "chunk_size": 6000, "max_workers": 2,
        "extraction_passes": 1, "context_window_chars": 0,
    }


def test_real_langextract_separates_example_facts_from_the_current_target_chunk(monkeypatch):
    prompts = []
    def respond(self, prompt, config):
        prompts.append(prompt)
        return types.ScoredOutput(score=1.0, output=json.dumps({
            "extractions": [{"person": "Bob", "person_attributes": {"role": "signer"}}],
        }))
    monkeypatch.setattr(OpenAILanguageModel, "_process_single_prompt", respond)
    active = recipe()
    before = active.model_dump()
    extracted = extract_document(parse_document("target.txt", b"Bob signed."), active, settings())
    description = prompts[0].split("\nExamples\n")[0]
    assert "final Q:" in description
    assert "not facts about the target" in description
    assert "Alice signed." in prompts[0]
    assert prompts[0].endswith("Q: Bob signed.\nA: ")
    assert extracted.extractions[0].extraction_text == "Bob"
    assert extracted.extractions[0].grounding == "exact"
    assert active.model_dump() == before


def test_malformed_model_output_fails_instead_of_dropping_chunk(monkeypatch):
    monkeypatch.setattr(OpenAILanguageModel, "_process_single_prompt",
                        lambda *args: types.ScoredOutput(score=1.0, output="not JSON"))
    with pytest.raises(CognitionError):
        extract_document(parse_document("a.txt", b"Alice signed."), recipe(), settings())


def processing_options(**values):
    assert hasattr(models, "ExtractionOptions"), "Extraction options model is missing"
    return models.ExtractionOptions(**values)


@pytest.mark.parametrize("field,minimum,maximum", [
    ("chunk_size", 500, 20_000), ("max_workers", 1, 16),
    ("extraction_passes", 1, 5), ("context_window_chars", 0, 6000),
])
def test_options_accept_exact_range_boundaries(field, minimum, maximum):
    assert getattr(processing_options(**{field: minimum}), field) == minimum
    assert getattr(processing_options(**{field: maximum}), field) == maximum


@pytest.mark.parametrize("field,value", [
    ("chunk_size", 499), ("chunk_size", 20_001), ("chunk_size", 1000.5),
    ("max_workers", 0), ("max_workers", 17), ("max_workers", True),
    ("max_workers", "4"), ("extraction_passes", 0), ("extraction_passes", 6),
    ("context_window_chars", -1), ("context_window_chars", 6001),
    ("context_window_chars", None), ("unknown_option", 1),
])
def test_options_reject_invalid_values_without_coercion(field, value):
    processing_options()
    with pytest.raises(ValidationError):
        processing_options(**{field: value})


def test_forwards_all_processing_options_and_records_the_run(monkeypatch):
    captured = {}

    def extract(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(extractions=[])

    monkeypatch.setattr("app.cognition.lx.extract", extract)
    options = processing_options(chunk_size=1000, max_workers=4,
                                 extraction_passes=2, context_window_chars=250)
    result = extract_document(parse_document("a.txt", b"Alice signed."), recipe(), settings(), options)
    assert captured["max_char_buffer"] == 1000
    assert captured["max_workers"] == captured["batch_length"] == 4
    assert captured["model"].max_workers == 4
    assert captured["extraction_passes"] == 2
    assert captured["context_window_chars"] == 250
    assert result.options == options
    assert result.recipe_id == "test-recipe"


def test_real_langextract_uses_concurrency_context_and_sequential_passes(monkeypatch):
    lock = threading.Lock()
    first_batch = threading.Barrier(3)
    active = peak = 0
    prompts = []

    def respond(self, prompt, config):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            prompts.append(prompt)
            index = len(prompts)
        try:
            if index <= 3:
                first_batch.wait(timeout=5)
            return types.ScoredOutput(score=1.0, output=json.dumps({
                "extractions": [{"person": "Alice", "person_attributes": {}}],
            }))
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(OpenAILanguageModel, "_process_single_prompt", respond)
    options = processing_options(chunk_size=500, max_workers=3,
                                 extraction_passes=2, context_window_chars=50)
    result = extract_document(parse_document("long.txt", b"Alice signed.\n" * 300),
                              recipe(), settings(), options)
    assert peak == 3
    assert len(prompts) > 6
    assert sum("[Previous text]:" not in prompt for prompt in prompts) == 2
    assert result.options == options
    assert result.extractions
    assert all(item.grounding == "exact" for item in result.extractions)
