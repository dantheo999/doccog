import json

import httpx
import pytest
from openai import OpenAI

from app.cognition import CognitionError, extract_document, generate_recipe
from app.documents import parse_document
from app.models import ExtractionOptions, Recipe
from app.settings import Settings


def settings():
    return Settings(endpoint="https://example.openai.azure.com", deployment="test", api_key="test")


def recipe():
    return Recipe(id="r1", version=1, instructions="Extract people.", goal="Find people",
                  source_name="source.txt", created_at="2026-10-13",
                  examples=[{"text": "Alice signed.", "extractions": [
                      {"extraction_class": "person", "extraction_text": "Alice"},
                  ]}])


def completion(content, finish_reason="stop"):
    return httpx.Response(200, json={
        "id": "test", "object": "chat.completion", "created": 0, "model": "test",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": {
            "role": "assistant", "content": content,
        }}],
    })


def generation(quote):
    return json.dumps({"instructions": "Extract people.", "examples": [
        {"passage_ids": ["source-0:0"], "extractions": [
            {"extraction_class": "person", "extraction_text": quote, "attributes": {}},
        ]},
    ]})


def test_generation_corrects_invalid_quotes_once_without_dropping_labels(caplog):
    requests = []
    replies = iter([generation("Invented"), generation("Alice")])
    def respond(request):
        requests.append(json.loads(request.content))
        return completion(next(replies))
    with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        result = generate_recipe(parse_document("a.txt", b"Alice signed."), "Find people", settings(), client=client)
    assert result.examples[0].text == "Alice signed."
    assert len(result.examples[0].extractions) == 1
    assert result.examples[0].extractions[0].extraction_text == "Alice"
    assert len(requests) == 2
    feedback = json.loads(requests[1]["messages"][1]["content"])["validation_feedback"]
    assert "exact" in feedback
    assert "Invented" not in caplog.text


@pytest.mark.parametrize("bad", [
    "not JSON",
    '{"people": []}',
    '{"extractions": [{"person": {"name": "Alice"}}]}',
    '{"extractions": [{"extraction_class": "person", "extraction_text": "Alice", "attributes": {}}]}',
])
@pytest.mark.parametrize("max_workers", [1, 6])
def test_only_the_malformed_chunk_is_retried_with_real_langextract(monkeypatch, bad, max_workers, caplog):
    attempts = {}
    def respond(request):
        body = json.loads(request.content)
        prompt = body["messages"][-1]["content"]
        target = prompt.rsplit("\nQ: ", 1)[1].removesuffix("\nA: ")
        attempts[target] = attempts.get(target, 0) + 1
        if target.startswith("Alice") and attempts[target] == 1:
            return completion(bad)
        return completion('{"extractions": [{"person": "Alice", "person_attributes": {}}]}')
    client = OpenAI(api_key="test", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: client)
    document = parse_document("a.txt", (("Alice signed. " * 40) + ("Bob signed. " * 40)).encode())
    result = extract_document(document, recipe(), settings(), ExtractionOptions(chunk_size=500, max_workers=max_workers))
    assert len(attempts) > 1
    assert any(count == 2 for count in attempts.values())
    assert all(count == 1 for target, count in attempts.items() if target.startswith("Bob"))
    assert result.extractions
    assert bad not in caplog.text


def test_repeated_malformed_output_fails_explicitly_after_two_calls(monkeypatch, caplog):
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        return completion('{"extractions": [{"PRIVATE_MARKER": {"name": "PRIVATE_MARKER"}}]}')
    client = OpenAI(api_key="test", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: client)
    with pytest.raises(CognitionError, match="format.*retry|retry.*format") as failure:
        extract_document(parse_document("a.txt", b"Alice signed."), recipe(), settings())
    assert len(requests) == 2
    assert "PRIVATE_MARKER" not in str(failure.value)
    assert "PRIVATE_MARKER" not in caplog.text


def test_repeated_generation_validation_failure_stops_after_two_calls():
    requests = []
    def respond(request):
        requests.append(request)
        return completion(generation("Invented"))
    with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        with pytest.raises(CognitionError, match="retry"):
            generate_recipe(parse_document("a.txt", b"Alice signed."), "Find people", settings(), client=client)
    assert len(requests) == 2


def test_generation_correction_cannot_drop_the_invalid_annotation():
    initial = json.loads(generation("Invented"))
    initial["examples"][0]["extractions"].append({
        "extraction_class": "action", "extraction_text": "signed", "attributes": {},
    })
    replies = iter([json.dumps(initial), generation("Alice")])
    with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(
        lambda _: completion(next(replies)),
    ))) as client:
        with pytest.raises(CognitionError, match="annotation"):
            generate_recipe(parse_document("a.txt", b"Alice signed."), "Find people", settings(), client=client)


def test_extraction_retries_truncated_json_even_if_the_json_is_parseable(monkeypatch):
    requests = []
    replies = iter([
        completion('{"extractions": []}', "length"),
        completion('{"extractions": [{"person": "Alice", "person_attributes": {}}]}'),
    ])
    def respond(request):
        requests.append(request)
        return next(replies)
    client = OpenAI(api_key="test", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: client)
    result = extract_document(parse_document("a.txt", b"Alice signed."), recipe(), settings())
    assert len(requests) == 2
    assert result.extractions[0].extraction_text == "Alice"
    assert result.extractions[0].grounding == "exact"


def test_generation_corrects_invalid_json_once():
    requests = []
    replies = iter(["not JSON", generation("Alice")])
    def respond(request):
        requests.append(request)
        return completion(next(replies))
    with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        result = generate_recipe(parse_document("a.txt", b"Alice signed."), "Find people", settings(), client=client)
    assert len(requests) == 2
    assert result.examples[0].extractions[0].extraction_text == "Alice"


@pytest.mark.parametrize("operation", ["generate", "extract"])
def test_authentication_failure_is_not_a_corrective_retry(monkeypatch, operation):
    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(401, json={"error": {"message": "PRIVATE_CREDENTIAL_MARKER"}})
    client = OpenAI(api_key="test", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: client)
    doc = parse_document("a.txt", b"Alice signed.")
    with pytest.raises(CognitionError) as failure:
        if operation == "generate":
            generate_recipe(doc, "Find people", settings(), client=client)
        else:
            extract_document(doc, recipe(), settings())
    assert len(requests) == 1
    assert "PRIVATE_CREDENTIAL_MARKER" not in str(failure.value)
    client.close()
