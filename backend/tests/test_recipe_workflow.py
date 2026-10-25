import hashlib
import json

import httpx
import pytest
from openai import OpenAI

from app import cognition
from app.cognition import CognitionError, generate_recipe
from app.documents import parse_document
from app.models import ExtractionItem, ExtractionOptions, ExtractionResult, Recipe
from app.settings import Settings


def settings():
    return Settings(endpoint="https://example.openai.azure.com", deployment="test", api_key="test")


def recipe():
    return Recipe(id="r1", version=1, goal="Find people", instructions="Extract people.",
                  source_name="legacy.txt", created_at="2026-10-12",
                  examples=[{"text": "Alice signed.", "extractions": [
                      {"extraction_class": "person", "extraction_text": "Alice"},
                  ]}])


def model_client(reply):
    def respond(request):
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        output = reply(payload)
        return httpx.Response(200, json={
            "id": "test", "object": "chat.completion", "created": 0, "model": "test",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps(output),
            }}],
        })
    return OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond)))


def chosen(passage, name="Alice"):
    return {"passage_ids": [passage["passage_id"]], "extractions": [
        {"extraction_class": "person", "extraction_text": name, "attributes": {}},
    ]}


def verified_recipe(documents):
    return generate_recipe(documents, "Find people", settings(), client=model_client(lambda payload: {
        "instructions": "Extract people.", "examples": [
            chosen(source["passages"][0], source["passages"][0]["text"].split()[0])
            for source in payload["source_documents"]
        ],
    }))


def result(document, active=None):
    active = active or recipe()
    return ExtractionResult(document_id=document.id, text=document.text, recipe_id=active.id,
                            recipe_version=active.version, extractions=[], elapsed_seconds=1,
                            options=ExtractionOptions())


def optimize(*args, **kwargs):
    assert hasattr(cognition, "optimize_recipe"), "Optimization service is missing"
    return cognition.optimize_recipe(*args, **kwargs)


def test_multiple_documents_contribute_exact_source_examples():
    documents = [parse_document("a.txt", "Alice signed.\r\n\U0001f600 End.".encode()),
                 parse_document("b.txt", b"Bob signed.")]
    def reply(payload):
        sources = payload["source_documents"]
        assert [item["name"] for item in sources] == ["a.txt", "b.txt"]
        return {"instructions": "Extract people.", "examples": [
            chosen(sources[0]["passages"][0]), chosen(sources[1]["passages"][0], "Bob"),
        ]}
    generated = generate_recipe(documents, "Find people", settings(), client=model_client(reply))
    assert [example.text for example in generated.examples] == [doc.text for doc in documents]
    for example, document in zip(generated.examples, documents):
        assert document.text[example.source.start:example.source.end] == example.text
        assert example.source.document_sha256 == hashlib.sha256(document.text.encode()).hexdigest()
        assert example.source.text_sha256 == hashlib.sha256(example.text.encode()).hexdigest()


def test_generation_cannot_omit_a_selected_source():
    documents = [parse_document("a.txt", b"Alice signed."), parse_document("b.txt", b"Bob signed.")]
    def reply(payload):
        return {"instructions": "Extract people.", "examples": [
            chosen(payload["source_documents"][0]["passages"][0]),
        ]}
    with pytest.raises(CognitionError, match="source|exemplar"):
        generate_recipe(documents, "Find people", settings(), client=model_client(reply))


def test_generation_cannot_choose_an_invented_passage():
    def reply(payload):
        return {"instructions": "Extract people.", "examples": [
            chosen({"passage_id": "invented"}),
        ]}
    with pytest.raises(CognitionError, match="unknown source passage"):
        generate_recipe(parse_document("a.txt", b"Alice signed."), "Find people", settings(),
                        client=model_client(reply))


def test_optimization_sends_the_complete_document_and_all_findings_without_applying():
    document = parse_document("long.txt", (("Alice signed.\r\n" * 1600) + "LAST EVIDENCE.").encode())
    document.id = "doc1"
    active = recipe()
    findings = result(document, active)
    before = active.model_dump()
    observed = {}
    def reply(payload):
        observed.update(payload)
        assert payload["document"]["text"] == document.text
        assert payload["findings"] == []
        assert payload["recipe"]["instructions"] == active.instructions
        assert payload["recipe"]["examples"][0]["text"] == "Alice signed."
        assert payload["extraction_options"]["max_workers"] == 2
        return {"instructions": "Extract named signers in source order.",
                "examples": [],
                "reasons": ["Clarify the requested person role."]}
    proposal = optimize(document, active, findings, settings(), client=model_client(reply))
    assert observed["document"]["text"].endswith("LAST EVIDENCE.")
    assert proposal.base_recipe_id == active.id
    assert proposal.base_recipe_version == 1
    assert proposal.candidate.id == active.id
    assert proposal.candidate.version == 2
    assert proposal.candidate.examples == active.examples
    assert proposal.candidate.examples[0].source is None
    assert active.model_dump() == before
    assert findings.recipe_version == 1


def test_optimization_rejects_stale_recipe_before_inference():
    document = parse_document("a.txt", b"Alice signed.")
    document.id = "doc1"
    findings = result(document)
    findings.recipe_version = 2
    with pytest.raises(CognitionError, match="Extract|recipe"):
        optimize(document, recipe(), findings, settings(),
                 client=model_client(lambda _: pytest.fail("Stale findings reached the model")))


def test_optimization_rejects_mismatched_document_text():
    document = parse_document("a.txt", b"Alice signed.")
    document.id = "doc1"
    findings = result(document)
    findings.text = "A different document."
    with pytest.raises(CognitionError, match="document|Extract"):
        optimize(document, recipe(), findings, settings(),
                 client=model_client(lambda _: pytest.fail("Wrong document reached the model")))


def test_optimization_reports_context_capacity_without_sampling():
    document = parse_document("a.txt", b"Alice signed.")
    document.id = "doc1"
    def respond(request):
        return httpx.Response(400, json={"error": {"message": "Too many tokens",
                                                 "code": "context_length_exceeded"}})
    client = OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    with pytest.raises(CognitionError, match="context"):
        optimize(document, recipe(), result(document), settings(), client=client)


def test_optimization_sends_ungrounded_findings_and_original_run_settings():
    document = parse_document("a.txt", b"Alice signed.")
    document.id = "doc1"
    findings = result(document)
    findings.options = ExtractionOptions(max_workers=9, extraction_passes=3)
    findings.extractions = [
        ExtractionItem(id="1", extraction_class="person", extraction_text="Alice",
                       attributes={"role": "signer"}, start=0, end=5,
                       alignment_status="match_exact", grounding="exact"),
        ExtractionItem(id="2", extraction_class="suspected_person", extraction_text="Bob",
                       attributes={"review": ["unsupported"]}, start=None, end=None,
                       alignment_status=None, grounding="ungrounded"),
    ]
    def reply(payload):
        assert len(payload["findings"]) == 2
        assert payload["findings"][1]["extraction_text"] == "Bob"
        assert payload["findings"][1]["grounding"] == "ungrounded"
        assert payload["findings"][1]["attributes"] == {"review": ["unsupported"]}
        assert payload["extraction_options"]["max_workers"] == 9
        assert payload["extraction_options"]["extraction_passes"] == 3
        return {"instructions": "Extract supported people.",
                "examples": [],
                "reasons": ["Require explicit source support."]}
    proposal = optimize(document, recipe(), findings, settings(), client=model_client(reply))
    assert proposal.candidate.instructions == "Extract supported people."
    assert len(findings.extractions) == 2


def test_optimization_can_reuse_unchanged_source_backed_examples_without_duplicating_text():
    original = parse_document("a.txt", b"Alice signed.")
    original.id = "example-doc"
    active = generate_recipe(original, "Find people", settings(), client=model_client(lambda payload: {
        "instructions": "Extract people.", "examples": [
            chosen(payload["source_documents"][0]["passages"][0]),
        ],
    }))
    target = parse_document("b.txt", b"Bob signed.")
    target.id = "target-doc"
    def reply(payload):
        assert payload["existing_source_passages"] == [{"passage_id": "existing:0", "recipe_example_index": 0}]
        assert "text" not in payload["existing_source_passages"][0]
        return {"instructions": "Extract named signers.", "examples": [
            chosen({"passage_id": "existing:0"}),
        ], "reasons": ["Retain an applicable original exemplar."]}
    proposal = optimize(target, active, result(target, active), settings(), client=model_client(reply))
    assert proposal.candidate.examples[0].model_dump() == active.examples[0].model_dump()
    assert proposal.candidate.examples[0].source.document_id == "example-doc"


def test_generation_stores_the_entire_document_with_selective_annotations():
    document = parse_document("repeat.txt", (("Alice signed.\r\n" * 2500) + "\U0001f600").encode())
    observed = {}
    def reply(payload):
        source = payload["source_documents"][0]
        observed.update(source)
        for passage in source["passages"]:
            assert "start" in passage and "end" in passage, "Sample gaps must be visible to the model"
            assert document.text[passage["start"]:passage["end"]] == passage["text"]
        return {"instructions": "Extract people.", "examples": [chosen(source["passages"][-1])]}
    generated = generate_recipe(document, "Find people", settings(), client=model_client(reply))
    example = generated.examples[0]
    assert observed["characters_used"] == len(document.text)
    assert observed["characters_total"] == len(document.text)
    assert len(observed["passages"]) == 1
    assert observed["passages"][0]["text"] == document.text
    assert example.text == document.text
    assert example.source.start == 0
    assert example.source.end == len(document.text)
    assert "\U0001f600" in example.text
    assert generated.source_characters_used == len(document.text)
    assert len(example.extractions) == 1


def test_generation_rejects_duplicate_examples_for_the_same_document():
    with pytest.raises(CognitionError, match="exactly one"):
        generate_recipe(parse_document("a.txt", b"Alice signed."), "Find people", settings(),
                        client=model_client(lambda payload: {
                            "instructions": "Extract people.", "examples": [
                                chosen(payload["source_documents"][0]["passages"][0]),
                                chosen(payload["source_documents"][0]["passages"][0]),
                            ],
                        }))


def test_optimization_preserves_a_complete_original_document_example():
    document = parse_document("long.txt", b"Alice signed.\n" * 2500)
    document.id = "doc1"
    active = verified_recipe([document])
    observed = {}
    def reply(payload):
        observed.update(payload["document"])
        return {"instructions": "Extract people, not just illustrative example annotations.",
                "examples": [chosen({"passage_id": "existing:0"})],
                "reasons": ["Retain the complete original document with selective labels."]}
    proposal = optimize(document, active, result(document, active), settings(), client=model_client(reply))
    assert observed["text"] == document.text
    assert proposal.candidate.examples[0].text == document.text
    assert proposal.candidate.examples[0].source.start == 0
    assert proposal.candidate.examples[0].source.end == len(document.text)


@pytest.mark.parametrize("ids", [
    ["source-0:0", "source-0:0"],
    ["source-1:0", "source-0:0"],
])
def test_generation_rejects_repeated_or_mixed_source_passages(ids):
    documents = [parse_document("a.txt", b"Alice signed."), parse_document("b.txt", b"Bob signed.")]
    def reply(payload):
        return {"instructions": "Extract people.", "examples": [{
            "passage_ids": ids, "extractions": [
                {"extraction_class": "person", "extraction_text": "Alice"},
            ],
        }]}
    with pytest.raises(CognitionError):
        generate_recipe(documents, "Find people", settings(), client=model_client(reply))


def test_optimization_cannot_select_legacy_example_text():
    document = parse_document("a.txt", b"Alice signed.")
    document.id = "doc1"
    def reply(payload):
        assert payload["existing_source_passages"] == []
        return {"instructions": "Extract people.", "examples": [
            chosen({"passage_id": "existing:0"}),
        ], "reasons": ["Use existing example."]}
    with pytest.raises(CognitionError):
        optimize(document, recipe(), result(document), settings(), client=model_client(reply))


def test_optimization_timeout_is_explicit_and_preserves_the_active_recipe():
    document = parse_document("a.txt", b"Alice signed.")
    document.id = "doc1"
    active = recipe()
    def respond(request):
        raise httpx.ReadTimeout("Model did not respond", request=request)
    client = OpenAI(api_key="test", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    with pytest.raises(CognitionError, match="timed out") as failure:
        optimize(document, active, result(document), settings(), client=client)
    assert failure.value.status == 504
    assert active.version == 1
    assert active.instructions == "Extract people."


@pytest.mark.parametrize("response", [
    {"choices": []},
    {"choices": [{"index": 0, "finish_reason": "length", "message": {"role": "assistant", "content": "{}"}}]},
    {"choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": ""}}]},
    {"choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "refusal": "No", "content": "{}"}}]},
    {"choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "bad JSON"}}]},
])
def test_incomplete_optimization_output_never_substitutes_a_recipe(response):
    document = parse_document("a.txt", b"Alice signed.")
    document.id = "doc1"
    active = recipe()
    original = active.model_dump()
    client = OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json={
            "id": "test", "object": "chat.completion", "created": 0, "model": "test", **response,
        }))))
    with pytest.raises(CognitionError):
        optimize(document, active, result(document), settings(), client=client)
    assert active.model_dump() == original


def test_generation_supplies_the_response_shape_and_allowed_passage_ids():
    def reply(payload):
        schema = payload.get("response_schema")
        assert schema is not None, "Model must receive the actual recipe response contract"
        assert schema["additionalProperties"] is False
        assert schema["required"] == ["instructions", "examples"]
        assert schema["properties"]["examples"]["maxItems"] == 1
        example_schema = schema["$defs"]["SelectedExample"]
        assert example_schema["required"] == ["passage_ids", "extractions"]
        assert example_schema["additionalProperties"] is False
        assert example_schema["properties"]["passage_ids"]["items"]["enum"] == ["source-0:0"]
        assert "text" not in example_schema["properties"]
        return {"instructions": "Extract people.", "examples": [
            chosen(payload["source_documents"][0]["passages"][0]),
        ]}
    generated = generate_recipe(parse_document("a.txt", b"Alice signed."), "Find people", settings(),
                                client=model_client(reply))
    assert generated.examples[0].text == "Alice signed."


@pytest.mark.parametrize("output,expected", [
    ({"instructions": "Extract people.", "examples": [{
        "text": "Alice signed.", "extractions": [
            {"extraction_class": "person", "extraction_text": "Alice"},
        ],
    }]}, ("passage_ids", "required")),
    ({"instructions": "Extract people.", "examples": [{
        "passage_ids": ["source-0:0"], "extractions": [
            {"extraction_class": "person", "extraction_text": "Alice", "attributes": []},
        ],
    }]}, ("attributes", "object")),
    ({"instructions": "Extract people.", "examples": [
        chosen({"passage_id": "source-0:0"}) for _ in range(6)
    ]}, ("examples", "5")),
    ({"instructions": "Extract people.", "examples": []}, ("examples", "1")),
    ({"instructions": "Extract people.", "examples": [
        chosen({"passage_id": "source-0:0"}, "Invented person"),
    ]}, ("extraction_text", "exact")),
])
def test_generation_reports_the_specific_validation_failure(output, expected, caplog):
    with pytest.raises(CognitionError) as failure:
        generate_recipe(parse_document("a.txt", b"Alice signed."), "Find people", settings(),
                        client=model_client(lambda _: output))
    for detail in expected:
        assert detail in str(failure.value)
    assert "Use a more specific goal or remove irrelevant exemplars" not in str(failure.value)
    assert "validation" in caplog.text
    assert "Alice signed." not in caplog.text


def test_validation_diagnostics_do_not_echo_model_values_or_arbitrary_field_names(caplog):
    private_marker = "MODEL_CONTENT_NOT_FOR_LOGS"
    with pytest.raises(CognitionError) as failure:
        generate_recipe(parse_document("a.txt", b"Alice signed."), "Find people", settings(),
                        client=model_client(lambda _: {
                            "instructions": "Extract people.", "examples": [
                                chosen({"passage_id": "source-0:0"}),
                            ],
                            private_marker: private_marker,
                        }))
    assert "unexpected field" in str(failure.value)
    assert private_marker not in str(failure.value)
    assert private_marker not in caplog.text


def test_generation_identifies_the_missing_selected_source():
    documents = [parse_document("a.txt", b"Alice signed."), parse_document("b.txt", b"Bob signed.")]
    def reply(payload):
        return {"instructions": "Extract people.", "examples": [
            chosen(payload["source_documents"][0]["passages"][0]),
        ]}
    with pytest.raises(CognitionError, match="source 2"):
        generate_recipe(documents, "Find people", settings(), client=model_client(reply))


def test_optimization_uses_its_own_response_schema_and_reports_invalid_reasons():
    document = parse_document("a.txt", b"Alice signed.")
    document.id = "doc1"
    def reply(payload):
        schema = payload.get("response_schema")
        assert schema is not None, "Optimization also needs its exact response contract"
        assert schema["required"] == ["instructions", "examples", "reasons"]
        assert schema["properties"]["examples"]["minItems"] == 0
        assert schema["properties"]["examples"]["maxItems"] == 0
        assert "enum" not in schema["$defs"]["SelectedExample"]["properties"]["passage_ids"]["items"]
        return {"instructions": "Extract people.", "examples": [], "reasons": []}
    with pytest.raises(CognitionError, match="reasons"):
        optimize(document, recipe(), result(document), settings(), client=model_client(reply))


@pytest.mark.parametrize("selection", [
    ["source-0:0", "source-0:0"],
    ["source-0:0", "source-1:0"],
])
def test_generation_rejects_more_than_one_source_reference_per_example(selection):
    documents = [parse_document("a.txt", b"Alice signed."), parse_document("b.txt", b"Bob signed.")]
    with pytest.raises(CognitionError, match="passage_ids.*1 item"):
        generate_recipe(documents, "Find people", settings(), client=model_client(lambda _: {
            "instructions": "Extract people.", "examples": [{
                "passage_ids": selection, "extractions": [
                    {"extraction_class": "person", "extraction_text": "Alice"},
                ],
            }],
        }))


@pytest.mark.parametrize("extraction,expected", [
    ("Alice  Smith", "Alice  Smith"), ("Alice Smith", "Alice  Smith"),
    ("Room 6.1,\nBirmingham", "Room 6.1,\nBirmingham"),
    ("Room 6.1, Birmingham", "Room 6.1,\nBirmingham"),
])
def test_unique_pdf_whitespace_is_restored_to_the_literal_source(extraction, expected):
    document = parse_document("minutes.txt", b"Alice  Smith attended.\nRoom 6.1,\nBirmingham")
    client = model_client(lambda payload: {
        "instructions": "Extract people and locations.", "examples": [
            chosen(payload["source_documents"][0]["passages"][0], extraction),
        ],
    })
    generated = generate_recipe(document, "Find people and locations", settings(), client=client)
    assert generated.examples[0].text == document.text
    assert generated.examples[0].extractions[0].extraction_text == expected


def test_source_whitespace_restoration_preserves_unicode_provenance_and_attributes(caplog):
    text = "\U0001f600 Zoë\u00a0Smith  \r\napproved."
    document = parse_document("minutes.txt", text.encode())
    quote = "\U0001f600 Zoë Smith\napproved."
    def reply(payload):
        return {"instructions": "Extract actions.", "examples": [{
            "passage_ids": [payload["source_documents"][0]["passages"][0]["passage_id"]],
            "extractions": [{"extraction_class": "action", "extraction_text": quote,
                             "attributes": {"owner": "Zoë Smith", "review": ["proposed"]}}],
        }]}
    with caplog.at_level("INFO", logger="app.exemplars"):
        generated = generate_recipe(document, "Find actions", settings(), client=model_client(reply))
    example = generated.examples[0]
    assert example.text == text
    assert example.extractions[0].extraction_text == text
    assert example.extractions[0].extraction_class == "action"
    assert example.extractions[0].attributes == {"owner": "Zoë Smith", "review": ["proposed"]}
    assert example.source.start == 0 and example.source.end == len(text)
    assert example.source.text_sha256 == hashlib.sha256(text.encode()).hexdigest()
    assert "Restored source whitespace" in caplog.text
    assert "Zoë" not in caplog.text


@pytest.mark.parametrize("text,quote", [
    ("Alice  Smith attended.", "Alice Brown"),
    ("Alice  Smith attended!", "Alice Smith attended."),
    ("ALICE  SMITH attended.", "alice smith"),
    ("An ofﬁcer attended.", "An officer attended."),
    ("Alice  Smith attended. Alice\nSmith apologised.", "Alice Smith"),
])
def test_whitespace_restoration_rejects_changed_words_punctuation_case_and_ambiguity(text, quote):
    document = parse_document("minutes.txt", text.encode())
    with pytest.raises(CognitionError, match="exact substring"):
        generate_recipe(document, "Find people", settings(), client=model_client(lambda payload: {
            "instructions": "Extract people.", "examples": [
                chosen(payload["source_documents"][0]["passages"][0], quote),
            ],
        }))


def test_restored_source_text_still_obeys_the_annotation_length_limit():
    document = parse_document("minutes.txt", ("Alice" + " " * 12_000 + "Smith").encode())
    with pytest.raises(CognitionError, match="12000"):
        generate_recipe(document, "Find people", settings(), client=model_client(lambda payload: {
            "instructions": "Extract people.", "examples": [
                chosen(payload["source_documents"][0]["passages"][0], "Alice Smith"),
            ],
        }))


def test_optimization_restores_source_whitespace_without_changing_the_active_recipe():
    document = parse_document("minutes.txt", b"Alice  Smith approved.")
    document.id = "doc1"
    active = verified_recipe([document])
    original = active.model_dump()
    proposal = optimize(document, active, result(document, active), settings(), client=model_client(lambda payload: {
        "instructions": "Extract people.", "examples": [
            chosen({"passage_id": "existing:0"}, "Alice Smith"),
        ], "reasons": ["Show a supported person example."]}))
    assert proposal.candidate.examples[0].extractions[0].extraction_text == "Alice  Smith"
    assert active.model_dump() == original


@pytest.mark.parametrize("selection", [[], ["existing:0"], ["existing:0", "existing:0"], ["target:0"]])
def test_optimization_rejects_omitted_duplicated_or_replacement_exemplars(selection):
    documents = [parse_document("nda.txt", b"Alice signed."), parse_document("arac.txt", b"Bob approved.")]
    documents[0].id, documents[1].id = "nda", "arac"
    active = verified_recipe(documents)
    before = active.model_dump()
    with pytest.raises(CognitionError):
        optimize(documents[0], active, result(documents[0], active), settings(), client=model_client(lambda _: {
            "instructions": "Extract named people.", "examples": [
                chosen({"passage_id": key}) for key in selection
            ], "reasons": ["Clarify supported roles."],
        }))
    assert active.model_dump() == before


def test_optimization_corrects_missing_exemplar_once_instead_of_accepting_partial_replacement():
    documents = [parse_document("nda.txt", b"Alice signed."), parse_document("arac.txt", b"Bob approved.")]
    active = verified_recipe(documents)
    requests = []
    def reply(payload):
        requests.append(payload)
        choices = [chosen({"passage_id": "existing:0"})]
        if len(requests) == 2:
            choices.append(chosen({"passage_id": "existing:1"}, "Bob"))
        return {"instructions": "Extract named people.", "examples": choices,
                "reasons": ["Keep both original exemplar patterns."]}
    proposal = optimize(documents[0], active, result(documents[0], active), settings(), client=model_client(reply))
    assert len(requests) == 2
    assert "validation_feedback" in requests[1]
    assert [example.text for example in proposal.candidate.examples] == [doc.text for doc in documents]


def test_optimization_preserves_source_order_and_metadata_even_if_model_reorders_proposals():
    documents = [parse_document("nda.txt", b"Alice signed."), parse_document("arac.txt", b"Bob approved.")]
    documents[0].id, documents[1].id = "nda", "arac"
    active = verified_recipe(documents)
    def reply(payload):
        schema = payload["response_schema"]
        assert schema["properties"]["examples"]["minItems"] == 2
        assert schema["properties"]["examples"]["maxItems"] == 2
        assert schema["$defs"]["SelectedExample"]["properties"]["passage_ids"]["items"]["enum"] == [
            "existing:0", "existing:1",
        ]
        assert payload["document"]["text"] == documents[0].text
        assert "target:0" not in json.dumps(schema)
        choices = [chosen({"passage_id": "existing:1"}, "Bob"), chosen({"passage_id": "existing:0"})]
        choices[1]["extractions"][0]["attributes"] = {"role": "signer"}
        return {"instructions": "Extract named people.", "examples": choices,
                "reasons": ["Clarify explicit role attributes."]}
    proposal = optimize(documents[0], active, result(documents[0], active), settings(), client=model_client(reply))
    assert [example.text for example in proposal.candidate.examples] == [example.text for example in active.examples]
    assert [example.source for example in proposal.candidate.examples] == [example.source for example in active.examples]
    assert proposal.candidate.examples[0].extractions[0].attributes == {"role": "signer"}
    assert proposal.candidate.source_name == active.source_name
    assert proposal.candidate.source_characters_used == active.source_characters_used
    assert proposal.candidate.source_characters_total == active.source_characters_total


def test_optimization_preserves_five_original_exemplars_without_adding_the_target():
    documents = [parse_document(f"{name}.txt", f"{name} signed.".encode())
                 for name in ["Alice", "Bob", "Carol", "Dave", "Eve"]]
    active = verified_recipe(documents)
    target = parse_document("another.txt", b"Frank signed.")
    def reply(payload):
        assert payload["response_schema"]["properties"]["examples"]["minItems"] == 5
        return {"instructions": "Extract named people.", "examples": [
            chosen({"passage_id": f"existing:{index}"}, doc.text.split()[0])
            for index, doc in enumerate(documents)
        ], "reasons": ["Retain all exemplar patterns."]}
    proposal = optimize(target, active, result(target, active), settings(), client=model_client(reply))
    assert len(proposal.candidate.examples) == 5
    assert [example.text for example in proposal.candidate.examples] == [doc.text for doc in documents]


def test_optimization_preserves_unverified_examples_between_verified_sources():
    documents = [parse_document("nda.txt", b"Alice signed."), parse_document("arac.txt", b"Carol approved.")]
    active = verified_recipe(documents)
    legacy = recipe().examples[0].model_copy(update={"text": "Bob signed.",
        "extractions": [recipe().examples[0].extractions[0].model_copy(update={"extraction_text": "Bob"})]})
    active.examples.insert(1, legacy)
    proposal = optimize(documents[0], active, result(documents[0], active), settings(), client=model_client(lambda _: {
        "instructions": "Extract named people.", "examples": [
            chosen({"passage_id": "existing:2"}, "Carol"), chosen({"passage_id": "existing:0"}),
        ], "reasons": ["Keep the unverified example unchanged."],
    }))
    assert len(proposal.candidate.examples) == 3
    assert proposal.candidate.examples[1].model_dump() == legacy.model_dump()
    assert [example.text for example in proposal.candidate.examples] == [example.text for example in active.examples]


def test_optimization_can_refine_instructions_without_recertifying_legacy_sources():
    target = parse_document("target.txt", b"Bob approved.")
    active = recipe()
    proposal = optimize(target, active, result(target, active), settings(), client=model_client(lambda payload: {
        "instructions": "Extract only explicitly named people.", "examples": [],
        "reasons": ["Keep legacy annotations unchanged; improve instructions."],
    }))
    assert proposal.candidate.instructions == "Extract only explicitly named people."
    assert proposal.candidate.examples == active.examples
    assert proposal.candidate.examples[0].source is None
