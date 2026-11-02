"""Offline tests for the Unstract solution: prompt/project generation, API deployment client, result mapping."""
import json

import httpx
import pytest

from app.azure import ServiceError
from app.documents import DocumentStore, ParsedDocument, parse_document
from app.results import Leaf, finalise
from app.schema import ExtractionSchema, meeting_minutes_schema
from app.solutions import unstract as module
from app.solutions.base import RunContext
from app.solutions.unstract import UnstractSolution, attach_confidence
from app.unstract_export import (decode_json, enforce_type, exemplar_notes_from_records, import_records,
                                 output_to_record, parse_results_file, prompt_list_text, schema_to_project_export,
                                 schema_to_prompts)


API_URL = "http://unstract.test:8000/deployment/api/org_abc/minutes_api/"
FILE = "CSPL_Thursday_17_October_2019_minutes.pdf"
TEXT = ("COMMITTEE ON STANDARDS IN PUBLIC LIFE\nTWO HUNDRED AND SEVENTY-FIRST MEETING\n"
        "HELD AT 10.00 ON THURSDAY 17 OCTOBER 2019\nIN 70 WHITEHALL, LONDON\nMINUTES\n"
        "Present: Lord Evans KCB DL, Chair\nDr Jane Martin CBE\nLesley Bainsfair, Secretary\n"
        "Apologies: Rt Hon Dame Margaret Beckett DBE MP\nMonisha Shah\n1. REGISTERS\n"
        "2 MINUTES AND MATTERS ARISING\nThe minutes of the meeting held on 19 September 2019 were agreed.\n")

# Trimmed from a real Unstract v0.197.0 status response (GET {status_api}, include_metadata/metrics=true).
OUTPUT = {
    "chair": "Lord Evans KCB DL",
    "actions": [],
    "location": "70 WHITEHALL, LONDON",
    "attendees": [
        {"name": "Lord Evans KCB DL", "role": "Chair", "attendance": "Present"},
        {"name": "Dr Jane Martin CBE", "role": None, "attendance": "present"},
        {"name": "Lesley Bainsfair", "role": "Secretary", "attendance": "present"},
        {"name": "Rt Hon Dame Margaret Beckett DBE MP", "role": None, "attendance": "apologies"},
        {"name": "Monisha Shah", "role": "NA", "attendance": "apologies"},
    ],
    "agenda_items": [
        {"title": "REGISTERS", "number": "1.", "decisions": []},
        {"title": "MINUTES AND MATTERS ARISING", "number": "2",
         "decisions": ["The minutes of the meeting held on 19 September 2019 were agreed."]},
    ],
    "meeting_date": "17 October 2019",
    "organisation": "Committee on Standards in Public Life",
    "meeting_title": "TWO HUNDRED AND SEVENTY-FIRST MEETING",
    "next_meeting_date": None,
}


def item(output=None, status="Success", error=None, metadata=None, file=FILE):
    return {
        "file": file, "file_execution_id": "8f3230bd-cad2-48d5-a311-cc2b80503cc8", "status": status,
        "result": None if error else {"output": output if output is not None else OUTPUT,
                                      "metrics": {"chair": {"extraction_llm": {"time_taken(s)": 0.018}}},
                                      **({"metadata": metadata} if metadata is not None else {})},
        "error": error,
        "metadata": {"source_name": file, "workflow_id": "75ce1db7", "execution_id": "d6f339d7",
                     "total_elapsed_time": 13.4,
                     "usage": {"embedding_tokens": 0, "prompt_tokens": 12595, "completion_tokens": 201,
                               "total_tokens": 12796, "cost_in_dollars": 0.0}},
    }


STATUS_DONE = {"status": "COMPLETED", "message": [item(metadata={
    "file_name": FILE, "context": {"chair": [TEXT]}, "required_fields": {"chair": None}})]}
POST_PENDING = {"message": {
    "execution_id": "01aee8e0-a146-4d02-8938-ef5ba257c91d", "execution_status": "PENDING",
    "status_api": "/deployment/api/org_abc/minutes_api/?execution_id=01aee8e0-a146-4d02-8938-ef5ba257c91d",
    "error": None, "result": None}}


@pytest.fixture
def schema():
    return ExtractionSchema(id="minutes", version=2, created_at="2026-10-25T00:00:00Z", **meeting_minutes_schema())


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("UNSTRACT_API_URL", API_URL)
    monkeypatch.setenv("UNSTRACT_API_KEY", "test-key-123")
    monkeypatch.delenv("UNSTRACT_UI_URL", raising=False)
    monkeypatch.delenv("UNSTRACT_ALLOWED_ORIGINS", raising=False)


@pytest.fixture
def store_doc():
    store = DocumentStore()
    content = b"%PDF-1.4 fake bytes"
    document = store.add(ParsedDocument(name=FILE, text=TEXT, pages=1, file_sha256="abc"), content)
    return store, document, content


def mock_solution(monkeypatch, handler, *, sleeps=None):
    solution = UnstractSolution()
    monkeypatch.setattr(solution, "_client", lambda settings: httpx.Client(transport=httpx.MockTransport(handler)))

    def no_sleep(seconds, context, deadline):
        context.check()
        if sleeps is not None:
            sleeps.append(seconds)
    monkeypatch.setattr(UnstractSolution, "_sleep", staticmethod(no_sleep))
    return solution


def form_fields(request: httpx.Request) -> dict:
    body = request.read().decode("latin-1")
    fields = {}
    for part in body.split("--" + request.headers["content-type"].split("boundary=")[1])[1:-1]:
        head, _, value = part.partition("\r\n\r\n")
        name = head.split('name="')[1].split('"')[0]
        filename = head.split('filename="')[1].split('"')[0] if 'filename="' in head else None
        fields[name] = (filename, value.rstrip("\r\n")) if filename else value.rstrip("\r\n")
    return fields


# --- Prompt Studio generation -------------------------------------------------------------------

def test_prompts_one_per_top_level_field_with_enforce_types(schema):
    prompts = schema_to_prompts(schema)
    assert [prompt["prompt_key"] for prompt in prompts] == [field.name for field in schema.fields]
    types = {prompt["prompt_key"]: prompt["enforce_type"] for prompt in prompts}
    assert types == {"meeting_title": "text", "organisation": "text", "meeting_date": "date", "location": "text",
                     "chair": "text", "attendees": "json", "agenda_items": "json", "actions": "json",
                     "next_meeting_date": "date"}
    assert [prompt["sequence_number"] for prompt in prompts] == list(range(1, len(prompts) + 1))
    assert all(prompt["prompt_type"] == "PROMPT" and prompt["active"] for prompt in prompts)


def test_object_array_prompt_describes_exact_item_structure(schema):
    attendees = next(prompt for prompt in schema_to_prompts(schema) if prompt["prompt_key"] == "attendees")["prompt"]
    assert "JSON array with one object per item" in attendees
    for key in ('"name"', '"role"', '"attendance"', '"present", "in attendance", "apologies", "absent"'):
        assert key in attendees
    assert "Every person listed as present" in attendees  # field description
    shape = json.loads(attendees.split("Shape: ", 1)[1])
    assert set(shape[0]) == {"name", "role", "attendance"}
    agenda = next(prompt for prompt in schema_to_prompts(schema) if prompt["prompt_key"] == "agenda_items")["prompt"]
    assert json.loads(agenda.split("Shape: ", 1)[1]) == [{"number": "<exact text>", "title": "<exact text>",
                                                           "decisions": ["<text>"]}]


def test_scalar_prompts_use_unstract_na_convention(schema):
    prompts = {prompt["prompt_key"]: prompt["prompt"] for prompt in schema_to_prompts(schema)}
    assert "answer NA" in prompts["meeting_date"] and "YYYY-MM-DD" in prompts["meeting_date"]
    assert "exactly as it appears" in prompts["meeting_title"]


def test_exemplar_notes_per_field_and_global(schema):
    notes = exemplar_notes_from_records(schema, [{"chair": "Jane Doe", "attendees": [{"name": "A"}] * 5}, {}])
    assert notes["chair"].startswith("Example answers") and "Jane Doe" in notes["chair"]
    assert notes["attendees"].count('"name"') == 3  # at most three items per example
    prompts = {prompt["prompt_key"]: prompt["prompt"] for prompt in schema_to_prompts(schema, notes)}
    assert "Guidance: Example answers" in prompts["chair"] and "Guidance" not in prompts["location"]
    everywhere = schema_to_prompts(schema, "Ignore the cover page.")
    assert all(prompt["prompt"].endswith("Guidance: Ignore the cover page.") for prompt in everywhere)


def test_project_export_matches_prompt_studio_import_format(schema):
    project = schema_to_project_export(schema, guidance="UK public bodies.")
    # Keys validate_import_file / create_tool_from_import_data / import_prompts read.
    assert {"tool_metadata", "tool_settings", "prompts"} <= set(project)
    assert {"tool_name", "description", "author"} <= set(project["tool_metadata"])
    assert project["tool_metadata"]["tool_name"] == "Meeting minutes"
    assert project["tool_settings"]["preamble"].endswith("UK public bodies.")
    assert project["tool_settings"]["single_pass_extraction_mode"] is False
    assert project["default_profile_settings"]["chunk_size"] == 0
    assert project["export_metadata"]["schema_id"] == "minutes"
    assert all({"prompt_key", "prompt"} <= set(prompt) for prompt in project["prompts"])
    json.dumps(project)  # serialisable as the .json file Prompt Studio imports
    blank = schema.model_copy(update={"description": ""})
    assert schema_to_project_export(blank)["tool_metadata"]["description"]  # description is NOT NULL in Unstract


def test_prompt_list_text(schema):
    text = prompt_list_text(schema_to_prompts(schema))
    assert "### attendees  (enforce type: json)" in text


def test_enforce_type_for_scalar_arrays():
    from app.schema import SchemaField
    assert enforce_type(SchemaField(name="tags", type="text", array=True)) == "json"
    assert enforce_type(SchemaField(name="count", type="integer")) == "number"
    assert enforce_type(SchemaField(name="flag", type="boolean")) == "boolean"


# --- Output mapping ------------------------------------------------------------------------------

def test_real_output_maps_to_schema_and_cleans_na(schema):
    record, notes = output_to_record(OUTPUT, schema)
    assert notes == []
    assert record["attendees"][4] == {"name": "Monisha Shah", "role": None, "attendance": "apologies"}
    assert record["agenda_items"][1]["decisions"] == ["The minutes of the meeting held on 19 September 2019 were agreed."]
    data, values = finalise(TEXT, schema, __import__("app.results", fromlist=["annotate_plain"]).annotate_plain(
        record, schema.fields))
    assert data["meeting_date"] == "2019-10-17"
    assert data["attendees"][0]["attendance"] == "present"
    assert any(value.field == "attendees[].name" and value.grounding == "exact" for value in values)


def test_key_spelling_and_string_encoded_json(schema):
    output = {
        "Meeting Title": "MINUTES",
        "meeting-date": "NA",
        "ATTENDEES": '§§§\n[{"Name": "Jane Ramsey", "Attendance": "present"}]\n§§§',
        "agenda_items": '```json\n[{"number": "1", "title": "REGISTERS", "decisions": "[\\"Agreed\\"]"}]\n```',
        "actions": '{"actions": [{"description": "Send papers", "owner": "Secretariat", "due_date": "N/A"}]}',
        "extra_prompt": "something",
    }
    record, notes = output_to_record(output, schema)
    assert record["meeting_title"] == "MINUTES"
    assert record["meeting_date"] is None
    assert record["attendees"] == [{"name": "Jane Ramsey", "role": None, "attendance": "present"}]
    assert record["agenda_items"] == [{"number": "1", "title": "REGISTERS", "decisions": ["Agreed"]}]
    assert record["actions"] == [{"description": "Send papers", "owner": "Secretariat", "due_date": None}]
    assert any("extra_prompt" in note for note in notes)
    assert any("no prompt for" in note for note in notes)


def test_single_json_blob_prompt_holds_whole_record(schema):
    blob = json.dumps({"meeting_title": "Board", "Chair": "A. Person",
                       "attendees": [{"name": "A. Person", "role": "Chair", "attendance": "present"}]})
    record, notes = output_to_record({"minutes": blob}, schema)
    assert record["meeting_title"] == "Board" and record["chair"] == "A. Person"
    assert record["attendees"][0]["role"] == "Chair"
    assert any("JSON object in Unstract prompt 'minutes'" in note for note in notes)
    assert not any("Ignored" in note for note in notes)


def test_wrappers_and_shape_mismatches(schema):
    record, notes = output_to_record({
        "attendees": {"items": [{"name": "X"}]},  # list wrapped in an object
        "chair": ["Lord Evans", "Someone else"],   # several values for a scalar
        "location": {"location": "London"},        # scalar wrapped in an object
    }, schema)
    assert record["attendees"] == [{"name": "X", "role": None, "attendance": None}]
    assert record["chair"] == "Lord Evans" and record["location"] == "London"
    assert any("kept the first" in note for note in notes)


def test_unusable_output(schema):
    record, notes = output_to_record("not json", schema)
    assert all(value is None for value in record.values())
    assert "not a JSON object" in notes[0]
    assert decode_json('{"a": 1}') == {"a": 1} and decode_json("[1") == "[1"


# --- Results files -------------------------------------------------------------------------------

DOCS = [ParsedDocument(name=FILE, text=TEXT, id="a" * 24), ParsedDocument(name="Other minutes.pdf", text="x", id="b" * 24)]


@pytest.mark.parametrize("payload", [
    STATUS_DONE,                                                     # GET status response
    {"message": {"execution_status": "COMPLETED", "result": [item()]}},  # synchronous POST response
    [item()],                                                        # the result list
    item(),                                                          # one per-file item
    {"output": OUTPUT, "metadata": {"file_name": FILE}},             # per-file output JSON
    {FILE: OUTPUT},                                                  # {file name: output}
    json.dumps(STATUS_DONE),                                         # string-encoded
])
def test_parse_results_file_shapes(payload):
    parsed = parse_results_file(payload, DOCS)
    assert list(parsed) == [FILE]
    assert parsed[FILE]["chair"] == "Lord Evans KCB DL"


def test_parse_results_file_matching_and_errors():
    plain = {"chair": "X"}
    assert parse_results_file(plain, DOCS) == {}  # ambiguous without a file name
    assert parse_results_file(plain, DOCS, file_name="cspl_thursday_17_october_2019_minutes.json") == {FILE: plain}
    assert parse_results_file(plain, DOCS[:1]) == {FILE: plain}  # single document
    failed = item(status="Failed", error="LLM timeout")
    assert parse_results_file([failed, item(file="unknown.pdf")], DOCS) == {FILE: {"error": "LLM timeout"}}
    renamed = item(file="uploads/CSPL Thursday 17 October 2019 minutes.PDF")
    assert list(parse_results_file([renamed], DOCS)) == [FILE]


def test_import_records_for_run_import(schema):
    records, warnings = import_records([item(), item(file="nope.pdf"),
                                        item(file="Other minutes.pdf", status="Failed", error="boom")], DOCS, schema)
    assert warnings == ["No document matches Unstract result for 'nope.pdf'."]
    assert records[0]["document_id"] == "a" * 24 and records[0]["data"]["attendees"][4]["role"] is None
    assert records[0]["raw"]["notes"] == []
    assert records[1] == {"document_id": "b" * 24, "data": None, "error": "boom", "raw": records[1]["raw"]}
    from app.results import ImportedRecord
    for record in records:
        ImportedRecord.model_validate(record)


# --- API deployment client ----------------------------------------------------------------------

def test_async_upload_then_poll(monkeypatch, env, schema, store_doc):
    store, document, content = store_doc
    calls, sleeps = [], []

    def handler(request: httpx.Request):
        calls.append(request)
        assert request.headers["authorization"] == "Bearer test-key-123"
        if request.method == "POST":
            return httpx.Response(200, json=POST_PENDING)
        if len([call for call in calls if call.method == "GET"]) < 3:
            return httpx.Response(422, json={"status": "EXECUTING", "message": None})
        return httpx.Response(200, json=STATUS_DONE)

    solution = mock_solution(monkeypatch, handler, sleeps=sleeps)
    settings = solution.validate_settings({"llm_profile_id": "prof-1", "poll_interval": 2})
    outcome = solution.extract(document, schema, settings, RunContext(store=store))

    post = calls[0]
    assert str(post.url) == API_URL
    fields = form_fields(post)
    assert fields["files"] == (FILE, content.decode("latin-1"))
    assert fields["timeout"] == "-1" and fields["include_metadata"] == "false" and fields["include_metrics"] == "true"
    assert fields["llm_profile_id"] == "prof-1"
    gets = [call for call in calls if call.method == "GET"]
    assert len(gets) == 3 and sleeps == [2.0, 2.0, 2.0]
    assert gets[0].url.params["execution_id"] == "01aee8e0-a146-4d02-8938-ef5ba257c91d"
    assert gets[0].url.params["include_metrics"] == "true"
    assert gets[0].url.host == "unstract.test" and gets[0].url.path == "/deployment/api/org_abc/minutes_api/"

    assert isinstance(outcome.record["chair"], Leaf) and outcome.record["chair"].value == "Lord Evans KCB DL"
    assert outcome.record["attendees"][4]["role"].value is None
    assert outcome.usage.prompt_tokens == 12595 and outcome.usage.completion_tokens == 201
    assert outcome.raw["execution_id"] == "01aee8e0-a146-4d02-8938-ef5ba257c91d"
    assert outcome.raw["item"]["result"]["metadata"]["context"] == {"chair": f"<{len(TEXT)} characters omitted>"}
    assert "test-key-123" not in json.dumps(outcome.raw)
    data, _ = finalise(document.text, schema, outcome.record)
    assert data["meeting_date"] == "2019-10-17"


def test_synchronous_mode_returns_inline(monkeypatch, env, schema, store_doc):
    store, document, _ = store_doc
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"message": {**POST_PENDING["message"], "execution_status": "COMPLETED",
                                                     "result": [item()]}})

    solution = mock_solution(monkeypatch, handler)
    outcome = solution.extract(document, schema, solution.validate_settings({"server_wait": 120}),
                               RunContext(store=store))
    assert len(calls) == 1 and form_fields(calls[0])["timeout"] == "120"
    assert outcome.record["organisation"].value == "Committee on Standards in Public Life"


@pytest.mark.parametrize("post_status, body, expected", [
    (401, {"type": "client_error", "errors": [{"code": "error", "detail": "Unauthorized", "attr": None}]},
     "rejected the API key"),
    (404, {"type": "client_error", "errors": [{"detail": "API not found or Inactive"}]}, "not found"),
    (429, {"message": {"error": "Rate limit exceeded"}}, "rate limit"),
    (422, {"message": {"execution_status": "ERROR", "error": "Tool not found"}}, "Tool not found"),
])
def test_upload_errors(monkeypatch, env, schema, store_doc, post_status, body, expected):
    store, document, _ = store_doc
    solution = mock_solution(monkeypatch, lambda request: httpx.Response(post_status, json=body))
    with pytest.raises(ServiceError, match=f"(?i){expected}"):
        solution.extract(document, schema, solution.validate_settings({}), RunContext(store=store))


@pytest.mark.parametrize("status_response, expected", [
    (httpx.Response(422, json={"status": "ERROR", "message": [{"file": FILE, "error": "LLM failed"}]}), "LLM failed"),
    (httpx.Response(406, json={"status": "COMPLETED", "message": "Result already acknowledged"}), "served once"),
    (httpx.Response(200, json={"status": "COMPLETED", "message": [item(status="Failed", error="bad PDF")]}),
     "bad PDF"),
    (httpx.Response(422, json={"status": "STOPPED", "message": None}), "stopped"),
])
def test_poll_errors(monkeypatch, env, schema, store_doc, status_response, expected):
    store, document, _ = store_doc

    def handler(request):
        return httpx.Response(200, json=POST_PENDING) if request.method == "POST" else status_response

    solution = mock_solution(monkeypatch, handler)
    with pytest.raises(ServiceError, match=f"(?i){expected}"):
        solution.extract(document, schema, solution.validate_settings({}), RunContext(store=store))


def test_poll_gives_up_after_max_wait(monkeypatch, env, schema, store_doc):
    store, document, _ = store_doc
    clock = [0.0]

    class FakeTime:
        @staticmethod
        def monotonic():
            return clock[0]

    def handler(request):
        if request.method == "POST":
            return httpx.Response(200, json=POST_PENDING)
        clock[0] += 20
        return httpx.Response(422, json={"status": "EXECUTING", "message": None})

    solution = mock_solution(monkeypatch, handler)
    monkeypatch.setattr(module, "time", FakeTime)
    with pytest.raises(ServiceError, match="did not finish within 60") as error:
        solution.extract(document, schema, solution.validate_settings({"max_wait": 60}), RunContext(store=store))
    assert error.value.status == 504


def test_cancel_while_polling(monkeypatch, env, schema, store_doc):
    store, document, _ = store_doc
    context = RunContext(store=store)

    def handler(request):
        if request.method == "POST":
            return httpx.Response(200, json=POST_PENDING)
        context.cancel.set()
        return httpx.Response(422, json={"status": "PENDING", "message": None})

    solution = mock_solution(monkeypatch, handler)
    with pytest.raises(ServiceError, match="cancelled"):
        solution.extract(document, schema, solution.validate_settings({}), context)


def test_confidence_from_metadata(monkeypatch, env, schema, store_doc):
    store, document, _ = store_doc
    metadata = {"confidence_data": {"chair": 0.91, "attendees": [{"name": 0.8, "role": 0.4, "attendance": 0.7}] * 5},
                "highlight_data": {"location": [[1, 10, 20, 30, 0.6], [1, 40, 20, 30, 0.8]]}}
    response = {"status": "COMPLETED", "message": [item(metadata=metadata)]}

    def handler(request):
        return httpx.Response(200, json=POST_PENDING) if request.method == "POST" else httpx.Response(200, json=response)

    solution = mock_solution(monkeypatch, handler)
    outcome = solution.extract(document, schema, solution.validate_settings({"include_metadata": True}),
                               RunContext(store=store))
    assert outcome.record["chair"].confidence == 0.91
    assert outcome.record["attendees"][0]["role"].confidence == 0.4
    assert outcome.record["attendees"][4]["role"].confidence is None  # null value keeps no score
    assert outcome.record["location"].confidence == pytest.approx(0.7)
    assert outcome.record["organisation"].confidence is None


def test_attach_confidence_average_for_mismatched_structure():
    record = {"a": Leaf("x"), "b": [Leaf("y")]}
    attach_confidence(record, {"unrelated": [0.2, 0.4]})
    assert record["a"].confidence == pytest.approx(0.3) and record["b"][0].confidence == pytest.approx(0.3)


def test_text_fallback_when_source_missing(monkeypatch, env, schema):
    store = DocumentStore()
    document = store.add(parse_document("notes.txt", b"Chair: Jane"))  # no stored source bytes
    seen = []

    def handler(request):
        seen.append(form_fields(request)["files"])
        return httpx.Response(200, json={"message": {**POST_PENDING["message"], "execution_status": "COMPLETED",
                                                     "result": [item(output={"chair": "Jane"}, file="notes.txt")]}})

    solution = mock_solution(monkeypatch, handler)
    outcome = solution.extract(document, schema, solution.validate_settings({}), RunContext(store=store))
    assert seen == [("notes.txt", "Chair: Jane")]
    assert "Sent the stored text" in outcome.notes[0]
    pdf = store.add(ParsedDocument(name="scan.pdf", text="t", file_sha256="z"))
    with pytest.raises(ServiceError, match="not stored") as error:
        solution.extract(pdf, schema, solution.validate_settings({}), RunContext(store=store))
    assert error.value.status == 410


# --- Settings, status, preview -------------------------------------------------------------------

def test_settings_only_allow_configured_unstract_host(env, monkeypatch):
    solution = UnstractSolution()
    assert solution.validate_settings({})["api_url"] == API_URL
    other = "http://unstract.test:8000/deployment/api/org_abc/other_api/"
    assert solution.validate_settings({"api_url": other})["api_url"] == other
    with pytest.raises(ServiceError, match="configured"):
        solution.validate_settings({"api_url": "https://attacker.example/deployment/api/x/y/"})
    monkeypatch.setenv("UNSTRACT_ALLOWED_ORIGINS", "https://attacker.example")
    assert solution.validate_settings({"api_url": "https://attacker.example/x/"})
    with pytest.raises(ServiceError, match="Unknown setting"):
        solution.validate_settings({"api_key": "nope"})


def test_status_requires_env(monkeypatch):
    monkeypatch.delenv("UNSTRACT_API_URL", raising=False)
    monkeypatch.delenv("UNSTRACT_API_KEY", raising=False)
    status = UnstractSolution().status()
    assert status["ready"] is False and "UNSTRACT_API_URL and UNSTRACT_API_KEY" in status["detail"]


@pytest.mark.parametrize("probe, ready, detail", [
    (httpx.Response(400, json={"type": "validation_error", "errors": [{"attr": "execution_id"}]}), True, "reachable"),
    (httpx.Response(401, json={"type": "client_error"}), False, "rejected"),
    (httpx.Response(404, json={}), False, "no active API deployment"),
    (httpx.ConnectError("refused"), False, "not reachable"),
])
def test_status_probe(monkeypatch, env, probe, ready, detail):
    calls = []

    def fake_get(url, headers, timeout):
        calls.append((url, headers["Authorization"]))
        if isinstance(probe, Exception):
            raise probe
        return probe

    monkeypatch.setattr(module.httpx, "get", fake_get)
    solution = UnstractSolution()
    status = solution.status()
    assert status["ready"] is ready and detail in status["detail"] and status["model"] == "minutes_api"
    solution.status()
    assert len(calls) == 1  # cached
    assert "test-key-123" not in json.dumps(status)


def test_status_never_raises(monkeypatch, env):
    monkeypatch.setattr(module.httpx, "get", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    status = UnstractSolution().status()
    assert status["ready"] is False and "boom" in status["detail"]


def test_info_preview_and_registry(env, schema):
    solution = UnstractSolution()
    info = solution.info()
    assert info["id"] == "unstract" and info["route"] == "/apps/unstract" and info["runtime"] == "local"
    assert info["external_url"] == "http://frontend.unstract.localhost"
    assert {spec["key"] for spec in info["settings"]} >= {"api_url", "server_wait", "include_metadata"}
    assert "api_key" not in {spec["key"] for spec in info["settings"]}
    preview = solution.preview(schema, solution.resolve_settings({"prompt_guidance": "Be strict."}))
    project_key = next(key for key in preview if key.startswith("Prompt Studio project"))
    project = json.loads(preview[project_key])
    assert project["prompts"][0]["prompt_key"] == "meeting_title"
    assert "Be strict." in project["tool_settings"]["preamble"]
    assert solution.model_label({"api_url": API_URL}) == "Unstract · minutes_api"
