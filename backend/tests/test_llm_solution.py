"""Offline tests for the direct-LLM solution (chat_json and the client are faked)."""
import json
import threading

import pytest

from app.azure import ServiceError
from app.documents import DocumentStore, ParsedDocument
from app.grounding import Grounder
from app.results import Leaf, Usage, finalise, plain
from app.schema import ExtractionSchema, meeting_minutes_schema
from app.settings import Settings
from app.solutions import llm
from app.solutions.base import RunContext
from app.solutions.llm import (
    LLMSolution, build_messages, check_quotes, parse_confidence, response_format, response_schema,
    split_text, tidy_string, to_record,
)


AZURE = Settings(endpoint="https://example.openai.azure.com", deployment="test-deployment", api_key="key")


def make_schema(data: dict | None = None) -> ExtractionSchema:
    return ExtractionSchema(id="s", version=1, created_at="2026-01-01", **(data or meeting_minutes_schema()))


NESTED = {
    "name": "Contracts",
    "description": "Agreements with parties and obligations.",
    "fields": [
        {"name": "title", "type": "verbatim"},
        {"name": "value", "type": "number"},
        {"name": "term_months", "type": "integer"},
        {"name": "signed", "type": "boolean"},
        {"name": "tags", "type": "text", "array": True},
        {"name": "parties", "type": "object", "array": True, "fields": [
            {"name": "name", "type": "verbatim"},
            {"name": "kind", "type": "enum", "options": ["buyer", "seller"]},
            {"name": "obligations", "type": "object", "array": True, "fields": [
                {"name": "duty", "type": "text"},
                {"name": "deadline", "type": "date"},
                {"name": "channels", "type": "enum", "array": True, "options": ["email", "post"]},
            ]},
        ]},
        {"name": "address", "type": "object", "fields": [
            {"name": "city", "type": "text"}, {"name": "postcode", "type": "verbatim"}]},
    ],
}

ALLOWED_KEYS = {"type", "properties", "required", "additionalProperties", "items", "enum"}


def assert_strict(node, depth=1, counter=None):
    """Structured-outputs strict-mode shape: closed objects, all keys required, nullable leaves."""
    counter = counter if counter is not None else {"properties": 0, "enum": 0, "depth": 0}
    counter["depth"] = max(counter["depth"], depth)
    assert set(node) <= ALLOWED_KEYS, set(node) - ALLOWED_KEYS
    kind = node["type"]
    if kind == "object":
        assert node["additionalProperties"] is False
        assert node["required"] == list(node["properties"])
        counter["properties"] += len(node["properties"])
        for child in node["properties"].values():
            assert_strict(child, depth + 1, counter)
    elif kind == "array":
        assert_strict(node["items"], depth + 1, counter)
    else:
        assert isinstance(kind, list) and kind[-1] == "null" and kind[0] in ("string", "number", "integer", "boolean")
        if "enum" in node:
            assert node["enum"][-1] is None and kind[0] == "string"
            counter["enum"] += len(node["enum"])
    return counter


def leaf_schema_at(schema, *path):
    node = schema
    for key in path:
        node = node["items"] if key == "[]" else node["properties"][key]
    return node


# -- JSON schema construction --------------------------------------------------------------

def test_response_schema_is_strict_for_meeting_minutes():
    schema = response_schema(make_schema())
    counter = assert_strict(schema)
    assert counter["depth"] <= 10 and counter["properties"] < 5_000
    assert list(schema["properties"]) == [field["name"] for field in meeting_minutes_schema()["fields"]]
    leaf = leaf_schema_at(schema, "attendees", "[]", "attendance")
    assert list(leaf["properties"]) == ["quote", "value", "confidence"]
    assert leaf["properties"]["value"]["enum"] == ["present", "in attendance", "apologies", "absent", None]
    decisions = leaf_schema_at(schema, "agenda_items", "[]", "decisions")
    assert decisions["type"] == "array" and decisions["items"]["properties"]["value"]["type"] == ["string", "null"]


def test_response_schema_nested_arrays_and_value_types():
    schema = response_schema(make_schema(NESTED))
    assert_strict(schema)
    value_type = lambda *path: leaf_schema_at(schema, *path)["properties"]["value"]["type"]  # noqa: E731
    assert value_type("value") == ["number", "null"]
    assert value_type("term_months") == ["integer", "null"]
    assert value_type("signed") == ["boolean", "null"]
    assert value_type("tags", "[]") == ["string", "null"]
    assert value_type("parties", "[]", "obligations", "[]", "deadline") == ["string", "null"]
    channels = leaf_schema_at(schema, "parties", "[]", "obligations", "[]", "channels")
    assert channels["type"] == "array"
    assert channels["items"]["properties"]["value"]["enum"] == ["email", "post", None]
    address = leaf_schema_at(schema, "address")
    assert address["type"] == "object" and list(address["properties"]) == ["city", "postcode"]


def test_response_format_wrapper():
    wrapper = response_format(make_schema())
    assert wrapper["type"] == "json_schema"
    assert wrapper["json_schema"]["strict"] is True and wrapper["json_schema"]["name"] == "extraction"
    json.dumps(wrapper)  # serialisable


# -- prompt ---------------------------------------------------------------------------------

def test_messages_contain_rules_guide_document_and_extras():
    schema = make_schema()
    messages = build_messages(schema, "DOC TEXT", extra_instructions="Prefer full names.", part_note="PART")
    system, user = messages[0]["content"], messages[1]["content"]
    assert "untrusted" in system and "Prefer full names." in system and "Confidence" in system
    assert "attendees[].attendance" in user and '"present"' in user
    assert user.index("PART") < user.index("<document>\nDOC TEXT\n</document>")
    review = build_messages(schema, "DOC", draft={"chair": {"value": "X"}})
    assert "reviewer" in review[0]["content"] and '"chair"' in review[1]["content"]


# -- response -> record ---------------------------------------------------------------------

def test_to_record_maps_and_coerces_leaves():
    schema = make_schema(NESTED)
    data = {
        "title": {"quote": "Master\nAgreement", "value": "Master\nAgreement", "confidence": 0.97},
        "value": {"quote": "£1,200.50", "value": "£1,200.50", "confidence": 90},
        "term_months": {"quote": "24 months", "value": 24.0, "confidence": 1.4},
        "signed": {"quote": "signed", "value": "yes", "confidence": "0.8"},
        "tags": [{"quote": "a", "value": "alpha", "confidence": 0.9},
                 {"quote": None, "value": None, "confidence": 0.9}, "bare"],
        "parties": [
            {"name": {"quote": "Acme Ltd", "value": " Acme Ltd ", "confidence": 0.99},
             "kind": {"quote": "Buyer", "value": "Buyer", "confidence": 0.9},
             "obligations": [{"duty": {"quote": "pay", "value": "Pay invoices", "confidence": 0.8},
                              "deadline": {"quote": "1st March 2026", "value": "1st March 2026", "confidence": 0.7},
                              "channels": [{"quote": "email", "value": "EMAIL", "confidence": 0.6}]},
                             {"duty": {"quote": None, "value": None, "confidence": None},
                              "deadline": {"quote": None, "value": "", "confidence": None},
                              "channels": []}]},
            {"name": {"quote": None, "value": None, "confidence": 0.9}, "kind": None, "obligations": None},
        ],
        "address": {"city": {"quote": "Leeds", "value": "Leeds", "confidence": 0.9}},
        "unknown_key": {"value": "ignored"},
    }
    record = to_record(data, schema.fields)
    assert record["title"] == Leaf("Master Agreement", "Master\nAgreement", 0.97)
    assert record["value"].value == 1200.5 and record["value"].confidence == 0.9
    assert record["term_months"].value == 24 and record["term_months"].confidence == 1.0
    assert record["signed"].value is True and record["signed"].confidence == 0.8
    assert [leaf.value for leaf in record["tags"]] == ["alpha", "bare"]
    assert record["tags"][1].quote is None and record["tags"][1].confidence is None
    assert len(record["parties"]) == 1  # the all-null party is dropped
    party = record["parties"][0]
    assert party["name"].value == "Acme Ltd" and party["kind"].value == "buyer"
    assert len(party["obligations"]) == 1  # empty obligation dropped
    obligation = party["obligations"][0]
    assert obligation["deadline"].value == "2026-03-01"
    assert obligation["channels"][0].value == "email"
    assert record["address"]["city"].value == "Leeds" and record["address"]["postcode"] == Leaf(None)
    assert "unknown_key" not in record


def test_to_record_tolerates_garbage():
    schema = make_schema()
    record = to_record(["not", "an", "object"], schema.fields)
    assert record["attendees"] == [] and record["chair"] == Leaf(None)
    record = to_record({"attendees": {"name": {"value": "Solo"}}, "chair": "Bare Name"}, schema.fields)
    assert record["attendees"][0]["name"].value == "Solo"
    assert record["chair"] == Leaf("Bare Name")


@pytest.mark.parametrize("raw, expected", [
    (0.5, 0.5), (1, 1.0), (1.3, 1.0), (-0.2, 0.0), (95, 0.95), ("80%", 0.8), (500, 1.0),
    (None, None), (True, None), ("high", None), (float("nan"), None),
])
def test_parse_confidence(raw, expected):
    assert parse_confidence(raw) == expected


def test_tidy_string_joins_lines_but_keeps_paragraphs():
    assert tidy_string("Room 6.1,\nTwo Snowhill") == "Room 6.1, Two Snowhill"
    assert tidy_string("first\r\n  second\n\nnext para \n") == "first second\n\nnext para"


# -- quote checks ---------------------------------------------------------------------------

def test_check_quotes_repairs_layout_and_penalises_unsupported():
    text = "The Board agreed the minutes\nof the meeting. Chair: Jane Smith. Status: done."
    schema = make_schema({"name": "t", "fields": [
        {"name": "decision", "type": "text"}, {"name": "chair", "type": "verbatim"},
        {"name": "status", "type": "enum", "options": ["done", "open"]},
        {"name": "invented", "type": "text"}, {"name": "close", "type": "verbatim"}]})
    record = {
        "decision": Leaf("Minutes agreed", "The Board agreed the minutes of the meeting.", 0.9),
        "chair": Leaf("Jane Smith", "Jane Smith", 0.99),
        "status": Leaf("open", "Status: pending", 0.8),           # quote not in text, enum not groundable
        "invented": Leaf("Budget approved", "budget was approved", 0.9),
        "close": Leaf("Jane  Smyth", "Chair: Jane Smyth", 0.9),  # fuzzy quote, verbatim mismatch
    }
    check = check_quotes(record, schema.fields, Grounder(text))
    assert record["decision"].quote == "The Board agreed the minutes\nof the meeting."
    assert check.repaired == 1 and check.exact == 1 and check.fuzzy == 1 and check.missing == 2
    assert check.unsupported == 2
    assert record["status"].confidence == 0.4 and record["invented"].confidence == 0.45
    assert record["chair"].confidence == 0.99 and record["close"].confidence == 0.9
    assert check.verbatim_total == 2 and check.verbatim_mismatch == 1
    notes = " ".join(check.notes())
    assert "2 of 5 quotes were not found" in notes and "multiplied by 0.5" in notes


# -- chunking -------------------------------------------------------------------------------

def test_split_text_short_document_is_one_chunk():
    assert [(chunk.start, chunk.end) for chunk in split_text("short", 100)] == [(0, 5)]


def test_split_text_paragraph_boundaries_overlap_and_coverage():
    paragraphs = [f"Paragraph {index} " + "word " * (20 + index % 7) for index in range(60)]
    text = "\n\n".join(paragraphs)
    chunks = split_text(text, 1_000, overlap=150)
    assert len(chunks) > 3
    assert chunks[0].start == 0 and chunks[-1].end == len(text)
    for previous, chunk in zip(chunks, chunks[1:]):
        assert chunk.start < previous.end  # overlapping
        assert previous.end - chunk.start <= 150
        assert chunk.start > previous.start
        assert text[chunk.start - 2:chunk.start] == "\n\n" or text[chunk.start - 1] == "\n"
    for chunk in chunks:
        assert len(chunk.text) <= 1_000 and chunk.text == text[chunk.start:chunk.end]
        if chunk.end < len(text):
            assert text[chunk.end - 2:chunk.end] == "\n\n"


def test_split_text_without_line_breaks_still_progresses():
    text = "x" * 5_000
    chunks = split_text(text, 1_000)
    assert chunks[-1].end == len(text)
    assert all(len(chunk.text) <= 1_000 for chunk in chunks)
    assert all(b.start > a.start for a, b in zip(chunks, chunks[1:]))


# -- extract() with a fake chat_json -------------------------------------------------------

class FakeClient:
    closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def fake(monkeypatch):
    state = {"calls": [], "responses": [], "client": FakeClient()}

    def chat_json(settings, messages, *, response_format, operation, reasoning_effort=None,
                  max_completion_tokens=0, client=None, timeout=0):
        state["calls"].append({"messages": messages, "operation": operation, "effort": reasoning_effort,
                               "format": response_format, "client": client})
        response = state["responses"].pop(0)
        if isinstance(response, Exception):
            raise response
        if callable(response):
            response = response(messages)
        return response, Usage(calls=1, prompt_tokens=100, completion_tokens=50), 0.5

    monkeypatch.setattr(llm, "chat_json", chat_json)
    monkeypatch.setattr(llm, "make_client", lambda settings, timeout=0: state["client"])
    return state


def context():
    return RunContext(store=DocumentStore())


MINUTES = ("Board meeting minutes\nMeeting date 30 October 2019\nPresent: Alice Jones (Chair), Bob Lee\n"
           "Apologies: Carol King\n1 Welcome\nThe Board agreed the minutes.\nAction: Bob Lee to circulate the paper.")


def minutes_response(**overrides):
    leaf = lambda value, quote=None, confidence=0.95: {"quote": quote or value, "value": value, "confidence": confidence}  # noqa: E731
    response = {
        "meeting_title": leaf("Board meeting minutes"), "organisation": leaf(None, None, 0.9),
        "meeting_date": leaf("2019-10-30", "30 October 2019", 0.99), "location": leaf(None, None, 0.9),
        "chair": leaf("Alice Jones"),
        "attendees": [
            {"name": leaf("Alice Jones"), "role": leaf("Chair"), "attendance": leaf("present", "Present:")},
            {"name": leaf("Bob Lee"), "role": leaf(None, None, 0.9), "attendance": leaf("present", "Present:")},
            {"name": leaf("Carol King"), "role": leaf(None, None, 0.9), "attendance": leaf("apologies", "Apologies:")},
        ],
        "agenda_items": [{"number": leaf("1"), "title": leaf("Welcome"),
                          "decisions": [leaf("Minutes agreed", "The Board agreed the minutes.", 0.9)]}],
        "actions": [{"description": leaf("Circulate the paper", "to circulate the paper", 0.9),
                     "owner": leaf("Bob Lee"), "due_date": leaf(None, None, 0.8)}],
        "next_meeting_date": leaf(None, None, 0.8),
    }
    response.update(overrides)
    return response


def test_extract_single_call_end_to_end(fake):
    fake["responses"] = [minutes_response()]
    document = ParsedDocument(name="m.txt", text=MINUTES)
    schema = make_schema()
    outcome = LLMSolution(AZURE).extract(document, schema, {"reasoning_effort": "medium"}, context())
    assert len(fake["calls"]) == 1
    call = fake["calls"][0]
    assert call["effort"] == "medium" and call["client"] is fake["client"] and fake["client"].closed
    assert call["format"]["json_schema"]["strict"] is True
    assert "<document>\n" + MINUTES + "\n</document>" in call["messages"][1]["content"]
    assert "processed in parts" not in call["messages"][1]["content"]
    assert outcome.usage == Usage(calls=1, prompt_tokens=100, completion_tokens=50)
    assert outcome.raw["calls"][0]["response"]["chair"]["value"] == "Alice Jones"
    assert outcome.raw["quote_check"]["missing"] == 0
    assert outcome.notes == []
    record = outcome.record
    assert record["meeting_date"] == Leaf("2019-10-30", "30 October 2019", 0.99)
    assert [person["attendance"].value for person in record["attendees"]] == ["present", "present", "apologies"]
    data, values = finalise(document.text, schema, record)
    assert data["chair"] == "Alice Jones" and data["actions"][0]["owner"] == "Bob Lee"
    assert all(value.grounding != "none" for value in values)
    date = next(value for value in values if value.field == "meeting_date")
    assert MINUTES[date.start:date.end] == "30 October 2019"


def test_extract_reports_missing_quotes_and_penalises(fake):
    fake["responses"] = [minutes_response(
        organisation={"quote": "Northwind Trust", "value": "Northwind Trust", "confidence": 0.9})]
    outcome = LLMSolution(AZURE).extract(ParsedDocument(name="m.txt", text=MINUTES), make_schema(), {}, context())
    assert outcome.record["organisation"].confidence == 0.45
    assert any("not found in the source" in note for note in outcome.notes)
    assert outcome.raw["calls"][0]["response"]["organisation"]["confidence"] == 0.9  # raw keeps model value


def test_extract_chunks_long_documents_and_merges(fake):
    first = "Meeting title: Alpha review\n\n" + "Filler paragraph about context.\n\n" * 40
    second = "Present: Dana Fox\n\n" + "More filler text for the later part.\n\n" * 40
    text = first + second
    leaf = lambda value, confidence=0.9: {"quote": value, "value": value, "confidence": confidence}  # noqa: E731

    def respond(messages):
        body = messages[1]["content"]
        assert "processed in parts" in body
        document = body[body.index("<document>"):]
        response = to_json_nulls()
        if "Alpha review" in document:
            response["meeting_title"] = leaf("Alpha review", 0.8)
        if "Dana Fox" in document:
            response["meeting_title"] = leaf("Alpha review", 0.95) if "Alpha review" in document else leaf(None)
            response["attendees"] = [{"name": leaf("Dana Fox"), "role": leaf(None), "attendance": leaf("present")}]
        return response

    chunks = split_text(text, 1_500)
    fake["responses"] = [respond] * len(chunks)
    outcome = LLMSolution(AZURE).extract(ParsedDocument(name="long.txt", text=text), make_schema(),
                                         {"max_chunk_chars": 2_000}, context())
    assert len(fake["calls"]) == len(split_text(text, 2_000)) > 1
    assert outcome.usage.calls == len(fake["calls"])
    assert outcome.record["meeting_title"].value == "Alpha review"
    assert [person["name"].value for person in outcome.record["attendees"]] == ["Dana Fox"]  # overlap de-duplicated
    assert outcome.notes[0].startswith(f"Chunked into {len(fake['calls'])} parts")
    assert len(outcome.raw["calls"]) == len(fake["calls"])


def to_json_nulls():
    """A schema-shaped all-null strict response for the meeting-minutes schema."""
    null = {"quote": None, "value": None, "confidence": None}
    response = {field["name"]: ([] if field.get("array") else dict(null)) for field in meeting_minutes_schema()["fields"]}
    return response


def test_extract_splits_chunk_that_overflows_output(fake):
    text = "\n\n".join(f"Paragraph {index}: " + "content " * 30 for index in range(30))
    overflow = ServiceError("The model ran out of output tokens during extraction. Use smaller chunks.", 502)
    fake["responses"] = [overflow] + [lambda messages: to_json_nulls()] * 4
    outcome = LLMSolution(AZURE).extract(ParsedDocument(name="o.txt", text=text), make_schema(), {}, context())
    assert len(fake["calls"]) >= 3
    assert all("processed in parts" in call["messages"][1]["content"] for call in fake["calls"][1:])
    assert any("exceeded the model's output limit" in note for note in outcome.notes)


def test_extract_propagates_service_errors(fake):
    fake["responses"] = [ServiceError("Azure rate limit reached. Wait a moment and retry.", 429)]
    with pytest.raises(ServiceError) as error:
        LLMSolution(AZURE).extract(ParsedDocument(name="m.txt", text=MINUTES), make_schema(), {}, context())
    assert error.value.status == 429 and fake["client"].closed


def test_extract_overflow_on_small_chunk_is_not_retried(fake):
    fake["responses"] = [ServiceError("The model ran out of output tokens during extraction.", 502)]
    with pytest.raises(ServiceError):
        LLMSolution(AZURE).extract(ParsedDocument(name="m.txt", text=MINUTES), make_schema(), {}, context())
    assert len(fake["calls"]) == 1


def test_extract_cancellation_between_chunks(fake):
    run = RunContext(store=DocumentStore(), cancel=threading.Event())
    text = "\n\n".join("Paragraph " + "text " * 50 for _ in range(40))

    def cancel_after_first(messages):
        run.cancel.set()
        return to_json_nulls()

    fake["responses"] = [cancel_after_first, cancel_after_first]
    with pytest.raises(ServiceError) as error:
        LLMSolution(AZURE).extract(ParsedDocument(name="c.txt", text=text), make_schema(),
                                   {"max_chunk_chars": 2_000}, run)
    assert error.value.status == 409 and len(fake["calls"]) == 1


def test_extract_self_check_uses_reviewed_record(fake):
    reviewed = minutes_response(chair={"quote": "Alice Jones", "value": "Alice Jones", "confidence": 0.7})
    reviewed["attendees"] = reviewed["attendees"][:2]
    fake["responses"] = [minutes_response(), reviewed]
    outcome = LLMSolution(AZURE).extract(ParsedDocument(name="m.txt", text=MINUTES), make_schema(),
                                         {"self_check": True}, context())
    assert [call["operation"] for call in fake["calls"]] == ["extraction", "self-check"]
    review_prompt = fake["calls"][1]["messages"][1]["content"]
    assert "Draft extraction to review" in review_prompt and '"Carol King"' in review_prompt
    assert outcome.record["chair"].confidence == 0.7 and len(outcome.record["attendees"]) == 2
    assert outcome.usage.calls == 2
    assert any(note.startswith("Self-check: 0 values added or changed, 2 removed") for note in outcome.notes)


def test_extract_requires_configuration():
    with pytest.raises(ServiceError) as error:
        LLMSolution(Settings()).extract(ParsedDocument(name="m.txt", text=MINUTES), make_schema(), {}, context())
    assert error.value.status == 503


# -- metadata, settings, status --------------------------------------------------------------

def test_status_and_model_label():
    assert LLMSolution(AZURE).status()["ready"] is True
    assert LLMSolution(Settings()).status()["ready"] is False
    label = LLMSolution(AZURE).model_label({"reasoning_effort": "high", "self_check": True})
    assert label == "test-deployment · high reasoning · self-check"


def test_status_never_raises_on_bad_endpoint(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "http://not-azure.example.com")
    status = LLMSolution().status()
    assert status["ready"] is False and "endpoint" in status["detail"].lower()


def test_settings_spec_and_validation():
    solution = LLMSolution(AZURE)
    specs = {spec.key: spec for spec in solution.settings_spec()}
    assert set(specs) == {"reasoning_effort", "max_chunk_chars", "self_check", "extra_instructions"}
    defaults = solution.validate_settings({})
    assert defaults["self_check"] is False and defaults["reasoning_effort"] in specs["reasoning_effort"].options
    with pytest.raises(ServiceError):
        solution.validate_settings({"reasoning_effort": "extreme"})
    with pytest.raises(ServiceError):
        solution.validate_settings({"max_chunk_chars": 10})
    info = solution.info()
    assert info["id"] == "llm" and info["route"] == "/apps/llm" and info["runtime"] == "cloud"
    assert solution.document_concurrency >= 2


def test_plain_record_round_trip(fake):
    fake["responses"] = [minutes_response()]
    outcome = LLMSolution(AZURE).extract(ParsedDocument(name="m.txt", text=MINUTES), make_schema(), {}, context())
    data = plain(outcome.record)
    assert data["attendees"][2] == {"name": "Carol King", "role": None, "attendance": "apologies"}
    json.dumps(data)


# -- stitching items split across chunks ----------------------------------------------------

def test_stitch_joins_continuations_and_keeps_distinct_items():
    from app.solutions.llm import Chunk, carry_hint, stitch
    text = ("5 Baseline approval\nThe Board discussed the baseline.\n"
            "The Board approved the baseline.\nAction: Ann Lee to publish the plan.\n"
            "6 Procurement\nAction: Ann Lee to brief suppliers.\n")
    end, start = text.index("6 Procurement"), text.index("The Board approved")  # overlapping parts
    first, second = Chunk(0, end, text[:end]), Chunk(start, len(text), text[start:])
    schema = make_schema()
    item = lambda number, title, decisions: {  # noqa: E731
        "number": Leaf(number, number, 0.9), "title": Leaf(title, title, 0.9),
        "decisions": [Leaf(value, value, 0.9) for value in decisions]}
    action = lambda description, quote: {  # noqa: E731
        "description": Leaf(description, quote, 0.9), "owner": Leaf("Ann Lee", "Ann Lee", 0.9), "due_date": Leaf(None)}
    one = to_record({}, schema.fields)
    one["agenda_items"] = [item("5", "Baseline approval", ["The Board discussed the baseline."])]
    one["actions"] = [action("Publish the plan", "to publish the plan")]
    two = to_record({}, schema.fields)
    two["agenda_items"] = [item("5", None, ["The Board approved the baseline."]), item("6", "Procurement", [])]
    two["actions"] = [action("Publish the plan.", "Ann Lee to publish the plan"),
                      action("Brief suppliers", "to brief suppliers")]
    records, joined = stitch(schema.fields, [(first, one), (second, two)])
    assert joined == 2
    merged = merge_annotated_for(schema, records)
    assert [(entry["number"], entry["title"], len(entry["decisions"])) for entry in merged["agenda_items"]] == [
        ("5", "Baseline approval", 2), ("6", "Procurement", 0)]
    assert [entry["description"] for entry in merged["actions"]] == ["Publish the plan", "Brief suppliers"]
    assert carry_hint(schema.fields, two) == {"agenda_items": {"number": "6", "title": "Procurement"},
                                              "actions": {"owner": "Ann Lee"}}


def test_stitch_does_not_join_conflicting_items():
    from app.solutions.llm import Chunk, stitch
    schema = make_schema()
    chunk = Chunk(0, 10, "0123456789")
    one, two = to_record({}, schema.fields), to_record({}, schema.fields)
    one["attendees"] = [{"name": Leaf("Bob Lee"), "role": Leaf(None), "attendance": Leaf("present")}]
    two["attendees"] = [{"name": Leaf("Bob Lee"), "role": Leaf(None), "attendance": Leaf("apologies")},
                        {"name": Leaf("Cy Ng"), "role": Leaf(None), "attendance": Leaf("present")}]
    records, joined = stitch(schema.fields, [(chunk, one), (chunk, two)])
    assert joined == 0 and len(records[1]["attendees"]) == 2


def merge_annotated_for(schema, records):
    from app.results import merge_annotated
    return plain(merge_annotated(schema.fields, records))
