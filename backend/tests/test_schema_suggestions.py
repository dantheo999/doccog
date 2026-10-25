"""Offline tests for schema suggestion: the shared suggestion and each solution's own method."""
from types import ModuleType, SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import nuextract_engine, schema_suggest
from app.azure import ServiceError
from app.cognition import CognitionError
from app.documents import DocumentStore, parse_document
from app.models import Recipe
from app.results import Usage
from app.schema import SchemaDraft
from app.settings import Settings
from app.solutions import langextract as langextract_module, nuextract as nuextract_module
from app.solutions.base import RunContext, SettingSpec, Solution, SuggestionOutcome
from app.solutions.langextract import DEFAULT_SUGGESTION_GOAL, LangExtractSolution, infer_type, schema_from_recipe
from app.solutions.llm import LLMSolution
from app.solutions.nuextract import NuExtractSolution
from app.solutions.unstract import UnstractSolution


AZURE = Settings(endpoint="https://example.openai.azure.com", deployment="test-deployment", api_key="key")
MINUTES = ("Present\nSimon Dow (SD) Interim Chair\nLiz Butler (LB) Member\nMeeting held on 22 October 2019.\n"
           "Action: CK to add workshops by 1 December 2019.\nAGREED: respond to Unite.\n")
OTHER = "Apologies: Jane Roe.\nDecisions: approve budget; defer item 4.\n"
SUGGESTED = {
    "name": "Meeting minutes", "description": "Board minutes.",
    "fields": [
        {"name": "Meeting Date", "type": "date", "description": "Date held.", "array": False, "options": ["x"],
         "fields": []},
        {"name": "attendees", "type": "object", "description": "People present.", "array": True, "options": [],
         "fields": [{"name": "name", "type": "verbatim", "description": "", "array": False, "options": [],
                     "fields": []}]},
    ],
    "notes": ["Exemplars differ in layout."],
}


def documents():
    store = DocumentStore()
    return store, [store.add(parse_document("a.txt", MINUTES.encode())), store.add(parse_document("b.txt", OTHER.encode()))]


class Minimal(Solution):
    id, name, tagline, description, technology = "minimal", "Minimal", "t", "d", "x"
    runtime = "local"
    capabilities: list[str] = []
    confidence_method, route = "none", "/apps/minimal"

    def status(self):
        return {"ready": True, "detail": "", "model": ""}

    def extract(self, document, schema, settings, context):
        raise NotImplementedError


class Suggesting(Minimal):
    id, name = "suggesting", "Suggesting"
    schema_suggestion_method = "Echoes a fixed schema."

    def __init__(self, ready: bool = True, error: ServiceError | None = None):
        self.ready, self.error, self.calls = ready, error, []

    def status(self):
        return {"ready": self.ready, "detail": "" if self.ready else "Down for maintenance.", "model": ""}

    def suggestion_settings_spec(self):
        return [SettingSpec("depth", "Depth", "integer", 2, min=1, max=3)]

    def suggest_schema(self, documents, guidance, settings, context):
        self.calls.append((documents, guidance, settings))
        if self.error:
            raise self.error
        draft = SchemaDraft.model_validate({"name": "Fixed", "fields": [{"name": "title", "type": "verbatim"}]})
        return SuggestionOutcome(draft=draft, notes=["fixed note"], usage=Usage(calls=1, prompt_tokens=5),
                                 raw={"big": "x" * 10})


# --- Base capability ----------------------------------------------------------------------

def test_solutions_do_not_suggest_by_default():
    solution = Minimal()
    assert solution.info()["schema_suggestion"] == {
        "supported": False, "method": "This solution does not suggest schemas.", "settings": []}
    with pytest.raises(ServiceError) as error:
        solution.suggest_schema([], "", {}, RunContext(store=DocumentStore()))
    assert error.value.status == 422


def test_suggestion_settings_resolve_leniently():
    solution = Suggesting()
    assert solution.resolve_suggestion_settings({"depth": 9, "unknown": True}) == {"depth": 2}
    assert solution.resolve_suggestion_settings({"depth": 3}) == {"depth": 3}
    info = solution.info()["schema_suggestion"]
    assert info["supported"] and info["method"] == "Echoes a fixed schema." and info["settings"][0]["key"] == "depth"


def test_each_solution_declares_its_suggestion_capability():
    infos = {solution.id: solution.info()["schema_suggestion"] for solution in
             (LLMSolution(azure=AZURE), NuExtractSolution(), LangExtractSolution(), UnstractSolution())}
    assert infos["llm"]["supported"] and "same approach as the shared suggestion" in infos["llm"]["method"].lower()
    assert [spec["key"] for spec in infos["llm"]["settings"]] == ["reasoning_effort"]
    assert infos["llm"]["settings"][0]["default"] == "medium"
    assert infos["nuextract"]["supported"] and "template" in infos["nuextract"]["method"].lower()
    assert {spec["key"]: spec["default"] for spec in infos["nuextract"]["settings"]} == {
        "enable_thinking": False, "max_tokens": 4096}
    assert infos["langextract"]["supported"] and infos["langextract"]["method"].startswith("Derived")
    assert infos["unstract"] == {"supported": False, "settings": [], "method": (
        "Unstract's open-source edition has no prompt or schema generation; its source references an Agentic "
        "Studio project type that only exists in the cloud and on-prem editions (not verified here). Generate "
        "the Prompt Studio project from the agreed schema instead.")}


# --- Direct LLM: the shared method with its own settings ----------------------------------

def test_llm_suggestion_reuses_the_shared_method(monkeypatch):
    calls = []

    def fake_chat_json(settings, messages, **kwargs):
        calls.append((settings, messages, kwargs))
        return SUGGESTED, Usage(calls=1, prompt_tokens=100, completion_tokens=20), 1.0

    monkeypatch.setattr(schema_suggest, "chat_json", fake_chat_json)
    store, docs = documents()
    outcome = LLMSolution(azure=AZURE).suggest_schema(docs, "board minutes", {"reasoning_effort": "high"},
                                                      RunContext(store=store))
    settings, messages, kwargs = calls[0]
    assert settings is AZURE and kwargs["reasoning_effort"] == "high"
    assert kwargs["response_format"] == schema_suggest.response_format()
    assert messages[0]["content"] == schema_suggest.SUGGEST_PROMPT
    assert "board minutes" in messages[1]["content"] and "Simon Dow" in messages[1]["content"]
    assert [field.name for field in outcome.draft.fields] == ["meeting_date", "attendees"]
    assert outcome.draft.fields[0].options == []
    assert outcome.notes == ["Exemplars differ in layout."] and outcome.usage.prompt_tokens == 100
    LLMSolution(azure=AZURE).suggest_schema(docs, "", {"reasoning_effort": "bogus"}, RunContext(store=store))
    assert calls[1][2]["reasoning_effort"] == "medium"


# --- LangExtract: schema derived from a recipe's classes ----------------------------------

def recipe() -> Recipe:
    return Recipe.model_validate({
        "id": "r", "version": 1, "instructions": "Extract minutes.", "goal": "g", "source_name": "a.txt",
        "created_at": "2026-10-25",
        "examples": [
            {"text": MINUTES, "extractions": [
                {"extraction_class": "attendee", "extraction_text": "Simon Dow",
                 "attributes": {"role": "Interim Chair", "present": "yes"}},
                {"extraction_class": "attendee", "extraction_text": "Liz Butler",
                 "attributes": {"role": "Member", "present": True}},
                {"extraction_class": "meeting_date", "extraction_text": "22 October 2019"},
                {"extraction_class": "Action Item", "extraction_text": "CK to add workshops",
                 "attributes": {"owner": "CK", "due_date": "1 December 2019", "priority": 2,
                                "tags": ["planning"], "cost": "1.5", "Text": "x"}},
                {"extraction_class": "Decision", "extraction_text": "respond to Unite"},
            ]},
            {"text": OTHER, "extractions": [
                {"extraction_class": "attendee", "extraction_text": "Jane Roe",
                 "attributes": {"role": None, "present": "no"}},
                {"extraction_class": "Decision", "extraction_text": "approve budget"},
                {"extraction_class": "Decision", "extraction_text": "defer item 4"},
                {"extraction_class": "action-item", "extraction_text": "Jane Roe"},
            ]},
        ],
    })


@pytest.mark.parametrize("values, expected", [
    (["2019-10-22", "1 December 2019", None], ("date", False)),
    ([3, "4", "1,000"], ("integer", False)),
    (["1.5", 2], ("number", False)),
    ([True, "no", "Yes"], ("boolean", False)),
    ([["a", "b"], "c"], ("text", True)),
    (["Chair", "2019-10-22"], ("text", False)),
    ([None, ""], ("text", False)),
    (["4.1", "4.10", "5"], ("verbatim", False)),  # item numbers: 4.10 is not the number 4.1
    (["007", "012"], ("verbatim", False)),  # references keep their leading zeros
    (["4.1.2", 5], ("verbatim", False)),
    (["0", "0.5", "-3", 2.25], ("number", False)),
    (["12.3.2019", "1 December 2019"], ("date", False)),
])
def test_attribute_types_are_inferred_from_values(values, expected):
    assert infer_type(values) == expected


def test_schema_is_derived_from_recipe_classes():
    fields, notes = schema_from_recipe(recipe())
    by_name = {field["name"]: field for field in fields}
    assert list(by_name) == ["attendee", "meeting_date", "action_item", "decision", "action_item_2"]
    attendee = by_name["attendee"]
    assert attendee["type"] == "object" and attendee["array"]
    assert attendee["description"] == "From LangExtract class attendee (3 annotations)."
    assert [(child["name"], child["type"], child["array"]) for child in attendee["fields"]] == [
        ("text", "verbatim", False), ("role", "text", False), ("present", "boolean", False)]
    assert by_name["meeting_date"] == {"name": "meeting_date", "type": "verbatim", "array": False, "options": [],
                                       "fields": [], "description": "From LangExtract class meeting_date (1 annotation)."}
    action = by_name["action_item"]
    assert action["type"] == "object" and not action["array"]
    assert [(child["name"], child["type"], child["array"]) for child in action["fields"]] == [
        ("text", "verbatim", False), ("owner", "text", False), ("due_date", "date", False),
        ("priority", "integer", False), ("tags", "text", True), ("cost", "number", False), ("text_2", "text", False)]
    assert by_name["decision"]["type"] == "verbatim" and by_name["decision"]["array"]
    assert by_name["action_item_2"]["type"] == "verbatim" and not by_name["action_item_2"]["array"]
    assert "2 example(s) with 9 annotations in 5 classes" in notes[-1]
    assert not any("at most once per example" in note for note in notes)
    single = recipe().model_copy(update={"examples": [
        example.model_copy(update={"extractions": [item for item in example.extractions
                                                   if item.extraction_text != "defer item 4"]})
        for example in recipe().examples]})
    single_fields, single_notes = schema_from_recipe(single)
    assert not next(field for field in single_fields if field["name"] == "decision")["array"]
    assert any(note.startswith("Decision occurred at most once per example") for note in single_notes)


def test_langextract_suggestion_generates_a_recipe_without_a_schema(monkeypatch):
    calls = []

    def fake_generate_recipe(documents, goal, settings, *args, **kwargs):
        calls.append((documents, goal, args, kwargs))
        return recipe()

    monkeypatch.setattr(langextract_module, "generate_recipe", fake_generate_recipe)
    store, docs = documents()
    outcome = LangExtractSolution().suggest_schema(docs, "  ", {}, RunContext(store=store))
    assert calls[0][1] == DEFAULT_SUGGESTION_GOAL and calls[0][0] == docs
    assert calls[0][2] == () and "class_guide" not in calls[0][3]
    assert [field.name for field in outcome.draft.fields][:2] == ["attendee", "meeting_date"]
    assert outcome.usage == Usage()
    text = " ".join(outcome.notes)
    assert "no schema-suggestion feature" in text and "only nest one level" in text and "usage shows zero" in text
    assert outcome.raw["recipe"]["examples"][0]["extractions"][0]["extraction_text"] == "Simon Dow"
    LangExtractSolution().suggest_schema(docs, "minutes: attendees", {}, RunContext(store=store))
    assert calls[1][1] == "minutes: attendees"

    def failing(*args, **kwargs):
        raise CognitionError("Azure is busy.", 429)

    monkeypatch.setattr(langextract_module, "generate_recipe", failing)
    with pytest.raises(ServiceError) as error:
        LangExtractSolution().suggest_schema(docs, "", {}, RunContext(store=store))
    assert error.value.status == 429 and str(error.value) == "Azure is busy."


# --- NuExtract: native template generation, one call per exemplar --------------------------

TEMPLATES = {
    "a.txt": {"attendees": [{"name": "verbatim-string", "role": "string"}], "meeting_date": "date"},
    "b.txt": {"attendees": [{"name": "verbatim-string", "attendance": ["present", "apologies"]}],
              "meeting_date": "string", "venue": "verbatim-string"},
}


def fake_engine(templates: dict | None = None, *, with_generation: bool = True, errors: dict | None = None):
    module = ModuleType("fake_nuextract_engine")
    module.NuExtractSettings = nuextract_engine.NuExtractSettings
    module.NuExtractError = nuextract_engine.NuExtractError
    module.calls = []

    def generate_template(settings, *, text=None, description="", enable_thinking=False, temperature=0.2,
                          max_tokens=4096):
        module.calls.append({"text": text, "description": description, "enable_thinking": enable_thinking,
                             "max_tokens": max_tokens})
        name = "a.txt" if text == MINUTES else "b.txt"
        if errors and name in errors:
            raise errors[name]
        template = (templates or TEMPLATES).get(name)
        return SimpleNamespace(template=template, raw=str(template), reasoning=None, elapsed_seconds=1.5,
                               usage={"prompt_tokens": 100, "completion_tokens": 10})

    def template_to_fields(template):  # minimal stand-in for the engine's converter
        def convert(key, value):
            field = {"name": key, "description": "", "array": isinstance(value, list), "options": [], "fields": []}
            item = value[0] if isinstance(value, list) and len(value) == 1 else value
            if isinstance(item, dict):
                return {**field, "type": "object", "fields": [convert(k, v) for k, v in item.items()]}
            if isinstance(value, list):
                return {**field, "type": "enum", "array": False, "options": value}
            return {**field, "type": {"verbatim-string": "verbatim", "date": "date"}.get(item, "text")}
        return [convert(key, value) for key, value in template.items()]

    if with_generation:
        module.generate_template = generate_template
        module.template_to_fields = template_to_fields
    return module


def test_nuextract_merges_templates_from_each_exemplar(monkeypatch):
    engine = fake_engine()
    monkeypatch.setattr(nuextract_module, "engine", engine)
    store, docs = documents()
    outcome = NuExtractSolution().suggest_schema(docs, " meeting minutes ", {"max_tokens": 2048, "bogus": 1},
                                                 RunContext(store=store))
    assert [call["text"] for call in engine.calls] == [MINUTES, OTHER]
    assert all(call["description"] == "meeting minutes" and call["max_tokens"] == 2048
               and call["enable_thinking"] is False for call in engine.calls)
    fields = {field.name: field for field in outcome.draft.fields}
    assert list(fields) == ["attendees", "meeting_date", "venue"]
    assert fields["attendees"].array and [child.name for child in fields["attendees"].fields] == [
        "name", "role", "attendance"]
    assert fields["attendees"].fields[2].type == "enum" and fields["attendees"].fields[2].options == [
        "present", "apologies"]
    assert fields["meeting_date"].type == "date"  # the first exemplar's type wins
    assert outcome.usage == Usage(calls=2, prompt_tokens=200, completion_tokens=20)
    assert outcome.notes[0] == ("NuExtract template generation ran once per exemplar (2 exemplars): 2 calls completed; "
                                "2 templates were merged by name (the first type wins).")
    assert any("meeting_date" in note and "kept date" in note for note in outcome.notes)
    assert [item["document"] for item in outcome.raw["outputs"]] == ["a.txt", "b.txt"]


def test_nuextract_suggestion_with_the_engine_converter(monkeypatch):
    if not hasattr(nuextract_engine, "template_to_fields"):
        pytest.skip("This engine build has no template_to_fields yet.")
    engine = fake_engine()
    engine.template_to_fields = nuextract_engine.template_to_fields
    monkeypatch.setattr(nuextract_module, "engine", engine)
    store, docs = documents()
    draft = NuExtractSolution().suggest_schema(docs, "", {}, RunContext(store=store)).draft
    fields = {field.name: field for field in draft.fields}
    assert fields["attendees"].type == "object" and fields["attendees"].array
    assert {child.name for child in fields["attendees"].fields} >= {"name", "role", "attendance"}
    assert fields["meeting_date"].type == "date" and fields["venue"].type == "verbatim"


def test_nuextract_skips_failed_exemplars_and_reports_total_failure(monkeypatch):
    error = nuextract_engine.NuExtractError("Context too long.", kind="context_length")
    monkeypatch.setattr(nuextract_module, "engine", fake_engine(errors={"b.txt": error}))
    store, docs = documents()
    outcome = NuExtractSolution().suggest_schema(docs, "", {}, RunContext(store=store))
    assert [field.name for field in outcome.draft.fields] == ["attendees", "meeting_date"]
    assert any("No template for b.txt: Context too long." in note for note in outcome.notes)
    assert outcome.usage.calls == 1 and outcome.notes[0] == (
        "NuExtract template generation ran once per exemplar (2 exemplars): 1 call completed, 1 failed; "
        "the schema comes from a single template.")
    monkeypatch.setattr(nuextract_module, "engine", fake_engine(errors={"a.txt": error, "b.txt": error}))
    with pytest.raises(ServiceError) as failure:
        NuExtractSolution().suggest_schema(docs, "", {}, RunContext(store=store))
    assert failure.value.status == 413
    monkeypatch.setattr(nuextract_module, "engine", fake_engine({"a.txt": None, "b.txt": {}}))
    with pytest.raises(ServiceError, match="no usable template|did not return a usable template"):
        NuExtractSolution().suggest_schema(docs, "", {}, RunContext(store=store))


def test_nuextract_without_template_generation_is_501(monkeypatch):
    monkeypatch.setattr(nuextract_module, "engine", fake_engine(with_generation=False))
    store, docs = documents()
    with pytest.raises(ServiceError) as error:
        NuExtractSolution().suggest_schema(docs, "", {}, RunContext(store=store))
    assert error.value.status == 501 and "not available in this engine build" in str(error.value)


def test_field_lists_merge_recursively_by_name():
    first = schema_suggest.sanitise_fields([
        {"name": "items", "type": "object", "array": False, "fields": [{"name": "a", "type": "text"}]},
        {"name": "kind", "type": "enum", "options": ["x"]}])
    other = schema_suggest.sanitise_fields([
        {"name": "items", "type": "object", "array": True, "description": "Items.",
         "fields": [{"name": "a", "type": "date"}, {"name": "b", "type": "text"}]},
        {"name": "kind", "type": "enum", "options": ["X", "y"]}])
    merged, conflicts = schema_suggest.merge_field_lists(first, other)
    assert merged[0]["array"] and merged[0]["description"] == "Items."
    assert [(child["name"], child["type"]) for child in merged[0]["fields"]] == [("a", "text"), ("b", "text")]
    assert merged[1]["options"] == ["x", "y"]  # case variants are one option (the first spelling is kept)
    assert conflicts == ["items: one exemplar suggested a single value, another a list; kept a list.",
                         "items.a: kept text; another exemplar suggested date."]
    assert first[0]["fields"] == [{"name": "a", "type": "text", "description": "", "array": False,
                                   "options": [], "fields": []}]  # inputs are not mutated


def test_case_variant_enum_options_from_two_exemplars_merge(monkeypatch):
    templates = {"a.txt": {"attendees": [{"name": "verbatim-string", "status": ["Present", "Apologies"]}]},
                 "b.txt": {"attendees": [{"name": "verbatim-string", "status": ["present", "absent"]}]}}
    monkeypatch.setattr(nuextract_module, "engine", fake_engine(templates))
    store, docs = documents()
    draft = NuExtractSolution().suggest_schema(docs, "", {}, RunContext(store=store)).draft
    assert draft.fields[0].fields[1].options == ["Present", "Apologies", "absent"]
    assert schema_suggest.sanitise_fields([{"name": "s", "type": "enum", "options": ["Yes", " yes", "No", ""]}])[0][
        "options"] == ["Yes", "No"]  # the shared path tolerates case variants from one response too


def test_unique_names_stay_within_64_characters():
    long = "decision_recorded_by_the_board_of_directors_during_the_quarterly_meeting"
    seen: set[str] = set()
    names = [schema_suggest.unique_name(schema_suggest.snake_name(f"{long} {suffix}"), seen) for suffix in "ABC"]
    assert names == [long[:64], long[:62] + "_2", long[:62] + "_3"]
    assert schema_suggest.unique_name("x" * 63, {"x" * 63, "x" * 62 + "_2"}) == "x" * 62 + "_3"
    fields, _ = schema_from_recipe(Recipe.model_validate({
        "id": "r", "version": 1, "instructions": "i", "goal": "g", "source_name": "a.txt", "created_at": "2026-10-25",
        "examples": [{"text": f"{long} A. {long} B.", "extractions": [
            {"extraction_class": f"{long} A", "extraction_text": "A"},
            {"extraction_class": f"{long} B", "extraction_text": "B"}]}]}))
    assert [field["name"] for field in fields] == [long[:64], long[:62] + "_2"]


# --- API -----------------------------------------------------------------------------------

@pytest.fixture
def api(monkeypatch):
    from app import main
    client = TestClient(main.app)
    ids = [client.post("/api/documents", files={"file": (name, text.encode(), "text/plain")}).json()["id"]
           for name, text in (("a.txt", MINUTES), ("b.txt", OTHER))]
    return main, client, ids


def test_suggest_schema_endpoint_persists_lists_and_deletes(api, monkeypatch):
    main, client, ids = api
    solution = Suggesting()
    monkeypatch.setitem(main.solutions, "suggesting", solution)
    response = client.post("/api/solutions/suggesting/suggest-schema",
                           json={"document_ids": ids, "guidance": "minutes", "settings": {"depth": 3, "x": 1}})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["solution"] == "suggesting" and body["solution_name"] == "Suggesting"
    assert body["method"] == "Echoes a fixed schema." and body["draft"]["fields"][0]["name"] == "title"
    assert body["notes"] == ["fixed note"] and body["usage"] == {"calls": 1, "prompt_tokens": 5, "completion_tokens": 0}
    assert body["exemplars"] == ["a.txt", "b.txt"] and body["document_ids"] == ids
    assert body["guidance"] == "minutes" and body["settings"] == {"depth": 3} and body["raw"] == {"big": "x" * 10}
    assert isinstance(body["elapsed_seconds"], float) and body["created_at"]
    assert solution.calls[0][1:] == ("minutes", {"depth": 3})
    listed = client.get("/api/schema-suggestions").json()
    assert listed[0]["id"] == body["id"] and "raw" not in listed[0]
    assert client.get(f"/api/schema-suggestions/{body['id']}").json()["raw"] == {"big": "x" * 10}
    assert client.delete(f"/api/schema-suggestions/{body['id']}").json() == {"deleted": body["id"]}
    assert all(item["id"] != body["id"] for item in client.get("/api/schema-suggestions").json())
    assert client.get(f"/api/schema-suggestions/{body['id']}").status_code == 404


def test_suggest_schema_endpoint_errors(api, monkeypatch):
    main, client, ids = api
    url = "/api/solutions/{}/suggest-schema"
    assert client.post(url.format("missing"), json={"document_ids": ids}).status_code == 404
    assert client.post(url.format("missing"), json={}).status_code == 404  # unknown solution before body checks
    monkeypatch.setitem(main.solutions, "unstract", UnstractSolution())
    for body in ({"document_ids": ids}, {"document_ids": []}):
        unsupported = client.post(url.format("unstract"), json=body)
        assert unsupported.status_code == 422
        assert unsupported.json()["detail"] == UnstractSolution.schema_suggestion_unsupported
    monkeypatch.setitem(main.solutions, "suggesting", Suggesting())
    for body in ({"document_ids": []}, {"document_ids": [ids[0], ids[0]]}, {"document_ids": [f"d{index}" for index in range(6)]},
                 {"document_ids": ids, "guidance": "x" * 4_001}, {"document_ids": ids, "extra": 1}, None, [ids]):
        invalid = client.post(url.format("suggesting"), json=body)
        assert invalid.status_code == 422 and invalid.json()["detail"][0]["loc"][0] == "body", body
    monkeypatch.setitem(main.solutions, "suggesting", Suggesting(ready=False))
    down = client.post(url.format("suggesting"), json={"document_ids": ids})
    assert down.status_code == 503 and down.json()["detail"] == "Down for maintenance."
    monkeypatch.setitem(main.solutions, "suggesting", Suggesting(error=ServiceError("Upstream timed out.", 504)))
    failed = client.post(url.format("suggesting"), json={"document_ids": ids})
    assert failed.status_code == 504 and failed.json()["detail"] == "Upstream timed out."
    monkeypatch.setitem(main.solutions, "nuextract", NuExtractSolution())
    monkeypatch.setattr(nuextract_module, "engine", fake_engine(with_generation=False))
    monkeypatch.setattr(NuExtractSolution, "status", lambda self: {"ready": True, "detail": "", "model": "m"})
    assert client.post(url.format("nuextract"), json={"document_ids": ids[:1]}).status_code == 501


def test_llm_suggestion_through_the_api(api, monkeypatch):
    main, client, ids = api
    monkeypatch.setitem(main.solutions, "llm", LLMSolution(azure=AZURE))
    monkeypatch.setattr(schema_suggest, "chat_json",
                        lambda *args, **kwargs: (SUGGESTED, Usage(calls=1, prompt_tokens=7, completion_tokens=3), 0.1))
    response = client.post("/api/solutions/llm/suggest-schema", json={"document_ids": ids[:1]})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["solution"] == "llm" and body["settings"] == {"reasoning_effort": "medium"}
    assert body["usage"]["calls"] == 1 and [field["name"] for field in body["draft"]["fields"]] == [
        "meeting_date", "attendees"]


def test_shared_suggestion_keeps_its_response_and_is_persisted(api, monkeypatch):
    main, client, ids = api
    calls = []

    def fake_suggest(documents, guidance, settings, reasoning_effort):
        calls.append((documents, guidance, reasoning_effort))
        return (schema_suggest.build_draft("Minutes", "", SUGGESTED["fields"]), ["shared note"],
                Usage(calls=1, prompt_tokens=9, completion_tokens=4))

    monkeypatch.setattr(main, "suggest_schema", fake_suggest)
    response = client.post("/api/schemas/suggest", json={"document_ids": [ids[0], ids[0], ids[1]],
                                                         "guidance": "g", "reasoning_effort": "low"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"draft", "notes", "usage", "exemplars", "id", "elapsed_seconds"}
    assert body["notes"] == ["shared note"] and body["exemplars"] == ["a.txt", "b.txt"]
    assert body["usage"] == {"calls": 1, "prompt_tokens": 9, "completion_tokens": 4}
    assert [document.name for document in calls[0][0]] == ["a.txt", "b.txt"] and calls[0][2] == "low"
    saved = next(item for item in client.get("/api/schema-suggestions").json() if item["id"] == body["id"])
    assert saved["solution"] == "shared" and saved["solution_name"] == "Shared suggestion"
    assert saved["method"] == ("Solution-neutral: one Azure OpenAI call reads the exemplars' full text and returns "
                               "a schema as strict JSON.")
    assert saved["settings"] == {"reasoning_effort": "low"} and saved["guidance"] == "g"
    assert saved["draft"] == body["draft"] and saved["document_ids"] == ids
    assert client.post("/api/schemas/suggest", json={"document_ids": ids, "reasoning_effort": "max"}).status_code == 422
