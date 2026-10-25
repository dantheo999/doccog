import time

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.compare import compare_runs, value_similarity
from app.documents import DocumentStore, parse_document
from app.grounding import Grounder
from app.jobs import JobRunner
from app.models import ExtractionItem
from app.results import DocumentResult, Leaf, Run, ValueBuilder, annotate_plain, merge_annotated
from app.schema import ExtractionSchema, SchemaDraft, SchemaField, coerce_record, field_guide, meeting_minutes_schema, parse_date
from app.solutions.base import Outcome, SettingSpec, Solution
from app.solutions.langextract import assemble, class_guide, class_plan
from app.storage import RunStore, SchemaStore
from app.text_chunks import chunk_text


TEXT = ("PUBLIC MINUTES of the Board meeting on Tuesday 22 October 2019\n"
        "Present\nSimon Dow (SD) Interim Chair\nLiz Butler (LB)\n"
        "In attendance\nJonathan Walters (JW) Deputy Chief Executive\n"
        "4 Matters Arising\nIt was AGREED that management would respond to Unite.\n"
        "Action: CK to add the workshops to the forward planner by 1 December 2019.\n")


def draft() -> SchemaDraft:
    return SchemaDraft.model_validate(meeting_minutes_schema())


def schema() -> ExtractionSchema:
    return ExtractionSchema(**draft().model_dump(), id="s1", version=1, created_at="2026-10-25")


def test_schema_rejects_invalid_shapes():
    with pytest.raises(ValidationError, match="child"):
        SchemaField(name="people", type="object")
    with pytest.raises(ValidationError, match="options"):
        SchemaField(name="status", type="enum")
    with pytest.raises(ValidationError):
        SchemaField(name="Bad Name")
    deep = {"name": "a", "type": "object", "fields": [{"name": "b", "type": "object", "fields": [
        {"name": "c", "type": "object", "fields": [{"name": "d"}]}]}]}
    with pytest.raises(ValidationError, match="deeper"):
        SchemaDraft(name="x", fields=[deep])


def test_field_guide_lists_nested_paths():
    guide = field_guide(draft().fields)
    assert "attendees[].attendance" in guide and '"apologies"' in guide
    assert "agenda_items[].decisions[]" in guide


@pytest.mark.parametrize("text,expected", [
    ("22 October 2019", "2019-10-22"), ("Tuesday 22nd October 2019", "2019-10-22"),
    ("October 22, 2019", "2019-10-22"), ("22/10/2019", "2019-10-22"), ("2019-10-22", "2019-10-22"),
])
def test_parse_date_variants(text, expected):
    assert parse_date(text).isoformat() == expected


def test_coerce_shapes_model_output_to_schema():
    record = coerce_record({"meeting_date": "22 October 2019", "attendees": [
        {"name": "Simon Dow", "attendance": "PRESENT", "extra": 1}, {"name": None}],
        "unknown": "x"}, draft().fields)
    assert record["meeting_date"] == "2019-10-22"
    assert record["attendees"] == [{"name": "Simon Dow", "role": None, "attendance": "present"}]
    assert record["actions"] == [] and "unknown" not in record


def test_grounder_prefers_exact_then_normalised_then_fuzzy():
    grounder = Grounder(TEXT)
    exact = grounder.find("Simon Dow")
    assert exact.kind == "exact" and TEXT[exact.start:exact.end] == "Simon Dow"
    loose = grounder.find("simon   dow")
    assert loose.kind == "fuzzy" and (loose.start, loose.end) == (exact.start, exact.end)
    fuzzy = grounder.find("management will respond to Unite")
    assert fuzzy is not None and fuzzy.kind == "fuzzy"
    assert grounder.find("Completely unrelated sentence about cats") is None
    date = grounder.find_value("2019-10-22", SchemaField(name="d", type="date"))
    assert TEXT[date.start:date.end] == "22 October 2019"


def test_grounder_short_needles_respect_word_boundaries():
    grounder = Grounder("SDX and SD attended")
    match = grounder.find("SD")
    assert match.start == 8


def test_value_builder_grounds_children_near_their_item():
    record = annotate_plain({"attendees": [
        {"name": "Simon Dow", "role": "Interim Chair", "attendance": "present"},
        {"name": "Jonathan Walters", "role": "Deputy Chief Executive", "attendance": "in attendance"},
    ]}, draft().fields)
    data, values = ValueBuilder(TEXT).build(draft().fields, record)
    assert data["attendees"][1]["attendance"] == "in attendance"
    by_path = {tuple(value.path): value for value in values}
    attendance = by_path[("attendees", 1, "attendance")]
    assert TEXT[attendance.start:attendance.end] == "In attendance"
    assert by_path[("attendees", 0, "role")].grounding == "exact"


def test_merge_annotated_keeps_best_scalar_and_dedupes_arrays():
    fields = draft().fields
    first = annotate_plain({"meeting_title": "Minutes", "attendees": [{"name": "A"}]}, fields)
    second = annotate_plain({"meeting_title": "Board minutes", "attendees": [{"name": "A"}, {"name": "B"}]}, fields)
    first["meeting_title"].confidence, second["meeting_title"].confidence = 0.4, 0.9
    merged = merge_annotated(fields, [first, second])
    assert merged["meeting_title"].value == "Board minutes"
    assert [item["name"].value for item in merged["attendees"]] == ["A", "B"]


def test_chunk_text_splits_on_paragraphs():
    text = ("Paragraph one. " * 20 + "\n\n") * 10
    chunks = chunk_text(text, 1_000)
    assert len(chunks) > 1 and "".join(chunks) == text
    assert all(len(chunk) <= 1_000 for chunk in chunks)


def item(cls, text, attributes=None):
    start = TEXT.find(text)
    return ExtractionItem(id="0", extraction_class=cls, extraction_text=text, attributes=attributes or {},
                          start=start, end=start + len(text), alignment_status="match_exact", grounding="exact")


def test_langextract_plan_and_assembly_rebuild_nested_records():
    names = {spec.name: spec for spec in class_plan(draft().fields)}
    assert names["attendees"].text_field.name == "name"
    assert names["actions"].text_field.name == "description"
    assert names["agenda_items__decisions"].parent == "agenda_items"
    assert "agenda_items__decisions" in class_guide(draft().fields)
    record, notes = assemble(draft().fields, [
        item("meeting_date", "22 October 2019", {"value": "2019-10-22"}),
        item("attendees", "Simon Dow", {"role": "Interim Chair", "attendance": "present"}),
        item("agenda_items", "Matters Arising", {"number": "4"}),
        item("agenda_items__decisions", "management would respond to Unite", {"parent": "Matters Arising"}),
        item("actions", "add the workshops to the forward planner", {"owner": "CK", "due_date": "1 December 2019"}),
        item("not_in_schema", "Liz Butler"),
    ])
    data, values = ValueBuilder(TEXT).build(draft().fields, record)
    assert data["meeting_date"] == "2019-10-22"
    assert data["agenda_items"][0]["decisions"] == ["management would respond to Unite"]
    assert data["actions"][0]["due_date"] == "2019-12-01"
    assert any("not_in_schema" in note for note in notes)
    assert all(value.grounding == "exact" for value in values if value.field == "attendees[].name")


def run_with(run_id, data, solution="fake"):
    return Run(id=run_id, solution=solution, solution_name=solution, label=run_id, model="m",
               confidence_method="none", extraction_schema=schema(), settings={}, created_at="2026-10-25",
               documents=[DocumentResult(document_id="d1", document_name="doc.pdf", document_sha256="0" * 64,
                                         status="done", data=coerce_record(data, schema().fields))])


def test_object_array_similarity_aligns_items_out_of_order():
    field = next(field for field in schema().fields if field.name == "attendees")
    a = [{"name": "Simon Dow", "role": "Chair", "attendance": "present"}, {"name": "Liz Butler", "role": None, "attendance": "present"}]
    b = [{"name": "Liz Butler", "role": None, "attendance": "present"}, {"name": "Simon Dow", "role": "Interim Chair", "attendance": "present"}]
    assert value_similarity(a, b, field) > 0.85
    assert value_similarity(a, [], field) == 0


def test_compare_scores_against_reference_and_consensus():
    reference = run_with("ref", {"meeting_date": "2019-10-22", "attendees": [{"name": "Simon Dow"}, {"name": "Liz Butler"}]})
    good = run_with("good", {"meeting_date": "22 October 2019", "attendees": [{"name": "Liz Butler"}, {"name": "Simon Dow"}]})
    poor = run_with("poor", {"meeting_date": "2019-10-23", "attendees": [{"name": "Simon Dow"}]})
    result = compare_runs([reference, good, poor], "ref")
    stats = {run["id"]: run for run in result["runs"]}
    assert stats["good"]["agreement"] > stats["poor"]["agreement"]
    assert stats["good"]["fields"]["meeting_date"]["agreement"] == 1.0
    cell = next(cell for cell in result["cells"] if cell["field"] == "attendees")
    assert len(cell["rows"]) == 2 and set(cell["rows"][0]["cells"]) == {"ref", "good", "poor"}
    consensus = compare_runs([reference, good, poor])
    assert consensus["reference"] is None
    assert {run["id"]: run for run in consensus["runs"]}["poor"]["agreement"] < 1


def test_schema_store_versions_only_on_change():
    store = SchemaStore(None)
    first = store.save(draft())
    same = store.save(draft(), first.id)
    changed = store.save(draft().model_copy(update={"name": "Minutes v2"}), first.id)
    assert same.version == 1 and changed.version == 2
    assert store.get(first.id, 1).name == "Meeting minutes" and store.get(first.id).version == 2


class FakeSolution(Solution):
    id = "fake"
    name = "Fake"
    tagline = description = technology = "test"
    runtime = "local"
    capabilities = []
    confidence_method = "fixed"
    route = "/apps/fake"
    document_concurrency = 2

    def settings_spec(self):
        return [SettingSpec("fail_on", "Fail on", "text", "")]

    def status(self):
        return {"ready": True, "detail": "", "model": "fake-model"}

    def extract(self, document, schema, settings, context):
        if settings["fail_on"] and settings["fail_on"] in document.name:
            from app.azure import ServiceError
            raise ServiceError("Simulated failure.")
        record = annotate_plain({"meeting_title": document.text.split("\n")[0]}, schema.fields)
        record["meeting_title"] = Leaf(record["meeting_title"].value, confidence=0.75)
        return Outcome(record=record, notes=["fake note"])


def test_job_runner_records_results_and_failures():
    store = DocumentStore()
    good = store.add(parse_document("good.txt", TEXT.encode()))
    bad = store.add(parse_document("bad.txt", b"Other minutes\n"))
    runs = RunStore(None)
    run = Run(id="r1", solution="fake", solution_name="Fake", label="t", model="fake", confidence_method="fixed",
              extraction_schema=schema(), settings={"fail_on": "bad"}, created_at="now", documents=[
                  DocumentResult(document_id=doc.id, document_name=doc.name, document_sha256=doc.sha256)
                  for doc in (good, bad)])
    JobRunner(runs, store).start(run, FakeSolution(), wait=True)
    saved = runs.get("r1")
    assert saved.status == "done"
    done, failed = saved.documents
    assert done.status == "done" and done.values[0].confidence == 0.75 and done.values[0].grounding == "exact"
    assert "fake note" in done.notes
    assert failed.status == "error" and failed.error == "Simulated failure."
    assert saved.summary()["documents_failed"] == 1


def test_api_run_lifecycle_and_compare(monkeypatch):
    from app import main
    monkeypatch.setitem(main.solutions, "fake", FakeSolution())
    client = TestClient(main.app)
    assert any(item["id"] == "fake" for item in client.get("/api/solutions").json())
    collection = client.post("/api/collections", json={"name": "minutes"}).json()
    uploaded = client.post("/api/documents", files={"file": ("m.txt", TEXT.encode(), "text/plain")},
                           data={"collection_id": collection["id"], "path": "minutes/m.txt"}).json()
    assert client.get(f"/api/collections/{collection['id']}").json()["documents"][0]["path"] == "minutes/m.txt"
    saved = client.post("/api/schemas", json={"draft": client.get("/api/schemas/starter").json()}).json()
    response = client.post("/api/runs", json={"solution": "fake", "schema_id": saved["id"],
                                              "document_ids": [uploaded["id"]], "settings": {}})
    assert response.status_code == 200, response.text
    run_id = response.json()["id"]
    for _ in range(50):
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] not in ("queued", "running"):
            break
        time.sleep(0.05)
    assert run["status"] == "done" and run["documents"][0]["data"]["meeting_title"].startswith("PUBLIC MINUTES")
    comparison = client.post("/api/compare", json={"run_ids": [run_id]}).json()
    assert comparison["documents"][0]["name"] == "m.txt"
    assert client.post("/api/runs", json={"solution": "fake", "schema_id": saved["id"], "document_ids": [uploaded["id"]],
                                          "settings": {"unknown": 1}}).status_code == 422
    assert client.delete(f"/api/runs/{run_id}").status_code == 200
