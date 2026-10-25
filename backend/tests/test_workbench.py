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
    ("Tue 22 October 2019", "2019-10-22"), ("22-Oct-2019", "2019-10-22"),
    ("Tuesday, 22nd of October 2019", "2019-10-22"), ("Wed. 23 Oct. 2019", "2019-10-23"),
])
def test_parse_date_variants(text, expected):
    assert parse_date(text).isoformat() == expected


@pytest.mark.parametrize("text,kind,expected", [
    ("\u22125", "number", -5), ("\u2212\u00a33.2 million", "number", -3_200_000), ("\u00a31.5 million", "integer", 1_500_000),
    ("\u00a320k", "number", 20_000), ("1.5bn", "number", 1_500_000_000), ("12 months", "integer", 12),
    ("2.5", "integer", 2.5), ("1,234.50", "number", 1234.5), ("Item - 5", "number", "Item - 5"),
])
def test_coerce_numbers_written_in_text(text, kind, expected):
    from app.schema import coerce_scalar
    value = coerce_scalar(text, SchemaField(name="n", type=kind))
    assert value == expected and type(value) is type(expected)


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


def test_merge_keeps_repeats_within_a_chunk_but_drops_overlap_repeats():
    fields = draft().fields
    first = annotate_plain({"attendees": [{"name": "FSRU"}, {"name": "FSRU"}, {"name": "A"}]}, fields)
    second = annotate_plain({"attendees": [{"name": "A"}, {"name": "B"}]}, fields)
    merged = merge_annotated(fields, [first, second])
    assert [item["name"].value for item in merged["attendees"]] == ["FSRU", "FSRU", "A", "B"]


def test_layout_only_differences_ground_as_exact():
    grounder = Grounder("The Board agreed the minutes\nof the meeting.")
    assert grounder.find("The Board agreed the minutes of the meeting.").kind == "exact"
    assert grounder.find("the board agreed the minutes of the meeting.").kind == "fuzzy"


def test_item_children_are_anchored_by_their_most_distinctive_quote():
    text = ("1.1 Jane Smith welcomed members.\n" + "Filler. " * 400
            + "\nAnnex A\nJane Smith, Chair of the Finance Committee\n")
    fields = [SchemaField.model_validate({"name": "attendees", "type": "object", "array": True, "fields": [
        {"name": "name", "type": "verbatim"}, {"name": "role", "type": "verbatim"}]})]
    record = {"attendees": [{"name": Leaf("Jane Smith", quote="Jane Smith"),
                             "role": Leaf("Chair of the Finance Committee", quote="Chair of the Finance Committee")}]}
    _, values = ValueBuilder(text).build(fields, record)
    name = next(value for value in values if value.field == "attendees[].name")
    assert name.start == text.rindex("Jane Smith")


def test_unstract_results_file_is_collated_into_a_grounded_run():
    from app import main
    client = TestClient(main.app)
    uploaded = client.post("/api/documents", files={"file": ("u.txt", TEXT.encode(), "text/plain")}).json()
    saved = client.post("/api/schemas", json={"draft": client.get("/api/schemas/starter").json()}).json()
    payload = {"status": "COMPLETED", "message": [{"file": "u.txt", "status": "Success", "error": None,
               "result": {"output": {"meeting_date": "2019-10-22", "chair": "NA",
                                     "attendees": '[{"name": "Simon Dow", "role": "Interim Chair"}]'}},
               "metadata": {"usage": {"prompt_tokens": 10, "completion_tokens": 2}}},
               {"file": "unknown.pdf", "status": "Success", "result": {"output": {}}}]}
    response = client.post("/api/solutions/unstract/import", json={
        "schema_id": saved["id"], "document_ids": [uploaded["id"]], "files": [{"name": "results.json", "content": payload}]})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["documents_done"] == 1 and any("unknown.pdf" in warning for warning in body["warnings"])
    run = client.get(f"/api/runs/{body['id']}").json()
    data = run["documents"][0]["data"]
    assert data["meeting_date"] == "2019-10-22" and data["chair"] is None
    assert data["attendees"][0]["name"] == "Simon Dow"
    assert all(value["grounding"] == "exact" for value in run["documents"][0]["values"])
    empty = client.post("/api/solutions/unstract/import", json={
        "schema_id": saved["id"], "document_ids": [uploaded["id"]], "files": [{"name": "x.json", "content": {"message": []}}]})
    assert empty.status_code == 422


def test_recovery_only_closes_runs_whose_runner_is_gone():
    from app.storage import now
    runs = RunStore(None)
    base = dict(solution="fake", solution_name="Fake", label="t", model="m", confidence_method="c",
                extraction_schema=schema(), settings={}, documents=[
                    DocumentResult(document_id="d", document_name="d.txt", document_sha256="0" * 64, status="running")])
    runs.save(Run(id="live", status="running", created_at=now(), heartbeat_at=now(), **base))
    runs.save(Run(id="dead", status="running", created_at="2026-10-25T00:00:00+00:00",
                  heartbeat_at="2026-10-25T00:00:00+00:00", **base))
    JobRunner(runs, DocumentStore()).recover()
    assert runs.get("live").status == "running"
    dead = runs.get("dead")
    assert dead.status == "error" and dead.documents[0].status == "cancelled"


def test_api_closes_a_run_whose_heartbeat_went_stale_after_startup():
    """A backend that stopped and came back within the stale window: recover() kept the run, but
    once its heartbeat is stale and no runner owns it, reading or listing it closes it."""
    from app import main
    stale = "2026-10-25T00:00:00+00:00"
    main.runs.save(Run(id="orphan1", solution="fake", solution_name="Fake", label="t", model="m",
                       confidence_method="c", extraction_schema=schema(), settings={}, status="running",
                       created_at=stale, started_at=stale, heartbeat_at=stale, documents=[
                           DocumentResult(document_id="d", document_name="d.txt", document_sha256="0" * 64,
                                          status="running")]))
    client = TestClient(main.app)
    listed = next(item for item in client.get("/api/runs").json() if item["id"] == "orphan1")
    assert listed["status"] == "error" and listed["active"] is False
    run = client.get("/api/runs/orphan1").json()
    assert run["status"] == "error" and run["documents"][0]["status"] == "cancelled"
    assert "Interrupted" in run["error"]
    assert client.delete("/api/runs/orphan1").status_code == 200


def test_compare_skips_values_whose_shape_changed_between_schema_versions():
    """chair: verbatim -> object, agenda_items[].decisions: text[] -> object[] in v2."""
    raw = meeting_minutes_schema()
    for field in raw["fields"]:
        if field["name"] == "chair":
            field.update(type="object", fields=[{"name": "name", "type": "verbatim"}, {"name": "role", "type": "verbatim"}])
        if field["name"] == "agenda_items":
            next(child for child in field["fields"] if child["name"] == "decisions").update(
                type="object", fields=[{"name": "text", "type": "text"}])
    v2 = ExtractionSchema(**SchemaDraft.model_validate(raw).model_dump(), id="s1", version=2, created_at="2026-10-26")
    old = run_with("old", {"chair": "Simon Dow", "agenda_items": [{"title": "Finance", "decisions": ["Approve the budget"]}],
                           "meeting_date": "2019-10-22"})
    new = Run(**{**run_with("new", {}).model_dump(), "extraction_schema": v2.model_dump(), "documents": [
        {**run_with("new", {}).documents[0].model_dump(), "data": coerce_record(
            {"chair": {"name": "Simon Dow", "role": "Interim Chair"}, "meeting_date": "2019-10-22",
             "agenda_items": [{"title": "Finance", "decisions": [{"text": "Approve the budget"}]}]}, v2.fields)}]})
    for runs, reference in (([new, old], "new"), ([new, old], None), ([old, new], "old")):
        result = compare_runs(runs, reference)
        cells = {cell["field"]: cell for cell in result["cells"]}
        base, other = runs[0].id, runs[1].id
        for name in ("chair", "agenda_items"):
            assert set(cells[name]["values"]) == {base} and cells[name]["incomparable"] == [other]
        assert cells["meeting_date"]["agreement"][other] == 1.0
        assert any("“chair” has a different shape" in warning for warning in result["warnings"])
        assert any("“agenda_items” has a different shape" in warning for warning in result["warnings"])


def test_item_children_prefer_renderings_near_the_item_and_numbers_stay_local():
    fields = [SchemaField.model_validate({"name": "actions", "type": "object", "array": True, "fields": [
        {"name": "description", "type": "text"}, {"name": "owner", "type": "verbatim"},
        {"name": "due_date", "type": "date"}, {"name": "priority", "type": "integer"},
        {"name": "status", "type": "enum", "options": ["Finance", "Open"]}]})]
    action = "CK to add the workshops to the forward planner by 1st December 2019."
    text = "Date of next meeting: 1 December 2019. Agenda item 5: Finance.\n" + "Filler text. " * 180 + action
    record = annotate_plain({"actions": [{"description": "add the workshops to the forward planner", "owner": "CK",
                                          "due_date": "2019-12-01", "priority": 5, "status": "Finance"}]}, fields)
    _, values = ValueBuilder(text).build(fields, record)
    by_field = {value.field: value for value in values}
    due = by_field["actions[].due_date"]
    assert text[due.start:due.end] == "1st December 2019" and due.start > text.index(action)
    assert by_field["actions[].priority"].grounding == "none"
    assert by_field["actions[].status"].grounding == "none"


def finding(cls, text, start, attributes=None):
    return ExtractionItem(id="0", extraction_class=cls, extraction_text=text, attributes=attributes or {},
                          start=start, end=start + len(text), alignment_status="match_exact", grounding="exact")


def test_langextract_object_with_only_list_children_has_no_text_field():
    fields = [SchemaField.model_validate({"name": "attendance", "type": "object", "fields": [
        {"name": "present", "type": "verbatim", "array": True}, {"name": "apologies", "type": "verbatim", "array": True}]})]
    names = {spec.name: spec for spec in class_plan(fields)}
    assert names["attendance"].text_field is None and names["attendance__present"].parent == "attendance"
    assert "extraction_text = a short identifying phrase" in class_guide(fields)
    record, notes = assemble(fields, [finding("attendance", "Present", TEXT.index("Present")),
                                      finding("attendance__present", "Simon Dow", TEXT.index("Simon Dow")),
                                      finding("attendance__present", "Liz Butler", TEXT.index("Liz Butler"))])
    data, _ = ValueBuilder(TEXT).build(fields, record)
    assert data == {"attendance": {"present": ["Simon Dow", "Liz Butler"], "apologies": []}}


def test_langextract_repeated_single_object_keeps_children_of_every_finding():
    fields = [SchemaField.model_validate({"name": "meeting", "type": "object", "fields": [
        {"name": "title", "type": "verbatim"}, {"name": "attendees", "type": "verbatim", "array": True}]})]
    assert "meeting: at most one extraction" in class_guide(fields)
    record, notes = assemble(fields, [
        finding("meeting", "Board Meeting", 0), finding("meeting__attendees", "Alice Smith", 30),
        finding("meeting", "Board Meeting", 100), finding("meeting__attendees", "Bob Jones", 130),
        finding("meeting__attendees", "Carol White", 160, {"parent": "Board meeting of 3 May"})])
    assert record["meeting"]["title"].value == "Board Meeting"
    assert [leaf.value for leaf in record["meeting"]["attendees"]] == ["Alice Smith", "Bob Jones", "Carol White"]
    assert any("Merged repeated meeting findings" in note for note in notes)


def test_langextract_class_names_never_look_like_attribute_keys():
    from langextract.core.format_handler import FormatHandler
    from langextract.resolver import Resolver
    fields = [SchemaField.model_validate(raw) for raw in (
        {"name": "products", "type": "object", "array": True, "fields": [
            {"name": "name", "type": "verbatim"}, {"name": "attributes", "type": "verbatim", "array": True}]},
        {"name": "key_attributes", "type": "text", "array": True})]
    names = [spec.name for spec in class_plan(fields)]
    assert names == ["products", "products__attributes__1", "key_attributes__1"]
    assert all(name in class_guide(fields) for name in names)
    resolved = Resolver(FormatHandler(use_fences=False)).resolve(
        '{"extractions": [{"products": "Tent"}, {"products__attributes__1": "waterproof"}, '
        '{"key_attributes__1": "lightweight"}]}', suppress_parse_errors=False)
    assert [extraction.extraction_class for extraction in resolved] == names
    record, _ = assemble(fields, [finding("products", "Tent", 0), finding("products__attributes__1", "waterproof", 10),
                                  finding("key_attributes__1", "lightweight", 30)])
    assert [leaf.value for leaf in record["products"][0]["attributes"]] == ["waterproof"]
    assert [leaf.value for leaf in record["key_attributes"]] == ["lightweight"]


def test_langextract_primary_ignores_a_sibling_attribute_named_value():
    fields = [SchemaField.model_validate({"name": "readings", "type": "object", "array": True, "fields": [
        {"name": "date", "type": "date"}, {"name": "value", "type": "number"}]})]
    record, _ = assemble(fields, [finding("readings", "3 May 2024", 0, {"value": 12.5})])
    assert record["readings"][0]["date"].value == "2024-05-03" and record["readings"][0]["value"].value == 12.5
    single = [SchemaField.model_validate({"name": "meeting_date", "type": "date"})]
    record, _ = assemble(single, [finding("meeting_date", "22 October", 0, {"value": "2019-10-22"})])
    assert record["meeting_date"].value == "2019-10-22"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_numeric_settings_reject_non_finite_values(value):
    from app.azure import ServiceError
    from app.solutions.nuextract import NuExtractSolution
    from app.solutions.unstract import UnstractSolution
    for spec in (SettingSpec("t", "Temperature", "number", 0.2, min=0, max=2),
                 SettingSpec("n", "Count", "integer", 1, min=1, max=5)):
        with pytest.raises(ServiceError) as raised:
            spec.validate(value)
        assert raised.value.status == 422
    with pytest.raises(ServiceError):
        NuExtractSolution().validate_settings({"temperature": value})
    with pytest.raises(ServiceError):
        UnstractSolution().validate_settings({"poll_interval": value})
