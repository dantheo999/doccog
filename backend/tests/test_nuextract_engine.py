"""Offline tests for the NuExtract3 engine (no network: httpx.MockTransport)."""
import json
import math

import httpx
import pytest

from app import nuextract_engine as ne


SETTINGS = ne.NuExtractSettings(base_url="http://nuextract.test/v1", timeout=30.0)

MINUTES = [
    {"name": "meeting_date", "type": "date", "description": "Date of the meeting"},
    {"name": "title", "type": "verbatim", "description": ""},
    {"name": "attendees", "type": "object", "array": True, "description": "Everyone listed", "fields": [
        {"name": "name", "type": "verbatim", "description": "Name as written"},
        {"name": "role", "type": "text", "description": "Job title"},
        {"name": "attendance", "type": "enum", "options": ["present", "apologies"]},
        {"name": "contacts", "type": "object", "array": True, "fields": [
            {"name": "email", "type": "text", "description": "Email  address\n(work)"},
        ]},
    ]},
    {"name": "topics", "type": "enum", "array": True, "options": ["finance", "risk", "people"],
     "description": "Topics discussed"},
    {"name": "decisions", "type": "text", "array": True},
    {"name": "quorate", "type": "boolean"},
    {"name": "budget", "type": "number"},
    {"name": "headcount", "type": "integer"},
    {"name": "venue", "type": "object", "fields": [{"name": "room", "type": "verbatim"}]},
]


# --------------------------------------------------------------------------- template / instructions

def test_schema_to_template_maps_types_nesting_enums_and_arrays():
    assert ne.schema_to_template(MINUTES) == {
        "meeting_date": "date",
        "title": "verbatim-string",
        "attendees": [{
            "name": "verbatim-string",
            "role": "string",
            "attendance": ["present", "apologies"],
            "contacts": [{"email": "string"}],
        }],
        "topics": [["finance", "risk", "people"]],
        "decisions": ["string"],
        "quorate": "boolean",
        "budget": "number",
        "headcount": "integer",
        "venue": {"room": "verbatim-string"},
    }


def test_schema_to_template_pads_single_option_enum_and_rejects_bad_fields():
    assert ne.schema_to_template([{"name": "a", "type": "enum", "options": ["yes"]}]) == {"a": ["yes", "other"]}
    assert ne.schema_to_template([{"name": "a", "type": "enum", "options": ["other"], "array": True}]) == {
        "a": [["other", "none"]]}
    with pytest.raises(ValueError, match="unsupported type"):
        ne.schema_to_template([{"name": "a", "type": "colour"}])
    with pytest.raises(ValueError, match="needs at least one child"):
        ne.schema_to_template([{"name": "a", "type": "object", "fields": []}])
    with pytest.raises(ValueError, match="needs options"):
        ne.schema_to_template([{"name": "a", "type": "enum"}])
    with pytest.raises(ValueError, match="Duplicate field name 'o.x'"):
        ne.schema_to_template([{"name": "o", "type": "object", "fields": [{"name": "x"}, {"name": "x"}]}])
    with pytest.raises(ValueError):
        ne.schema_to_template([])


def test_schema_instructions_uses_paths_and_skips_empty_descriptions():
    text = ne.schema_instructions(MINUTES, extra="  Dates are in UK format.  ")
    assert text == (
        "Field guidance:\n"
        "- `meeting_date`: Date of the meeting\n"
        "- `attendees[]`: Everyone listed\n"
        "- `attendees[].name`: Name as written\n"
        "- `attendees[].role`: Job title\n"
        "- `attendees[].contacts[].email`: Email address (work)\n"
        "- `topics`: Topics discussed\n"
        "\n"
        "Dates are in UK format."
    )
    assert ne.schema_instructions([{"name": "a", "type": "text"}]) == ""
    assert ne.schema_instructions([{"name": "a", "type": "text"}], "Only this.") == "Only this."


# --------------------------------------------------------------------------- prompt rendering

def test_render_prompt_matches_chat_template_with_examples_instructions_and_image():
    # Golden string rendered by jinja2 from NuExtract3's chat_template.jinja.
    expected = (
        '<|im_start|>user\n【task】structured\n【template_start】{\n    "names": [\n        "string"\n    ],\n'
        '    "kind": [\n        "a",\n        "b"\n    ]\n}【template_end】\n'
        '【instructions_start】Upper-case names.【instructions_end】\n【examples_start】\n'
        '【example_input_start】Stephen manages Susan\'s store.【example_input_end】\n'
        '【example_output_start】{"names": ["STEPHEN", "SUSAN"], "kind": "a"}【example_output_end】\n'
        '【document_start】\n<|vision_start|><|image_pad|><|vision_end|>\n  John met Mary.\n'
        '【document_end】<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    )
    got = ne.render_prompt(
        {"names": ["string"], "kind": ["a", "b"]}, text="  John met Mary.\n", n_images=1,
        instructions="Upper-case names.",
        examples=[{"text": "Stephen manages Susan's store.", "output": {"names": ["STEPHEN", "SUSAN"], "kind": "a"}}])
    assert got == expected


def test_render_prompt_thinking_mode_and_unicode():
    expected = ('<|im_start|>user\n【task】structured\n【template_start】{\n    "x": "verbatim-string"\n}'
                '【template_end】\n【document_start】\nCafé 東京\n【document_end】<|im_end|>\n'
                '<|im_start|>assistant\n<think>\n')
    assert ne.render_prompt({"x": "verbatim-string"}, text="Café 東京", enable_thinking=True) == expected


def test_render_prompt_uses_media_marker_for_llama_cpp_images():
    prompt = ne.render_prompt({"x": "string"}, text=None, n_images=2, image_token="<__media_X__>")
    assert "【document_start】\n<__media_X__>\n<__media_X__>\n【document_end】" in prompt


# --------------------------------------------------------------------------- span parser

def test_parse_json_with_spans_records_leaf_and_empty_container_spans():
    text = 'x {"a": "q\\"uo", "b": [1, -2.5e3, true], "c": {"d": null, "e": []}, "f": {}, "g": "東京"} tail'
    value, end, spans = ne.parse_json_with_spans(text, 2)
    assert value == {"a": 'q"uo', "b": [1, -2500.0, True], "c": {"d": None, "e": []}, "f": {}, "g": "東京"}
    assert text[end:] == " tail"
    pieces = {path: text[s:e] for path, (s, e) in spans.items()}
    assert pieces == {
        ("a",): '"q\\"uo"',
        ("b", 0): "1", ("b", 1): "-2.5e3", ("b", 2): "true",
        ("c", "d"): "null", ("c", "e"): "[]",
        ("f",): "{}",
        ("g",): '"東京"',
    }


@pytest.mark.parametrize("bad", ['{"a": }', '{"a": 1', '{"a" 1}', '{"a": [1 2]}', '{"a": "x}', "{'a': 1}", ""])
def test_parse_json_with_spans_rejects_malformed_json(bad):
    with pytest.raises(ValueError):
        ne.parse_json_with_spans(bad)


# --------------------------------------------------------------------------- token alignment + confidences

def _llama_logprobs(pieces):
    """llama.cpp-style logprobs.content from [(bytes_or_str, prob)]."""
    content = []
    for piece, prob in pieces:
        raw = piece if isinstance(piece, bytes) else piece.encode()
        content.append({"id": 0, "token": raw.decode("utf-8", errors="replace"), "bytes": list(raw),
                        "logprob": math.log(prob), "top_logprobs": []})
    return {"content": content}


ANSWER_PIECES = [
    ('{"', 1.0), ("name", 1.0), ('":', 1.0), (' "', 0.9), ("Ed", 0.5), (" Smith", 0.98), ('",', 1.0),
    (' "', 1.0), ("due", 1.0), ('":', 1.0), (" null", 0.6), (",", 1.0),
    (' "', 1.0), ("tags", 1.0), ('":', 1.0), (" []", 0.8), (",", 1.0),
    (' "', 1.0), ("n", 1.0), ('":', 1.0), (" ", 1.0), ("1", 0.7), ("2", 0.9), ("}", 1.0),
]


def test_confidences_on_synthetic_token_stream():
    logprobs = _llama_logprobs(ANSWER_PIECES)
    text, tokens = ne.token_spans_from_logprobs(logprobs)
    assert text == '{"name": "Ed Smith", "due": null, "tags": [], "n": 12}'
    _, _, spans = ne.parse_json_with_spans(text)
    geo = ne.leaf_confidences(spans, tokens)
    # "Ed Smith" span ("…") overlaps ' "', 'Ed', ' Smith', '",'
    assert geo[("name",)] == pytest.approx((0.9 * 0.5 * 0.98 * 1.0) ** 0.25)
    assert geo[("due",)] == pytest.approx(0.6)  # probability the model chose null
    assert geo[("tags",)] == pytest.approx(0.8)  # probability of the empty list
    assert geo[("n",)] == pytest.approx((0.7 * 0.9) ** 0.5)
    low = ne.leaf_confidences(spans, tokens, method="min")
    assert low[("name",)] == pytest.approx(0.5)
    joint = ne.leaf_confidences(spans, tokens, method="product")
    assert joint[("name",)] == pytest.approx(0.9 * 0.5 * 0.98)
    with pytest.raises(ValueError):
        ne.leaf_confidences(spans, tokens, method="mean")


def test_token_spans_handle_multibyte_characters_split_across_tokens():
    pieces = [(b'{"city": "Z', 0.9), (b"\xc3", 0.5), (b'\xbcrich"}', 0.8)]
    text, tokens = ne.token_spans_from_logprobs(_llama_logprobs(pieces))
    assert text == '{"city": "Zürich"}'
    assert [(t.start, t.end) for t in tokens] == [(0, 11), (11, 12), (11, 18)]
    _, _, spans = ne.parse_json_with_spans(text)
    conf = ne.leaf_confidences(spans, tokens)
    assert conf[("city",)] == pytest.approx((0.9 * 0.5 * 0.8) ** (1 / 3))


def test_token_spans_from_legacy_completions_logprobs():
    legacy = {"tokens": ['{"', "a", '":', " 1", "}"], "token_logprobs": [0.0, 0.0, 0.0, math.log(0.25), 0.0],
              "text_offset": [0, 2, 3, 5, 7], "top_logprobs": None}
    text, tokens = ne.token_spans_from_logprobs(legacy, '{"a": 1}')
    _, _, spans = ne.parse_json_with_spans(text)
    assert ne.leaf_confidences(spans, tokens) == {("a",): pytest.approx(0.25)}
    # token strings that don't reproduce the text fall back to text_offset
    legacy["tokens"] = ['{"', "a", '":', " �", "}"]
    text, tokens = ne.token_spans_from_logprobs(legacy, '{"a": 1}')
    assert text == '{"a": 1}'
    assert ne.leaf_confidences(spans, tokens) == {("a",): pytest.approx(0.25)}
    assert ne.token_spans_from_logprobs(None) is None
    assert ne.token_spans_from_logprobs({"content": [{"token": "x", "bytes": [255], "logprob": 0}]}) is None


def test_fences_and_thinking_are_removed():
    assert ne.strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert ne.strip_fences('{"a": 1}') == '{"a": 1}'
    assert ne.locate_json('```json\n{"a": 1}\n```') == 8
    reasoning, answer, offset = ne.split_thinking('Look at page 1.\n</think>\n\n{"a": 1}')
    assert reasoning == "Look at page 1."
    assert answer == '\n\n{"a": 1}'
    assert offset == len("Look at page 1.\n</think>")
    assert ne.split_thinking('<think>\nhmm\n</think>{"a": 1}')[0] == "hmm"
    assert ne.split_thinking('{"a": 1}') == (None, '{"a": 1}', 0)


# --------------------------------------------------------------------------- HTTP: fake server

class FakeServer:
    """Records requests and answers them from a handler table."""

    def __init__(self, routes):
        self.routes = routes
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, body))
        handler = self.routes.get((request.method, request.url.path))
        if handler is None:
            return httpx.Response(404, json={"error": {"code": 404, "message": "File Not Found"}})
        return handler(request, body) if callable(handler) else handler


@pytest.fixture
def fake(monkeypatch):
    def install(routes):
        server = FakeServer(routes)

        def make_client(settings, timeout=None):
            return httpx.Client(base_url=settings.base_url.rstrip("/") + "/",
                                transport=httpx.MockTransport(server))
        monkeypatch.setattr(ne, "_make_client", make_client)
        return server
    return install


def completion(text, *, logprobs=None, finish="stop", usage=(10, 5)):
    return httpx.Response(200, json={
        "choices": [{"text": text, "index": 0, "logprobs": logprobs, "finish_reason": finish}],
        "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]},
        "timings": {"prompt_per_second": 30.0, "predicted_per_second": 4.5},
    })


TEMPLATE = {"name": "verbatim-string", "due": "date", "tags": ["string"], "n": "integer"}
ANSWER = '{"name": "Ed Smith", "due": null, "tags": [], "n": 12}'


def test_extract_completions_path_parses_answer_and_scores_leaves(fake):
    server = fake({("POST", "/v1/completions"): completion(ANSWER, logprobs=_llama_logprobs(ANSWER_PIECES))})
    out = ne.extract(SETTINGS, text="Ed Smith attended.", template=TEMPLATE, instructions="Be exact.",
                     examples=[{"text": "Jo attended.", "output": {"name": "Jo", "due": None, "tags": [], "n": 1}}])
    assert out.data == {"name": "Ed Smith", "due": None, "tags": [], "n": 12}
    assert out.raw == ANSWER and out.reasoning is None and out.finish_reason == "stop"
    assert out.usage == {"prompt_tokens": 10, "completion_tokens": 5}
    assert out.transport == "completions" and out.timings["predicted_per_second"] == 4.5
    assert set(out.confidences) == {("name",), ("due",), ("tags",), ("n",)}
    assert out.confidences[("due",)] == pytest.approx(0.6)
    method, path, body = server.requests[-1]
    assert body["logprobs"] == 1 and body["temperature"] == 0.2 and body["max_tokens"] == 4096
    assert body["prompt"] == ne.render_prompt(TEMPLATE, text="Ed Smith attended.", instructions="Be exact.",
                                              examples=[{"text": "Jo attended.",
                                                         "output": {"name": "Jo", "due": None, "tags": [], "n": 1}}])
    assert '【example_output_start】{"name": "Jo", "due": null, "tags": [], "n": 1}【example_output_end】' in body["prompt"]


def test_extract_tolerates_fences_and_keeps_confidence_alignment(fake):
    pieces = [("```", 1.0), ("json", 1.0), ("\n", 1.0)] + ANSWER_PIECES + [("\n```", 1.0)]
    fenced = "```json\n" + ANSWER + "\n```"
    fake({("POST", "/v1/completions"): completion(fenced, logprobs=_llama_logprobs(pieces))})
    out = ne.extract(SETTINGS, text="doc", template=TEMPLATE)
    assert out.data["name"] == "Ed Smith"
    assert out.confidences[("name",)] == pytest.approx((0.9 * 0.5 * 0.98) ** 0.25)


def test_extract_thinking_mode_splits_reasoning(fake):
    reasoning = "The attendee is Ed Smith.\n"
    pieces = [(reasoning, 0.9), ("</think>", 1.0), ("\n\n", 1.0)] + ANSWER_PIECES
    server = fake({("POST", "/v1/completions"): completion(
        reasoning + "</think>\n\n" + ANSWER, logprobs=_llama_logprobs(pieces))})
    out = ne.extract(SETTINGS, text="doc", template=TEMPLATE, enable_thinking=True, temperature=0.6)
    assert out.reasoning == "The attendee is Ed Smith."
    assert out.raw == ANSWER and out.data["n"] == 12
    assert out.confidences[("tags",)] == pytest.approx(0.8)
    assert server.requests[-1][2]["prompt"].endswith("<|im_start|>assistant\n<think>\n")


def test_extract_chat_path_uses_reasoning_content(fake):
    server = fake({("POST", "/v1/chat/completions"): httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": ANSWER, "reasoning_content": "Thought."},
                     "finish_reason": "stop", "logprobs": None}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4}})})
    out = ne.extract(SETTINGS, text="doc", template=TEMPLATE, transport="chat", enable_thinking=True,
                     examples=[{"text": "ex", "output": {"name": "X"}}])
    assert out.reasoning == "Thought." and out.data["name"] == "Ed Smith" and out.confidences == {}
    body = server.requests[-1][2]
    assert body["chat_template_kwargs"] == {"template": json.dumps(TEMPLATE, indent=4), "instructions": "",
                                            "enable_thinking": True}
    assert body["messages"][0] == {"role": "developer", "content": [
        {"type": "text", "text": "ex"}, {"type": "text", "text": '{"name": "X"}'}]}
    assert body["messages"][-1] == {"role": "user", "content": [{"type": "text", "text": "doc"}]}
    assert body["logprobs"] is True


def test_extract_images_on_llama_cpp_use_native_endpoint_media_marker_and_multimodal_data(fake):
    native = {"content": '{"n": 1}', "stop_type": "eos", "tokens_evaluated": 900, "tokens_predicted": 6,
              "timings": {"prompt_per_second": 20.0},
              "completion_probabilities": _llama_logprobs([('{"', 1.0), ("n", 1.0), ('":', 1.0), (" ", 1.0),
                                                           ("1", 0.64), ("}", 1.0)])["content"]}
    server = fake({
        ("GET", "/props"): httpx.Response(200, json={"default_generation_settings": {},
                                                     "modalities": {"vision": True}, "media_marker": "<__m__>"}),
        ("POST", "/completions"): httpx.Response(200, json=native),
    })
    out = ne.extract(SETTINGS, images=[b"\x89PNG fake"], template={"n": "integer"}, max_tokens=100)
    assert out.data == {"n": 1} and out.finish_reason == "stop" and out.transport == "completions"
    assert out.confidences == {("n",): pytest.approx(0.64)}
    assert out.usage == {"prompt_tokens": 900, "completion_tokens": 6}
    method, path, body = server.requests[-1]
    assert path == "/completions" and body["n_predict"] == 100 and body["n_probs"] == 1
    assert body["prompt"]["multimodal_data"] == ["iVBORyBmYWtl"]
    assert "【document_start】\n<__m__>\n【document_end】" in body["prompt"]["prompt_string"]


def test_native_limit_stop_maps_to_truncation(fake):
    fake({("GET", "/props"): httpx.Response(200, json={"default_generation_settings": {},
                                                       "modalities": {"vision": True}}),
          ("POST", "/completions"): httpx.Response(200, json={"content": '{"n": ', "stop_type": "limit"})})
    with pytest.raises(ne.NuExtractError) as info:
        ne.extract(SETTINGS, images=[b"x"], template={"n": "integer"})
    assert info.value.kind == "truncated"


def test_extract_images_without_llama_cpp_fall_back_to_chat(fake):
    server = fake({("POST", "/v1/chat/completions"): httpx.Response(200, json={
        "choices": [{"message": {"content": '{"n": 2}'}, "finish_reason": "stop"}]})})
    out = ne.extract(SETTINGS, text="caption", images=[b"\xff\xd8jpeg"], template={"n": "integer"})
    assert out.transport == "chat" and out.data == {"n": 2}
    content = server.requests[-1][2]["messages"][-1]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1] == {"type": "text", "text": "caption"}


def test_extract_rejects_images_when_llama_cpp_has_no_projector(fake):
    fake({("GET", "/props"): httpx.Response(200, json={"default_generation_settings": {},
                                                       "modalities": {"vision": False}})})
    with pytest.raises(ne.NuExtractError, match="vision projector") as info:
        ne.extract(SETTINGS, images=[b"x"], template={"n": "integer"})
    assert info.value.kind == "invalid_input"


# --------------------------------------------------------------------------- error mapping

def test_truncated_answer_raises_unless_not_strict(fake):
    fake({("POST", "/v1/completions"): completion('{"name": "Ed', finish="length")})
    with pytest.raises(ne.NuExtractError, match="max_tokens limit \\(4096\\)") as info:
        ne.extract(SETTINGS, text="doc", template=TEMPLATE)
    assert info.value.kind == "truncated" and info.value.raw == '{"name": "Ed'
    out = ne.extract(SETTINGS, text="doc", template=TEMPLATE, strict=False)
    assert out.data is None and out.finish_reason == "length" and out.raw == '{"name": "Ed'


def test_truncated_while_thinking_says_so(fake):
    fake({("POST", "/v1/completions"): completion("Let me think about", finish="length")})
    with pytest.raises(ne.NuExtractError, match="while still reasoning"):
        ne.extract(SETTINGS, text="doc", template=TEMPLATE, enable_thinking=True)


def test_non_json_answer_raises_invalid_json(fake):
    fake({("POST", "/v1/completions"): completion("I could not find anything.")})
    with pytest.raises(ne.NuExtractError, match="not a valid JSON object") as info:
        ne.extract(SETTINGS, text="doc", template=TEMPLATE)
    assert info.value.kind == "invalid_json"
    fake({("POST", "/v1/completions"): completion('["a list"]')})
    assert ne.extract(SETTINGS, text="doc", template=TEMPLATE, strict=False).data is None


@pytest.mark.parametrize("response, kind, text", [
    (httpx.Response(503, json={"error": {"code": 503, "message": "Loading model"}}), "unavailable", "loading"),
    (httpx.Response(400, json={"error": {"code": 400, "type": "exceed_context_size_error", "n_prompt_tokens": 40000,
                                         "n_ctx": 32768, "message": "the request exceeds the available context size"}}),
     "context_length", "40000 prompt tokens, context 32768"),
    (httpx.Response(400, json={"object": "error", "message": "This model's maximum context length is 16384 tokens."}),
     "context_length", "too long"),
    (httpx.Response(400, json={"error": {"message": "Unsupported param: best_of"}}), "bad_request", "best_of"),
    (httpx.Response(401, json={"error": {"message": "Invalid API Key"}}), "auth", "API key"),
    (httpx.Response(500, text="boom"), "server", "HTTP 500"),
    (httpx.Response(200, text="<html>"), "protocol", "not JSON"),
    (httpx.Response(200, json={"choices": []}), "protocol", "no completion choices"),
])
def test_http_errors_are_mapped_to_friendly_messages(fake, response, kind, text):
    fake({("POST", "/v1/completions"): response})
    with pytest.raises(ne.NuExtractError) as info:
        ne.extract(SETTINGS, text="doc", template=TEMPLATE)
    assert info.value.kind == kind
    assert text in str(info.value)


def test_missing_endpoint_and_transport_errors(fake, monkeypatch):
    fake({})
    with pytest.raises(ne.NuExtractError, match="should end in /v1") as info:
        ne.extract(SETTINGS, text="doc", template=TEMPLATE)
    assert info.value.kind == "not_found"

    def raiser(exc):
        def handler(request):
            raise exc
        return handler

    for exc, kind, text in [(httpx.ConnectError("refused"), "connection", "Is it running?"),
                            (httpx.ConnectTimeout("slow"), "connection", "Could not connect"),
                            (httpx.ReadTimeout("slow"), "timeout", "did not finish within 30 s")]:
        monkeypatch.setattr(ne, "_make_client", lambda settings, timeout=None, h=raiser(exc): httpx.Client(
            base_url=settings.base_url + "/", transport=httpx.MockTransport(h)))
        with pytest.raises(ne.NuExtractError, match=text) as info:
            ne.extract(SETTINGS, text="doc", template=TEMPLATE)
        assert info.value.kind == kind


def test_invalid_inputs_raise_before_any_request(fake):
    server = fake({})
    with pytest.raises(ne.NuExtractError, match="Nothing to extract"):
        ne.extract(SETTINGS, template=TEMPLATE)
    with pytest.raises(ne.NuExtractError, match="template"):
        ne.extract(SETTINGS, text="doc", template={})
    with pytest.raises(ne.NuExtractError, match="transport"):
        ne.extract(SETTINGS, text="doc", template=TEMPLATE, transport="grpc")
    assert server.requests == []


# --------------------------------------------------------------------------- server_status

def test_server_status_reports_llama_cpp_model(fake):
    fake({("GET", "/v1/models"): httpx.Response(200, json={"object": "list", "data": [
        {"id": "numind/NuExtract3", "owned_by": "llamacpp", "meta": {"n_ctx": 32768}}]})})
    assert ne.server_status(SETTINGS) == {"available": True, "model": "numind/NuExtract3", "backend": "llama.cpp",
                                          "detail": "llama.cpp server ready, context 32768 tokens."}


def test_server_status_flags_other_models_and_vllm(fake):
    fake({("GET", "/v1/models"): httpx.Response(200, json={"data": [
        {"id": "Qwen/Qwen3-8B", "owned_by": "vllm", "max_model_len": 16384}]})})
    status = ne.server_status(SETTINGS)
    assert status["available"] and status["backend"] == "vllm"
    assert "does not look like NuExtract3" in status["detail"]


@pytest.mark.parametrize("response, text", [
    (httpx.Response(503, json={"error": {"message": "Loading model"}}), "loading the model"),
    (httpx.Response(200, json={"data": []}), "lists no models"),
    (httpx.Response(500, text="x"), "HTTP 500"),
])
def test_server_status_unavailable_states(fake, response, text):
    fake({("GET", "/v1/models"): response})
    status = ne.server_status(SETTINGS)
    assert status["available"] is False and status["model"] is None and text in status["detail"]


def test_server_status_never_raises(monkeypatch):
    def broken(settings, timeout=None):
        assert timeout == ne.STATUS_TIMEOUT

        def handler(request):
            raise httpx.ConnectError("refused")
        return httpx.Client(base_url=settings.base_url + "/", transport=httpx.MockTransport(handler))
    monkeypatch.setattr(ne, "_make_client", broken)
    status = ne.server_status(SETTINGS)
    assert status["available"] is False and "Cannot reach" in status["detail"]

    monkeypatch.setattr(ne, "_make_client", lambda settings, timeout=None: (_ for _ in ()).throw(RuntimeError()))
    assert ne.server_status(SETTINGS)["available"] is False


# --------------------------------------------------------------------------- template generation

def test_render_template_generation_prompt_matches_chat_template():
    # Golden strings rendered by jinja2 with mode="template-generation".
    assert ne.render_template_generation_prompt(
        description="I want to extract the key details from a rental contract.") == (
        "<|im_start|>user\n【task】template generation\n【document_start】\n"
        "I want to extract the key details from a rental contract.\n【document_end】<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert ne.render_template_generation_prompt(description="Describe: minutes", text="DOC TEXT") == (
        "<|im_start|>user\n【task】template generation\n【document_start】\nDescribe: minutes\nDOC TEXT\n"
        "【document_end】<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert ne.render_template_generation_prompt(text="doc", enable_thinking=True).endswith("assistant\n<think>\n")


GENERATED = '{\n    "Meeting": {\n        "Date": "date-time",\n        "Attendees": [\n            {"Name": "verbatim-string"}\n        ]\n    }\n}'


def test_generate_template_parses_answer_and_sends_template_generation_prompt(fake):
    server = fake({("POST", "/v1/completions"): completion("```json\n" + GENERATED + "\n```", usage=(48, 30))})
    out = ne.generate_template(SETTINGS, text="Minutes of the board.", description="meeting minutes")
    assert out.template == {"Meeting": {"Date": "date-time", "Attendees": [{"Name": "verbatim-string"}]}}
    assert out.raw.startswith("```json") and out.reasoning is None and out.finish_reason == "stop"
    assert out.usage == {"prompt_tokens": 48, "completion_tokens": 30}
    body = server.requests[-1][2]
    assert body["prompt"] == ne.render_template_generation_prompt(text="Minutes of the board.",
                                                                  description="meeting minutes")
    assert "【task】template generation" in body["prompt"] and body["temperature"] == 0.2


def test_generate_template_thinking_and_errors(fake):
    fake({("POST", "/v1/completions"): completion("Fields needed.\n</think>\n\n" + GENERATED)})
    out = ne.generate_template(SETTINGS, description="minutes", enable_thinking=True)
    assert out.reasoning == "Fields needed." and out.template["Meeting"]["Date"] == "date-time"

    with pytest.raises(ne.NuExtractError, match="description and/or an exemplar") as info:
        ne.generate_template(SETTINGS, text="  ", description="")
    assert info.value.kind == "invalid_input"

    fake({("POST", "/v1/completions"): completion('{"Meeting": {"Date": "da', finish="length")})
    with pytest.raises(ne.NuExtractError, match="max_tokens limit") as info:
        ne.generate_template(SETTINGS, description="minutes", max_tokens=64)
    assert info.value.kind == "truncated"
    assert ne.generate_template(SETTINGS, description="minutes", strict=False).template is None

    for answer in ("Sorry, no template.", "{}"):
        fake({("POST", "/v1/completions"): completion(answer)})
        with pytest.raises(ne.NuExtractError, match="not a valid JSON object") as info:
            ne.generate_template(SETTINGS, description="minutes")
        assert info.value.kind == "invalid_json"


# --------------------------------------------------------------------------- template -> app fields

def _f(name, ftype, *, array=False, description="", options=(), fields=()):
    return {"name": name, "type": ftype, "description": description, "array": array,
            "options": list(options), "fields": list(fields)}


def test_template_to_fields_maps_types_and_constructors():
    template = {
        "Title": "verbatim-string", "Summary": "string", "Date": "date", "Starts": "date-time",
        "Total": "number", "Count": "integer", "Approved": "boolean",
        "Status": ["open", "closed"], "Tags": [["a", "b"]], "Emails": ["email-address"],
        "Currency": "currency", "Region": "region:GB", "Start time": "time", "Odd": "colour",
        "Items": [{"Name": "verbatim-string", "Qty": "integer"}], "Venue": {"Room": "verbatim-string"},
    }
    assert ne.template_to_fields(template) == [
        _f("title", "verbatim"), _f("summary", "text"), _f("date", "date"), _f("starts", "date"),
        _f("total", "number"), _f("count", "integer"), _f("approved", "boolean"),
        _f("status", "enum", options=["open", "closed"]),
        _f("tags", "enum", array=True, options=["a", "b"]),
        _f("emails", "verbatim", array=True, description="email address"),
        _f("currency", "text", description="currency (ISO 4217 code)"),
        _f("region", "text", description="region (ISO 3166-2:GB subdivision code)"),
        _f("start_time", "text", description="time (ISO 8601, e.g. 14:30)"),
        _f("odd", "text", description="type: colour"),
        _f("items", "object", array=True, fields=[_f("name", "verbatim"), _f("qty", "integer")]),
        _f("venue", "object", fields=[_f("room", "verbatim")]),
    ]


def test_template_to_fields_sanitises_and_dedupes_names():
    fields = ne.template_to_fields({"Invoice Number": "string", "invoiceNumber": "string", "HTTPServer": "url",
                                    "Montant TTC (€)": "number", "2nd party": "string", "!!!": "string",
                                    "x" * 80: "string", "Café": "string"})
    names = [f["name"] for f in fields]
    assert names == ["invoice_number", "invoice_number_2", "http_server", "montant_ttc", "f_2nd_party",
                     "field", "x" * 64, "cafe"]
    assert all(ne._NAME_RE.match(n) for n in names)


def test_template_to_fields_unwraps_root_caps_depth_and_drops_empty_objects():
    template = {"Meeting": {
        "Agenda Items": [{"Item": "verbatim-string",
                          "Action": {"Owner": "verbatim-string", "Due": {"Date": "date"}},
                          "Votes": [{"Member": "string"}]}],
        "Empty": {}, "Empty list": [{}], "Notes": ["string"],
    }}
    assert ne.template_to_fields(template) == [
        _f("agenda_items", "object", array=True, fields=[
            _f("item", "verbatim"),
            _f("action", "object", fields=[_f("owner", "verbatim"), _f("due_date", "date")]),
            _f("votes", "object", array=True, fields=[_f("member", "text")]),
        ]),
        _f("notes", "text", array=True),
    ]
    # without unwrapping, the root counts as a level: deeper objects flatten / become text
    wrapped = ne.template_to_fields(template, unwrap_root=False)
    agenda = wrapped[0]["fields"][0]
    assert [f["name"] for f in agenda["fields"]] == ["item", "action_owner", "action_due_date", "votes"]
    assert agenda["fields"][3] == _f("votes", "text", array=True, description="summary of: Member")


def test_template_to_fields_cleans_enum_options():
    fields = ne.template_to_fields({"a": ["Yes", "yes", " No ", ""], "b": [[" x ", "X", None]], "c": [1, 2],
                                    "d": ["", ""]})
    assert fields == [_f("a", "enum", options=["Yes", "No"]), _f("b", "enum", array=True, options=["x"]),
                      _f("c", "enum", options=["1", "2"]), _f("d", "text")]
    with pytest.raises(ValueError):
        ne.template_to_fields(["not", "a", "dict"])


def test_template_round_trips_with_schema_to_template():
    template = {"title": "verbatim-string", "date": "date", "people": [{"name": "verbatim-string",
                "attendance": ["present", "apologies"], "notes": ["string"]}], "topics": [["a", "b", "c"]],
                "budget": "number", "venue": {"room": "verbatim-string"}}
    assert ne.schema_to_template(ne.template_to_fields(template)) == template
    plain = [{**f, "description": ""} for f in MINUTES]
    for f in plain:
        f["fields"] = [{**c, "description": ""} for c in f.get("fields", [])]
    assert ne.schema_to_template(ne.template_to_fields(ne.schema_to_template(plain))) == ne.schema_to_template(plain)
