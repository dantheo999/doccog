"""NuExtract3 structured-extraction engine.

Self-contained client for ``numind/NuExtract3`` served by an OpenAI-compatible
server: llama.cpp ``llama-server`` (CPU, GGUF) or vLLM (GPU). It imports only the
standard library and ``httpx``.

Request path
------------
The prompt is rendered here, in pure Python, exactly as NuExtract3's
``chat_template.jinja`` renders it, and sent to ``POST /v1/completions``. This is
the default for every request because:

* llama.cpp rewrites ``developer`` chat messages to ``system`` for every template
  except GPT-OSS, so in-context examples sent through ``/v1/chat/completions``
  are silently turned into a system prompt (one example) or rejected with HTTP 400
  (two or more examples). Rendering the prompt ourselves avoids the server's
  chat-template handling entirely.
* The rendered prompt is byte-identical on llama.cpp and vLLM, and both return
  per-token logprobs from ``/v1/completions``.

Images on llama.cpp use the same self-rendered prompt, sent to llama.cpp's native
``POST /completions`` (``{"prompt_string", "multimodal_data"}`` with the server's
media marker from ``GET /props``): its ``/v1/completions`` only accepts a plain
string prompt once a vision projector is loaded. vLLM's completions endpoint does
not accept images, so for vLLM (or any non-llama.cpp server) image requests go to
``/v1/chat/completions`` with ``chat_template_kwargs`` and ``developer`` messages,
which vLLM renders with the model's own template.

Template generation
-------------------
:func:`generate_template` uses the chat template's ``template-generation`` mode
(same self-rendered prompt + ``/v1/completions`` path). That mode has one input
slot, so a natural-language description and exemplar document(s) are
concatenated (description first). :func:`template_to_fields` turns the result
into the app's schema fields.

Confidence scores
-----------------
Each generated token's logprob is mapped onto a character span of the generated
text (via the token's UTF-8 bytes). The answer JSON is parsed with a small
position-tracking parser that records the character span of every leaf value
(strings including their quotes, numbers, booleans, ``null``; empty ``[]``/``{}``
are recorded as leaves too). A leaf's confidence is the geometric mean of the
probabilities of the tokens overlapping its span, i.e. ``exp(mean(logprob))``.
The span includes the opening-quote token (which decides string-vs-null) and the
token that closes the value, so a ``null`` leaf gets ``P(null)`` and an empty
array gets the probability of ``[]``. Probabilities are the server's raw model
probabilities (before temperature), which both llama.cpp and vLLM report by
default. ``method="min"`` (weakest token) and ``method="product"`` (joint
probability) are available through :func:`leaf_confidences`.
"""
from __future__ import annotations

import base64
import bisect
import json
import math
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable

import httpx

__all__ = [
    "NuExtractError",
    "NuExtractOutput",
    "NuExtractSettings",
    "TemplateOutput",
    "extract",
    "generate_template",
    "leaf_confidences",
    "parse_json_with_spans",
    "render_prompt",
    "render_template_generation_prompt",
    "schema_instructions",
    "schema_to_template",
    "server_status",
    "template_to_fields",
]

VISION_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
DEFAULT_LLAMA_MEDIA_MARKER = "<__media__>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
STATUS_TIMEOUT = 3.0

# The model card's ICL layout puts developer (example) messages before the user
# message. With that ordering the official chat template never emits the closing
# examples marker (its `loop.last` check refers to the whole message list), so the
# prompt goes straight from the last example to 【document_start】. We reproduce that
# exactly so that this renderer, vLLM's chat path and the model card agree.
EMIT_EXAMPLES_END = False


# --------------------------------------------------------------------------- #
# Public data types
# --------------------------------------------------------------------------- #

class NuExtractError(Exception):
    """User-facing failure talking to NuExtract or interpreting its answer.

    ``kind`` is one of: ``connection``, ``timeout``, ``unavailable``, ``auth``,
    ``not_found``, ``context_length``, ``bad_request``, ``server``, ``protocol``,
    ``truncated``, ``invalid_json``, ``invalid_input``.
    """

    def __init__(self, message: str, *, kind: str = "server", status_code: int | None = None,
                 raw: str | None = None, detail: str | None = None):
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.status_code = status_code
        self.raw = raw
        self.detail = detail

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class NuExtractSettings:
    base_url: str  # OpenAI-compatible base URL, e.g. "http://127.0.0.1:8090/v1"
    model: str = "numind/NuExtract3"  # served model name (llama.cpp accepts any)
    api_key: str = "EMPTY"
    timeout: float = 900.0


@dataclass
class NuExtractOutput:
    data: dict | None  # parsed JSON answer (None if unparseable and strict=False)
    raw: str  # answer text, thinking removed
    reasoning: str | None
    confidences: dict[tuple, float]  # leaf path -> confidence in [0, 1]
    finish_reason: str | None
    usage: dict  # {"prompt_tokens": int, "completion_tokens": int}
    elapsed_seconds: float
    timings: dict = field(default_factory=dict)  # llama.cpp speed figures, when reported
    transport: str = "completions"  # "completions" or "chat"


# --------------------------------------------------------------------------- #
# Schema -> NuExtract template / instructions
# --------------------------------------------------------------------------- #

_SCALAR_TYPES = {
    "text": "string",
    "verbatim": "verbatim-string",
    # TYPES.md: "date" is an ISO 8601 date (reduced accuracy allowed, e.g. 2024-01);
    # "date-time" would invite times the app's date fields do not want.
    "date": "date",
    "number": "number",
    "integer": "integer",
    "boolean": "boolean",
}


def schema_to_template(fields: list[dict]) -> dict:
    """Convert the app's schema fields into a NuExtract template.

    text -> "string", verbatim -> "verbatim-string", date -> "date", number,
    integer, boolean unchanged; enum -> ["a", "b"]; enum + array -> [["a", "b"]];
    scalar + array -> ["string"]; object -> {...}; object + array -> [{...}].

    NuExtract reads a one-item list as "array of <type>", so a single-option enum
    cannot be expressed directly; it is padded with "other" (or "none" when the
    option itself is "other") to keep it an enum.
    """
    if not isinstance(fields, (list, tuple)) or not fields:
        raise ValueError("A schema needs at least one field.")
    return _fields_to_template(fields, path="")


def _fields_to_template(fields: Iterable[dict], path: str) -> dict:
    out: dict[str, Any] = {}
    for f in fields:
        name = f.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Every field needs a name (under '{path or 'root'}').")
        if name in out:
            raise ValueError(f"Duplicate field name '{_join(path, name)}'.")
        out[name] = _field_to_template(f, _join(path, name))
    return out


def _field_to_template(f: dict, path: str) -> Any:
    ftype = f.get("type", "text")
    is_array = bool(f.get("array", False))
    if ftype == "object":
        children = f.get("fields") or []
        if not children:
            raise ValueError(f"Object field '{path}' needs at least one child field.")
        node: Any = _fields_to_template(children, path + ("[]" if is_array else ""))
        return [node] if is_array else node
    if ftype == "enum":
        options = _dedupe([str(o) for o in (f.get("options") or [])])
        if not options:
            raise ValueError(f"Enum field '{path}' needs options.")
        if len(options) == 1:
            options.append("none" if options[0].casefold() == "other" else "other")
        return [options] if is_array else options
    if ftype not in _SCALAR_TYPES:
        raise ValueError(f"Field '{path}' has unsupported type '{ftype}'.")
    leaf = _SCALAR_TYPES[ftype]
    return [leaf] if is_array else leaf


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _join(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name


def schema_instructions(fields: list[dict], extra: str = "") -> str:
    """Field descriptions as concise instruction lines, e.g. ``- `attendees[].role`: Job title``.

    Fields without a description are omitted; ``extra`` (free text) is appended.
    Returns "" when there is nothing to say.
    """
    lines: list[str] = []
    _describe(fields, "", lines)
    parts = []
    if lines:
        parts.append("Field guidance:\n" + "\n".join(lines))
    if extra and extra.strip():
        parts.append(extra.strip())
    return "\n\n".join(parts)


def _describe(fields: Iterable[dict], prefix: str, lines: list[str]) -> None:
    for f in fields:
        name = f.get("name", "")
        is_array = bool(f.get("array", False))
        path = _join(prefix, name)
        shown = path + ("[]" if is_array and f.get("type") == "object" else "")
        description = " ".join(str(f.get("description") or "").split())
        if description:
            lines.append(f"- `{shown}`: {description}")
        if f.get("type") == "object":
            _describe(f.get("fields") or [], path + ("[]" if is_array else ""), lines)


# --------------------------------------------------------------------------- #
# Prompt rendering (pure-Python port of chat_template.jinja, structured mode)
# --------------------------------------------------------------------------- #

def _template_text(template: dict | str) -> str:
    if isinstance(template, str):
        return template
    return json.dumps(template, indent=4, ensure_ascii=False)


def _example_output_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    # NuExtract answers in compact single-line JSON; examples use the same style.
    return json.dumps(output, ensure_ascii=False)


def _render_user_content(text: str | None, n_images: int, image_token: str) -> str:
    # Images first, then the text: each item followed by "\n", the whole trimmed.
    rendered = "".join(image_token + "\n" for _ in range(n_images))
    if text is not None:
        rendered += text + "\n"
    return rendered.strip()


def render_prompt(template: dict | str, *, text: str | None = None, n_images: int = 0,
                  instructions: str = "", examples: Iterable[dict] = (), enable_thinking: bool = False,
                  image_token: str = VISION_PLACEHOLDER) -> str:
    """Render the full NuExtract3 structured-extraction prompt.

    Identical to ``chat_template.jinja`` with ``add_generation_prompt=True`` for the
    messages ``[developer(example)..., user([image]*n_images + [text])]`` and kwargs
    ``template``/``instructions``/``enable_thinking``. ``image_token`` lets llama.cpp's
    media marker stand in for the vision placeholder (mtmd adds the vision start/end
    tokens itself).
    """
    parts = ["<|im_start|>user\n", "【task】structured\n",
             "【template_start】", _template_text(template), "【template_end】\n"]
    if instructions:
        parts += ["【instructions_start】", instructions, "【instructions_end】\n"]
    examples = list(examples)
    if examples:
        parts.append("【examples_start】\n")
        for ex in examples:
            ex_input = (str(ex.get("text") or "") + "\n").strip()
            parts += ["【example_input_start】", ex_input, "【example_input_end】\n",
                      "【example_output_start】", _example_output_text(ex.get("output")).strip(),
                      "【example_output_end】\n"]
        if EMIT_EXAMPLES_END:
            parts.append("【examples_end】\n")
    parts += ["【document_start】\n", _render_user_content(text, n_images, image_token), "\n",
              "【document_end】<|im_end|>\n", "<|im_start|>assistant\n",
              "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"]
    return "".join(parts)


def _chat_messages(text: str | None, images: list[bytes], examples: list[dict]) -> list[dict]:
    messages: list[dict] = []
    for ex in examples:
        messages.append({"role": "developer", "content": [
            {"type": "text", "text": str(ex.get("text") or "")},
            {"type": "text", "text": _example_output_text(ex.get("output"))},
        ]})
    content: list[dict] = [{"type": "image_url", "image_url": {"url": _data_url(img)}} for img in images]
    if text is not None:
        content.append({"type": "text", "text": text})
    messages.append({"role": "user", "content": content})
    return messages


def _image_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/png"


def _data_url(data: bytes) -> str:
    return f"data:{_image_mime(data)};base64,{base64.b64encode(data).decode('ascii')}"


# --------------------------------------------------------------------------- #
# Answer post-processing
# --------------------------------------------------------------------------- #

def split_thinking(text: str) -> tuple[str | None, str, int]:
    """Split generated text into (reasoning, answer, answer_offset).

    The prompt already opens ``<think>`` in thinking mode, so the generation is
    ``reasoning</think>\\n\\nanswer``; a stray leading ``<think>`` is tolerated too.
    """
    idx = text.rfind(THINK_CLOSE)
    if idx < 0:
        return None, text, 0
    reasoning = text[:idx]
    if reasoning.lstrip().startswith(THINK_OPEN):
        reasoning = reasoning.lstrip()[len(THINK_OPEN):]
    return reasoning.strip() or None, text[idx + len(THINK_CLOSE):], idx + len(THINK_CLOSE)


_FENCE_RE = re.compile(r"\s*```[A-Za-z0-9_-]*[ \t]*\r?\n?")


def locate_json(text: str) -> int | None:
    """Index where the answer's JSON object starts (after whitespace / a ```json fence)."""
    pos = 0
    m = _FENCE_RE.match(text)
    if m:
        pos = m.end()
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos < len(text) and text[pos] in "{[":
        return pos
    brace = text.find("{", pos)
    return brace if brace >= 0 else None


def strip_fences(text: str) -> str:
    """Remove a surrounding Markdown code fence (```json ... ```), if any."""
    s = text.strip()
    m = _FENCE_RE.match(s)
    if m and s.startswith("```"):
        s = s[m.end():]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


# --------------------------------------------------------------------------- #
# Position-tracking JSON parser
# --------------------------------------------------------------------------- #

_WS = " \t\n\r"
_NUMBER_RE = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")
_STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"', re.DOTALL)
_LITERALS = (("true", True), ("false", False), ("null", None))


class _JSONSpanError(ValueError):
    pass


def parse_json_with_spans(text: str, start: int = 0) -> tuple[Any, int, dict[tuple, tuple[int, int]]]:
    """Parse one JSON value from ``text[start:]``.

    Returns ``(value, end, spans)`` where ``spans`` maps each leaf path (tuple of
    keys / indices) to its ``[start, end)`` character span in ``text``. Leaves are
    scalars and empty containers. Raises ``ValueError`` on malformed JSON.
    """
    spans: dict[tuple, tuple[int, int]] = {}
    value, end = _parse_value(text, _skip_ws(text, start), (), spans)
    return value, end, spans


def _skip_ws(s: str, i: int) -> int:
    n = len(s)
    while i < n and s[i] in _WS:
        i += 1
    return i


def _parse_value(s: str, i: int, path: tuple, spans: dict) -> tuple[Any, int]:
    if i >= len(s):
        raise _JSONSpanError(f"unexpected end of JSON at {i}")
    ch = s[i]
    if ch == "{":
        return _parse_object(s, i, path, spans)
    if ch == "[":
        return _parse_array(s, i, path, spans)
    if ch == '"':
        end = _string_end(s, i)
        spans[path] = (i, end)
        return json.loads(s[i:end], strict=False), end
    m = _NUMBER_RE.match(s, i)
    if m and m.end() > i:
        spans[path] = (i, m.end())
        return json.loads(m.group(0)), m.end()
    for literal, value in _LITERALS:
        if s.startswith(literal, i):
            spans[path] = (i, i + len(literal))
            return value, i + len(literal)
    raise _JSONSpanError(f"unexpected character {ch!r} at {i}")


def _string_end(s: str, i: int) -> int:
    m = _STRING_RE.match(s, i)
    if not m:
        raise _JSONSpanError(f"unterminated string at {i}")
    return m.end()


def _parse_object(s: str, i: int, path: tuple, spans: dict) -> tuple[dict, int]:
    start = i
    obj: dict[str, Any] = {}
    i = _skip_ws(s, i + 1)
    if i < len(s) and s[i] == "}":
        spans[path] = (start, i + 1)
        return obj, i + 1
    while True:
        if i >= len(s) or s[i] != '"':
            raise _JSONSpanError(f"expected object key at {i}")
        key_end = _string_end(s, i)
        key = json.loads(s[i:key_end], strict=False)
        i = _skip_ws(s, key_end)
        if i >= len(s) or s[i] != ":":
            raise _JSONSpanError(f"expected ':' at {i}")
        obj[key], i = _parse_value(s, _skip_ws(s, i + 1), path + (key,), spans)
        i = _skip_ws(s, i)
        if i < len(s) and s[i] == ",":
            i = _skip_ws(s, i + 1)
            continue
        if i < len(s) and s[i] == "}":
            return obj, i + 1
        raise _JSONSpanError(f"expected ',' or '}}' at {i}")


def _parse_array(s: str, i: int, path: tuple, spans: dict) -> tuple[list, int]:
    start = i
    arr: list[Any] = []
    i = _skip_ws(s, i + 1)
    if i < len(s) and s[i] == "]":
        spans[path] = (start, i + 1)
        return arr, i + 1
    while True:
        value, i = _parse_value(s, i, path + (len(arr),), spans)
        arr.append(value)
        i = _skip_ws(s, i)
        if i < len(s) and s[i] == ",":
            i = _skip_ws(s, i + 1)
            continue
        if i < len(s) and s[i] == "]":
            return arr, i + 1
        raise _JSONSpanError(f"expected ',' or ']' at {i}")


# --------------------------------------------------------------------------- #
# Token logprobs -> character spans -> leaf confidences
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TokenSpan:
    start: int  # character offsets in the generated text
    end: int
    logprob: float


def token_spans_from_logprobs(logprobs: Any, generated_text: str | None = None
                              ) -> tuple[str, list[TokenSpan]] | None:
    """Turn a server ``logprobs`` payload into (reconstructed_text, token spans).

    Supports the OpenAI chat / llama.cpp shape ``{"content": [{"token", "logprob",
    "bytes"}]}`` and the legacy completions shape (vLLM) ``{"tokens",
    "token_logprobs", "text_offset"}``. Returns None when unusable.
    """
    if not isinstance(logprobs, dict):
        return None
    if isinstance(logprobs.get("content"), list):
        pieces: list[bytes] = []
        values: list[float | None] = []
        for item in logprobs["content"]:
            if not isinstance(item, dict):
                return None
            raw = item.get("bytes")
            if isinstance(raw, list):
                try:
                    piece = bytes(raw)
                except (TypeError, ValueError):
                    return None
            else:
                piece = str(item.get("token", "")).encode("utf-8")
            pieces.append(piece)
            values.append(_as_logprob(item.get("logprob")))
        return _spans_from_bytes(pieces, values)
    tokens = logprobs.get("tokens")
    lps = logprobs.get("token_logprobs")
    if isinstance(tokens, list) and isinstance(lps, list) and len(tokens) == len(lps):
        values = [_as_logprob(v) for v in lps]
        joined = "".join(str(t) for t in tokens)
        if generated_text is None or joined == generated_text:
            return _spans_from_bytes([str(t).encode("utf-8") for t in tokens], values)
        offsets = logprobs.get("text_offset")
        if generated_text is not None and isinstance(offsets, list) and len(offsets) == len(tokens):
            base = offsets[0] if offsets else 0
            starts = [int(o) - base for o in offsets]
            ends = starts[1:] + [len(generated_text)]
            spans = [TokenSpan(s, e, v) for s, e, v in zip(starts, ends, values) if v is not None]
            return generated_text, spans
    return None


def _as_logprob(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if math.isnan(value):
        return None
    return min(float(value), 0.0)


def _spans_from_bytes(pieces: list[bytes], values: list[float | None]) -> tuple[str, list[TokenSpan]] | None:
    buf = b"".join(pieces)
    try:
        text = buf.decode("utf-8")
    except UnicodeDecodeError:
        return None
    # byte offset -> index of the character containing that byte
    char_of_byte = []
    for ci, ch in enumerate(text):
        char_of_byte.extend([ci] * len(ch.encode("utf-8")))
    spans = []
    pos = 0
    for piece, value in zip(pieces, values):
        if piece and value is not None:
            spans.append(TokenSpan(char_of_byte[pos], char_of_byte[pos + len(piece) - 1] + 1, value))
        pos += len(piece)
    return text, spans


def leaf_confidences(spans: dict[tuple, tuple[int, int]], tokens: list[TokenSpan], *,
                     offset: int = 0, method: str = "geomean") -> dict[tuple, float]:
    """Confidence per leaf from the tokens overlapping its character span.

    ``spans`` are in answer coordinates; ``offset`` shifts them into the token
    coordinates. ``method``: "geomean" (default, exp(mean logprob)), "min" or "product".
    """
    if method not in ("geomean", "min", "product"):
        raise ValueError(f"unknown confidence method {method!r}")
    ordered = sorted(tokens, key=lambda t: t.start)
    ends = [t.end for t in ordered]
    out: dict[tuple, float] = {}
    for path, (start, end) in spans.items():
        s, e = start + offset, end + offset
        k = bisect.bisect_right(ends, s)  # first token ending after s
        selected = []
        while k < len(ordered) and ordered[k].start < e:
            if ordered[k].end > s:
                selected.append(ordered[k].logprob)
            k += 1
        if not selected:
            continue
        if method == "geomean":
            score = math.exp(sum(selected) / len(selected))
        elif method == "min":
            score = math.exp(min(selected))
        else:
            score = math.exp(sum(selected))
        out[path] = max(0.0, min(1.0, score))
    return out


def _confidences_for_answer(data: Any, generated: str, logprobs: Any) -> dict[tuple, float]:
    """Align the token stream with the answer JSON and score every leaf."""
    try:
        recon = token_spans_from_logprobs(logprobs, generated)
        if recon is None:
            return {}
        token_text, tokens = recon
        _, answer, answer_offset = split_thinking(token_text)
        start = locate_json(answer)
        if start is None:
            return {}
        value, _, spans = parse_json_with_spans(answer, start)
        if value != data:
            return {}
        return leaf_confidences(spans, tokens, offset=answer_offset)
    except (ValueError, IndexError, TypeError):
        return {}


def _parse_answer(answer: str) -> dict | None:
    start = locate_json(answer)
    if start is None:
        return None
    try:
        value, _, _ = parse_json_with_spans(answer, start)
    except ValueError:
        try:
            value = json.loads(strip_fences(answer), strict=False)
        except ValueError:
            return None
    return value if isinstance(value, dict) else None


# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #

def _make_client(settings: NuExtractSettings, timeout: float | None = None) -> httpx.Client:
    """httpx client for the server (tests monkeypatch this to inject a MockTransport)."""
    total = settings.timeout if timeout is None else timeout
    return httpx.Client(
        base_url=settings.base_url.rstrip("/") + "/",
        headers={"Authorization": f"Bearer {settings.api_key or 'EMPTY'}"},
        timeout=httpx.Timeout(total, connect=min(10.0, total)),
    )


def _server_root(settings: NuExtractSettings) -> str:
    base = settings.base_url.rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def _error_message(response: httpx.Response) -> tuple[str, dict]:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip()[:500], {}
    err = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(err, dict):
        return str(err.get("message") or "").strip(), err
    if isinstance(err, str):
        return err, payload
    if isinstance(payload, dict) and "detail" in payload:
        return str(payload["detail"]), payload
    if isinstance(payload, dict) and "message" in payload:
        return str(payload["message"]), payload
    return response.text.strip()[:500], payload if isinstance(payload, dict) else {}


def _raise_for_status(response: httpx.Response, settings: NuExtractSettings) -> None:
    if response.status_code < 400:
        return
    message, err = _error_message(response)
    status = response.status_code
    lowered = message.lower()
    if status == 503 or "loading model" in lowered:
        raise NuExtractError("The NuExtract server is still loading the model. Try again in a moment.",
                             kind="unavailable", status_code=status, detail=message)
    if status in (401, 403):
        raise NuExtractError("The NuExtract server rejected the API key.", kind="auth",
                             status_code=status, detail=message)
    if status == 404:
        raise NuExtractError(
            f"The NuExtract server at {settings.base_url} has no such endpoint "
            f"(the base URL should end in /v1).", kind="not_found", status_code=status, detail=message)
    if err.get("type") == "exceed_context_size_error" or "context length" in lowered \
            or "context size" in lowered or "maximum context" in lowered:
        n_prompt, n_ctx = err.get("n_prompt_tokens"), err.get("n_ctx")
        sizes = f" ({n_prompt} prompt tokens, context {n_ctx})" if n_prompt and n_ctx else ""
        raise NuExtractError(
            f"The document is too long for the NuExtract server's context window{sizes}. "
            "Start the server with a larger context (llama.cpp -c / vLLM --max-model-len), "
            "lower max_tokens, or shorten the document.",
            kind="context_length", status_code=status, detail=message)
    if 400 <= status < 500:
        raise NuExtractError(f"The NuExtract server rejected the request: {message or status}",
                             kind="bad_request", status_code=status, detail=message)
    raise NuExtractError(f"The NuExtract server failed (HTTP {status}): {message or 'no details'}",
                         kind="server", status_code=status, detail=message)


def _native_to_oai(payload: Any) -> Any:
    """llama.cpp native /completions response -> OpenAI completions shape."""
    if not isinstance(payload, dict) or "content" not in payload or "choices" in payload:
        return payload
    probs = payload.get("completion_probabilities")
    return {
        "choices": [{
            "text": payload.get("content") or "",
            "finish_reason": "length" if payload.get("stop_type") == "limit" else "stop",
            "logprobs": {"content": probs} if isinstance(probs, list) else None,
        }],
        "usage": {"prompt_tokens": payload.get("tokens_evaluated") or 0,
                  "completion_tokens": payload.get("tokens_predicted") or 0},
        "timings": payload.get("timings") or {},
    }


def _post_json(client: httpx.Client, url: str, body: dict, settings: NuExtractSettings) -> dict:
    try:
        response = client.post(url, json=body)
    except httpx.TimeoutException as exc:
        if isinstance(exc, httpx.ConnectTimeout):
            raise NuExtractError(f"Could not connect to the NuExtract server at {settings.base_url}.",
                                 kind="connection", detail=str(exc)) from exc
        raise NuExtractError(
            f"The NuExtract server did not finish within {settings.timeout:g} s. CPU inference on long "
            "documents is slow: raise the timeout, use a smaller document, or use a GPU server.",
            kind="timeout", detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise NuExtractError(
            f"Could not connect to the NuExtract server at {settings.base_url}. Is it running?",
            kind="connection", detail=str(exc)) from exc
    _raise_for_status(response, settings)
    try:
        payload = response.json()
    except ValueError as exc:
        raise NuExtractError("The NuExtract server returned a response that is not JSON.",
                             kind="protocol", detail=response.text[:500]) from exc
    payload = _native_to_oai(payload)
    if not isinstance(payload, dict) or not payload.get("choices"):
        raise NuExtractError("The NuExtract server returned no completion choices.", kind="protocol",
                             detail=json.dumps(payload)[:500] if payload is not None else None)
    return payload


def _get_json(client: httpx.Client, url: str) -> tuple[int, Any]:
    response = client.get(url)
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, None


def _probe_llama_cpp(client: httpx.Client, settings: NuExtractSettings) -> dict | None:
    """llama.cpp's GET /props (media marker, modalities); None if not llama.cpp."""
    try:
        status, props = _get_json(client, _server_root(settings) + "/props")
    except httpx.HTTPError:
        return None
    if status != 200 or not isinstance(props, dict) or "default_generation_settings" not in props:
        return None
    return props


def server_status(settings: NuExtractSettings) -> dict:
    """{"available": bool, "model": str | None, "detail": str, "backend": str | None}. Never raises."""
    result: dict[str, Any] = {"available": False, "model": None, "detail": "", "backend": None}
    try:
        with _make_client(settings, timeout=STATUS_TIMEOUT) as client:
            status, payload = _get_json(client, "models")
    except httpx.TimeoutException:
        result["detail"] = f"No answer from {settings.base_url} within {STATUS_TIMEOUT:g} s."
        return result
    except Exception as exc:  # noqa: BLE001 - status must never raise
        result["detail"] = f"Cannot reach the NuExtract server at {settings.base_url} ({type(exc).__name__})."
        return result
    if status == 503:
        result["detail"] = "The server is loading the model."
        return result
    if status in (401, 403):
        result["detail"] = "The server rejected the API key."
        return result
    if status != 200 or not isinstance(payload, dict):
        result["detail"] = f"Unexpected answer from {settings.base_url}/models (HTTP {status})."
        return result
    models = [m for m in payload.get("data") or [] if isinstance(m, dict)]
    if not models:
        result["detail"] = "The server is up but lists no models."
        return result
    chosen = next((m for m in models if m.get("id") == settings.model), models[0])
    owner = str(chosen.get("owned_by") or "")
    backend = {"llamacpp": "llama.cpp", "vllm": "vllm"}.get(owner, owner or None)
    model_id = str(chosen.get("id") or "") or None
    result.update(available=True, model=model_id, backend=backend)
    detail = f"{backend or 'OpenAI-compatible'} server ready"
    ctx = (chosen.get("meta") or {}).get("n_ctx") or chosen.get("max_model_len")
    if ctx:
        detail += f", context {ctx} tokens"
    if model_id and "nuextract" not in model_id.lower():
        detail += f" (served model '{model_id}' does not look like NuExtract3)"
    result["detail"] = detail + "."
    return result


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

def extract(settings: NuExtractSettings, *, text: str | None = None, images: list[bytes] | None = None,
            template: dict, instructions: str = "", examples: Iterable[dict] = (),
            enable_thinking: bool = False, temperature: float = 0.2, max_tokens: int = 4096,
            strict: bool = True, transport: str = "auto") -> NuExtractOutput:
    """Run one NuExtract3 structured extraction.

    ``examples`` are in-context examples ``[{"text": str, "output": dict}]``.
    ``transport``: "auto" (completions; chat only for images on non-llama.cpp
    servers), "completions" or "chat".
    With ``strict`` (default) a truncated or non-JSON answer raises
    ``NuExtractError``; otherwise the output is returned with ``data=None``.
    A ``finish_reason`` of "length" is only an error when the answer does not parse.
    """
    images = list(images or [])
    examples = [dict(ex) for ex in examples]
    if text is None and not images:
        raise NuExtractError("Nothing to extract from: provide text and/or page images.", kind="invalid_input")
    if not isinstance(template, (dict, str)) or not template:
        raise NuExtractError("A non-empty NuExtract template is required.", kind="invalid_input")
    if transport not in ("auto", "completions", "chat"):
        raise NuExtractError(f"Unknown transport {transport!r}.", kind="invalid_input")
    if max_tokens < 1:
        raise NuExtractError("max_tokens must be positive.", kind="invalid_input")

    started = time.monotonic()
    with _make_client(settings) as client:
        marker = None
        if images and transport != "chat":
            props = _probe_llama_cpp(client, settings)
            if props is not None:
                if not (props.get("modalities") or {}).get("vision", False):
                    raise NuExtractError(
                        "The llama.cpp server was started without a vision projector "
                        "(add --mmproj, or use -hf which downloads it), so page images cannot be read.",
                        kind="invalid_input")
                marker = props.get("media_marker") or DEFAULT_LLAMA_MEDIA_MARKER
            elif transport == "completions":
                raise NuExtractError("Image input through /v1/completions needs a llama.cpp server; "
                                     "use transport='chat' for vLLM.", kind="invalid_input")
        use_chat = transport == "chat" or (bool(images) and marker is None)
        if use_chat:
            body = {
                "model": settings.model,
                "messages": _chat_messages(text, images, examples),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "logprobs": True,
                "top_logprobs": 1,
                "chat_template_kwargs": {"template": _template_text(template), "instructions": instructions,
                                         "enable_thinking": enable_thinking},
            }
            payload = _post_json(client, "chat/completions", body, settings)
        else:
            prompt_text = render_prompt(template, text=text, n_images=len(images), instructions=instructions,
                                        examples=examples, enable_thinking=enable_thinking,
                                        image_token=marker or VISION_PLACEHOLDER)
            if images:
                # llama.cpp native endpoint: the only one taking a prompt string plus images.
                body = {"prompt": {"prompt_string": prompt_text,
                                   "multimodal_data": [base64.b64encode(img).decode("ascii") for img in images]},
                        "n_predict": max_tokens, "temperature": temperature, "n_probs": 1, "stream": False}
                payload = _post_json(client, _server_root(settings) + "/completions", body, settings)
            else:
                body = {"model": settings.model, "prompt": prompt_text, "max_tokens": max_tokens,
                        "temperature": temperature, "logprobs": 1, "stream": False}
                payload = _post_json(client, "completions", body, settings)
    elapsed = time.monotonic() - started
    return _build_output(payload, use_chat=use_chat, enable_thinking=enable_thinking, max_tokens=max_tokens,
                         strict=strict, elapsed=elapsed)


def _build_output(payload: dict, *, use_chat: bool, enable_thinking: bool, max_tokens: int, strict: bool,
                  elapsed: float) -> NuExtractOutput:
    choice = payload["choices"][0] if isinstance(payload["choices"][0], dict) else {}
    finish = choice.get("finish_reason")
    reasoning: str | None = None
    if use_chat:
        message = choice.get("message") or {}
        content = message.get("content") or ""
        server_reasoning = message.get("reasoning_content") or message.get("reasoning")
        inline_reasoning, answer, _ = split_thinking(content)
        reasoning = (server_reasoning or "").strip() or inline_reasoning
        generated = ((server_reasoning + "\n" + THINK_CLOSE) if server_reasoning else "") + content
    else:
        generated = choice.get("text") or ""
        reasoning, answer, _ = split_thinking(generated)
        if enable_thinking and reasoning is None and THINK_CLOSE not in generated:
            # Still inside the reasoning block: there is no answer yet.
            reasoning, answer = generated.strip() or None, ""
    raw = answer.strip()
    data = _parse_answer(raw)
    usage, timings = _usage_and_timings(payload)
    if data is None and strict:
        _raise_unparsed(finish, raw, generated, max_tokens=max_tokens, thinking=enable_thinking,
                        hint="Raise max_tokens or reduce the schema/document size.")
    confidences = _confidences_for_answer(data, generated, choice.get("logprobs")) if data is not None else {}
    return NuExtractOutput(data=data, raw=raw, reasoning=reasoning, confidences=confidences,
                           finish_reason=finish, usage=usage, elapsed_seconds=round(elapsed, 3),
                           timings=timings, transport="chat" if use_chat else "completions")


def _usage_and_timings(payload: dict) -> tuple[dict, dict]:
    usage_in = payload.get("usage") or {}
    timings = payload.get("timings") if isinstance(payload.get("timings"), dict) else {}
    usage = {
        "prompt_tokens": int(usage_in.get("prompt_tokens") or timings.get("prompt_n") or 0),
        "completion_tokens": int(usage_in.get("completion_tokens") or timings.get("predicted_n") or 0),
    }
    return usage, dict(timings)


def _raise_unparsed(finish: str | None, raw: str, generated: str, *, max_tokens: int, thinking: bool,
                    hint: str) -> None:
    if finish == "length":
        where = " while still reasoning" if thinking and not raw else ""
        raise NuExtractError(
            f"NuExtract stopped at the max_tokens limit ({max_tokens}){where} before finishing its "
            f"JSON answer. {hint}", kind="truncated", raw=raw or generated)
    raise NuExtractError("NuExtract returned an answer that is not a valid JSON object.",
                         kind="invalid_json", raw=raw or generated)


# --------------------------------------------------------------------------- #
# Template generation (chat template mode "template-generation")
# --------------------------------------------------------------------------- #

@dataclass
class TemplateOutput:
    template: dict | None  # generated NuExtract template (None if unparseable and strict=False)
    raw: str  # answer text, thinking removed
    reasoning: str | None
    usage: dict  # {"prompt_tokens": int, "completion_tokens": int}
    elapsed_seconds: float
    finish_reason: str | None = None
    timings: dict = field(default_factory=dict)


def render_template_generation_prompt(*, text: str | None = None, description: str = "",
                                      enable_thinking: bool = False) -> str:
    """Prompt for ``mode="template-generation"``.

    The chat template has a single input slot for this mode: every user content
    part goes between 【document_start】 and 【document_end】, joined with "\\n"
    (``instructions`` and examples are ignored in this mode). A description and a
    document are therefore concatenated, description first. The official template
    rejects ``enable_thinking=True`` for this mode; here it is allowed as an
    explicit opt-in (the prompt then ends with an open ``<think>``).
    """
    parts = [p for p in (description.strip() if description else "", text) if p]
    content = "".join(p + "\n" for p in parts).strip()
    return ("<|im_start|>user\n【task】template generation\n【document_start】\n" + content
            + "\n【document_end】<|im_end|>\n<|im_start|>assistant\n"
            + ("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"))


def generate_template(settings: NuExtractSettings, *, text: str | None = None, description: str = "",
                      enable_thinking: bool = False, temperature: float = 0.2, max_tokens: int = 4096,
                      strict: bool = True) -> TemplateOutput:
    """Ask NuExtract3 to propose an extraction template.

    ``description`` is natural-language guidance ("meeting minutes: attendees with
    roles..."), ``text`` an exemplar document (several exemplars can be joined with
    blank lines). Either or both may be given. Convert the result with
    :func:`template_to_fields`. Raises ``NuExtractError`` like :func:`extract`.
    """
    if not (text and text.strip()) and not (description and description.strip()):
        raise NuExtractError("Provide a description and/or an exemplar document.", kind="invalid_input")
    if max_tokens < 1:
        raise NuExtractError("max_tokens must be positive.", kind="invalid_input")
    prompt = render_template_generation_prompt(text=text, description=description, enable_thinking=enable_thinking)
    started = time.monotonic()
    with _make_client(settings) as client:
        payload = _post_json(client, "completions", {
            "model": settings.model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": temperature, "stream": False}, settings)
    elapsed = time.monotonic() - started
    choice = payload["choices"][0] if isinstance(payload["choices"][0], dict) else {}
    finish = choice.get("finish_reason")
    generated = choice.get("text") or ""
    reasoning, answer, _ = split_thinking(generated)
    if enable_thinking and reasoning is None and THINK_CLOSE not in generated:
        reasoning, answer = generated.strip() or None, ""
    raw = answer.strip()
    template = _parse_answer(raw)
    if template is not None and not template:
        template = None
    usage, timings = _usage_and_timings(payload)
    if template is None and strict:
        _raise_unparsed(finish, raw, generated, max_tokens=max_tokens, thinking=enable_thinking,
                        hint="Raise max_tokens or shorten the exemplar.")
    return TemplateOutput(template=template, raw=raw, reasoning=reasoning, usage=usage,
                          elapsed_seconds=round(elapsed, 3), finish_reason=finish, timings=timings)


# NuExtract leaf type -> (app field type, description hint)
_TEMPLATE_TYPES: dict[str, tuple[str, str]] = {
    "verbatim-string": ("verbatim", ""),
    "string": ("text", ""),
    "": ("text", ""),
    "date": ("date", ""),
    "date-time": ("date", ""),  # app dates are ISO dates; times are dropped
    "time": ("text", "time (ISO 8601, e.g. 14:30)"),
    "duration": ("text", "duration (ISO 8601, e.g. P3D)"),
    "number": ("number", ""),
    "integer": ("integer", ""),
    "boolean": ("boolean", ""),
    "email-address": ("verbatim", "email address"),
    "url": ("verbatim", "URL"),
    "phone-number": ("text", "phone number (E.164 when the country code is known)"),
    "iban": ("text", "IBAN"),
    "bic": ("text", "BIC/SWIFT code"),
    "country": ("text", "country (ISO 3166-1 alpha-2 code)"),
    "currency": ("text", "currency (ISO 4217 code)"),
    "language": ("text", "language (ISO 639-3 code)"),
    "language-tag": ("text", "language tag (BCP 47)"),
    "script": ("text", "script (ISO 15924 code)"),
    "unit-code": ("text", "unit of measure (UCUM code)"),
}
MAX_FIELD_DEPTH = 3
_MAX_TOP_FIELDS, _MAX_CHILD_FIELDS, _MAX_OPTIONS, _MAX_OPTION_LEN = 60, 40, 50, 200
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def template_to_fields(template: dict, *, unwrap_root: bool = True) -> list[dict]:
    """Convert a NuExtract template into the app's schema fields (inverse of schema_to_template).

    * keys become lower_snake_case names matching ``^[a-z][a-z0-9_]{0,63}$``, deduplicated;
    * verbatim-string -> verbatim, string -> text, date/date-time -> date, number,
      integer, boolean as is; other NuExtract types (time, currency, email-address,
      country, ...) -> text or verbatim with the type described in ``description``;
    * ``["a", "b"]`` -> enum, ``[["a", "b"]]`` -> enum array, ``[x]`` -> array of x,
      ``{...}`` -> object, ``[{...}]`` -> object array; empty objects are dropped;
    * a template that is a single object under one key (``{"Meeting": {...}}``, as
      the model often writes) is unwrapped when ``unwrap_root``;
    * nesting is capped at three levels: a single object that would go deeper is
      flattened into prefixed sibling fields, an array of objects becomes a text array.
    """
    if not isinstance(template, dict):
        raise ValueError("A NuExtract template must be a JSON object.")
    if unwrap_root and len(template) == 1:
        (only,) = template.values()
        if isinstance(only, dict) and only:
            template = only
    return _template_fields(template, level=1, limit=_MAX_TOP_FIELDS)


def _template_fields(node: dict, *, level: int, limit: int) -> list[dict]:
    fields: list[dict] = []
    used: set[str] = set()
    for key, value in node.items():
        for field_ in _expand(_snake_name(key), value, level):
            if len(fields) >= limit:
                return fields
            field_["name"] = _unique_name(field_["name"], used)
            used.add(field_["name"])
            fields.append(field_)
    return fields


def _new_field(name: str, ftype: str, *, array: bool = False, description: str = "",
               options: list[str] | None = None, fields: list[dict] | None = None) -> dict:
    return {"name": name, "type": ftype, "description": description, "array": array,
            "options": options or [], "fields": fields or []}


def _expand(name: str, value: Any, level: int) -> list[dict]:
    """Fields for one template entry (several when an over-deep object is flattened)."""
    if isinstance(value, dict):
        if not value:
            return []
        if level >= MAX_FIELD_DEPTH:  # children would exceed the nesting limit: flatten
            out: list[dict] = []
            for key, child in value.items():
                out.extend(_expand(_snake_name(f"{name}_{_snake_name(key)}"), child, level))
            return out
        children = _template_fields(value, level=level + 1, limit=_MAX_CHILD_FIELDS)
        return [_new_field(name, "object", fields=children)] if children else []
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], list):  # multi-enum [["a", "b"]]
            options = _clean_options(value[0])
            return [_new_field(name, "enum", array=True, options=options) if options
                    else _new_field(name, "text", array=True)]
        if len(value) >= 2 and all(isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in value):
            options = _clean_options(value)
            return [_new_field(name, "enum", options=options) if options else _new_field(name, "text")]
        if value and isinstance(value[0], dict):
            item = value[0]
            if not item:
                return []
            if level >= MAX_FIELD_DEPTH:
                keys = ", ".join(str(k) for k in list(item)[:8])
                return [_new_field(name, "text", array=True, description=f"summary of: {keys}"[:1000])]
            children = _template_fields(item, level=level + 1, limit=_MAX_CHILD_FIELDS)
            return [_new_field(name, "object", array=True, fields=children)] if children else []
        if len(value) == 1:
            ftype, hint = _leaf_type(value[0])
            return [_new_field(name, ftype, array=True, description=hint)]
        return [_new_field(name, "text", array=True)]
    ftype, hint = _leaf_type(value)
    return [_new_field(name, ftype, description=hint)]


def _leaf_type(value: Any) -> tuple[str, str]:
    if not isinstance(value, str):
        return "text", ""
    key = value.strip().lower()
    if key in _TEMPLATE_TYPES:
        return _TEMPLATE_TYPES[key]
    if key.startswith("region:"):
        return "text", f"region (ISO 3166-2:{value.strip()[7:].upper()} subdivision code)"
    return "text", f"type: {value.strip()}"[:1000]


def _clean_options(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if isinstance(v, (dict, list)) or v is None:
            continue
        option = " ".join(str(v).split())[:_MAX_OPTION_LEN].strip()
        if option and option.casefold() not in seen:
            seen.add(option.casefold())
            out.append(option)
        if len(out) >= _MAX_OPTIONS:
            break
    return out


def _snake_name(key: Any) -> str:
    s = unicodedata.normalize("NFKD", str(key)).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    if not s:
        s = "field"
    if not s[0].isalpha():
        s = "f_" + s
    return s[:64].rstrip("_")


def _unique_name(name: str, used: set[str]) -> str:
    if name not in used:
        return name
    n = 2
    while True:
        suffix = f"_{n}"
        candidate = name[:64 - len(suffix)].rstrip("_") + suffix
        if candidate not in used:
            return candidate
        n += 1
