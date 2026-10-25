"""Bridge between the agreed schema and Unstract (Zipstack, AGPL-3.0): Prompt Studio prompts and project
files generated from the schema, and Unstract results mapped back onto schema-shaped records.

Formats follow the Unstract source (v0.197.0):
- Project file: what Prompt Studio's "Export project" writes and "Import project" accepts
  (backend/prompt_studio/prompt_studio_core_v2/prompt_studio_helper.py: export_project_settings,
  validate_import_file, create_tool_from_import_data, import_prompts). Required top-level keys are
  tool_metadata (tool_name, description, author), tool_settings and prompts (prompt_key, prompt).
- API deployment results: a list of per-file items {"file", "file_execution_id", "status" ("Success" /
  "Failed"), "result": {"output": {<prompt_key>: value}, "metadata": {...}, "metrics": {...}},
  "error", "metadata": {"source_name", ..., "usage"}} (unstract/core worker_models.FileExecutionResult,
  backend/workflow_manager/workflow_v2/dto.py ExecutionResponse). POST responses wrap it as
  {"message": {"execution_status", "status_api", "error", "result": [...]}}; status GETs as
  {"status": ..., "message": [...]}.

Everything here is pure (no I/O), so it is shared by the API-deployment solution and results import.
"""
from collections.abc import Iterable, Mapping
import json
import re
from typing import Any

from app.schema import ExtractionSchema, SchemaField, TYPE_LABELS


# --- Schema -> Prompt Studio ------------------------------------------------------------------

# Unstract's own defaults (prompt_studio_core_v2/constants.py DefaultPrompts), with the schema description added.
DEFAULT_PREAMBLE = (
    "Your ability to extract and summarize this context accurately is essential for effective analysis. "
    "Pay close attention to the context's language, structure, and any cross-references to ensure a "
    "comprehensive and precise extraction of information. Do not use prior knowledge or information from "
    "outside the context to answer the questions. Only use the information provided in the context to "
    "answer the questions."
)
DEFAULT_POSTAMBLE = ("Do not include any explanation in the reply. "
                     "Only include the extracted information in the reply.")

SCALAR_ENFORCE_TYPES = {"text": "text", "verbatim": "text", "enum": "text", "date": "date",
                        "number": "number", "integer": "number", "boolean": "boolean"}
NOT_FOUND = "NA"  # Unstract maps an "NA" answer to null for typed prompts.

SCALAR_INSTRUCTIONS = {
    "text": "Answer concisely in plain text.",
    "verbatim": "Copy the text exactly as it appears in the document, without rewording.",
    "date": "Answer with the date only, in YYYY-MM-DD format.",
    "number": "Answer with the number only.",
    "integer": "Answer with the whole number only.",
    "boolean": "Answer yes or no.",
}
PLACEHOLDERS = {"text": "<text>", "verbatim": "<exact text>", "date": "<YYYY-MM-DD>", "number": "<number>",
                "integer": "<whole number>", "boolean": "<true or false>"}


def enforce_type(field: SchemaField) -> str:
    """Prompt Studio enforce type for a top-level field: objects and lists are JSON prompts."""
    if field.array or field.type == "object":
        return "json"
    return SCALAR_ENFORCE_TYPES[field.type]


def _label(field: SchemaField) -> str:
    return field.name.replace("_", " ")


def _sentence(text: str) -> str:
    text = text.strip()
    return text if not text or text[-1] in ".!?" else text + "."


def _placeholder(field: SchemaField) -> Any:
    if field.type == "object":
        value: Any = {child.name: _placeholder(child) for child in field.fields}
    elif field.type == "enum":
        value = " | ".join(field.options)
    else:
        value = PLACEHOLDERS[field.type]
    return [value] if field.array else value


def _key_lines(fields: list[SchemaField], indent: str = "") -> list[str]:
    lines = []
    for field in fields:
        kind = TYPE_LABELS[field.type]
        if field.type == "enum":
            kind += ": " + ", ".join(f'"{option}"' for option in field.options)
        if field.array:
            kind = f"list of {'objects' if field.type == 'object' else kind}"
        line = f'{indent}- "{field.name}" ({kind})'
        if field.description:
            line += f": {_sentence(field.description)}"
        lines.append(line)
        if field.type == "object":
            lines.extend(_key_lines(field.fields, indent + "  "))
    return lines


def prompt_text(field: SchemaField, notes: str = "") -> str:
    """Prompt Studio prompt for one top-level field, describing the exact answer shape."""
    description = f" {_sentence(field.description)}" if field.description else ""
    if field.type == "object":
        shape = json.dumps(_placeholder(field), ensure_ascii=False)
        if field.array:
            parts = [f'List every item for "{_label(field)}".{description}',
                     "Return a JSON array with one object per item, in document order. "
                     "Each object must have exactly these keys:"]
            empty = "Use null for a value the document does not state and [] for an empty list. " \
                    "If there are no items, return []."
        else:
            parts = [f'Extract "{_label(field)}".{description}', "Return one JSON object with exactly these keys:"]
            empty = "Use null for a value the document does not state and [] for an empty list."
        parts += _key_lines(field.fields) + [empty, f"Shape: {shape}"]
    elif field.array:
        kind = TYPE_LABELS[field.type]
        if field.type == "enum":
            kind += " (" + ", ".join(f'"{option}"' for option in field.options) + ")"
        parts = [f'List every "{_label(field)}".{description}',
                 f"Return a JSON array of {kind} values, in document order. If there are none, return [].",
                 f"Shape: {json.dumps(_placeholder(field), ensure_ascii=False)}"]
    else:
        instruction = (f"Answer with exactly one of: {', '.join(json.dumps(option) for option in field.options)}."
                       if field.type == "enum" else SCALAR_INSTRUCTIONS[field.type])
        parts = [f'Extract "{_label(field)}".{description}', instruction,
                 f"If the document does not state it, answer {NOT_FOUND}."]
    if notes.strip():
        parts.append(f"Guidance: {notes.strip()}")
    return "\n".join(parts)


def _notes_for(field: SchemaField, exemplar_notes: Mapping[str, str] | str | None) -> str:
    if isinstance(exemplar_notes, str):
        return exemplar_notes
    return (exemplar_notes or {}).get(field.name, "")


def schema_to_prompts(schema: ExtractionSchema, exemplar_notes: Mapping[str, str] | str | None = None) -> list[dict]:
    """One Prompt Studio prompt per top-level field, in Unstract's export format (prompt_key = field name).

    exemplar_notes: optional guidance per field name (e.g. from exemplar_notes_from_records), or one string
    appended to every prompt.
    """
    return [{
        "prompt_key": field.name,
        "prompt": prompt_text(field, _notes_for(field, exemplar_notes)),
        "active": True,
        "required": None,
        "enforce_type": enforce_type(field),
        "sequence_number": index,
        "prompt_type": "PROMPT",
        "assert_prompt": None,
        "assertion_failure_prompt": None,
        "is_assert": False,
        "evaluate": False,
        "eval_quality_faithfulness": False,
        "eval_quality_correctness": False,
        "eval_quality_relevance": False,
        "eval_security_pii": False,
        "eval_guidance_toxicity": False,
        "eval_guidance_completeness": False,
        "enable_postprocessing_webhook": False,
        "postprocessing_webhook_url": None,
    } for index, field in enumerate(schema.fields, 1)]


def schema_to_project_export(schema: ExtractionSchema, exemplar_notes: Mapping[str, str] | str | None = None,
                             guidance: str = "", chunk_size: int = 0) -> dict:
    """An importable Prompt Studio project (Prompt Studio > Import Project, .json file).

    chunk_size 0 sends the whole document as context for every prompt (no vector retrieval), which suits
    meeting minutes; Unstract still requires an embedding and vector DB adapter on the profile.
    """
    preamble = DEFAULT_PREAMBLE
    if schema.description.strip():
        preamble += f"\n\nThe documents are: {_sentence(schema.description)}"
    if guidance.strip():
        preamble += f"\n\n{guidance.strip()}"
    return {
        "tool_metadata": {
            "tool_name": schema.name[:200],
            "description": (schema.description.strip() or f"Extraction against the agreed schema '{schema.name}'.")[:1000],
            "author": "doccog",
            "icon": "",
        },
        "tool_settings": {
            "preamble": preamble,
            "postamble": DEFAULT_POSTAMBLE,
            "summarize_prompt": "",
            "summarize_context": False,
            "summarize_as_source": False,
            "enable_challenge": False,
            "enable_highlight": False,
            "exclude_failed": True,
            "single_pass_extraction_mode": False,
            "prompt_grammer": None,
        },
        "default_profile_settings": {
            "chunk_size": chunk_size,
            "chunk_overlap": 0 if chunk_size == 0 else min(128, chunk_size // 8),
            "retrieval_strategy": "simple",
            "similarity_top_k": 3,
            "section": "Default",
            "profile_name": "Default Profile",
        },
        "prompts": schema_to_prompts(schema, exemplar_notes),
        "export_metadata": {
            "exported_at": schema.created_at or None,
            "tool_id": None,
            "generated_by": "doccog",
            "schema_id": schema.id,
            "schema_version": schema.version,
        },
    }


def prompt_list_text(prompts: list[dict]) -> str:
    """Copy-paste friendly rendering of the prompts (for adding them by hand in Prompt Studio)."""
    blocks = []
    for prompt in prompts:
        blocks.append(f"### {prompt['prompt_key']}  (enforce type: {prompt['enforce_type']})\n{prompt['prompt']}")
    return "\n\n".join(blocks)


def exemplar_notes_from_records(schema: ExtractionSchema, records: Iterable[Mapping], max_examples: int = 2,
                                max_chars: int = 600) -> dict[str, str]:
    """Per-field 'example answer' guidance from schema-shaped exemplar records (e.g. reviewed results)."""
    notes: dict[str, str] = {}
    records = list(records)
    for field in schema.fields:
        examples = []
        for record in records:
            value = record.get(field.name) if isinstance(record, Mapping) else None
            if value in (None, "", [], {}):
                continue
            if field.array and isinstance(value, list):
                value = value[:3]
            rendered = json.dumps(value, ensure_ascii=False)
            examples.append(rendered if len(rendered) <= max_chars else rendered[:max_chars - 1] + "…")
            if len(examples) >= max_examples:
                break
        if examples:
            notes[field.name] = "Example answers from similar documents: " + " ; ".join(examples)
    return notes


# --- Unstract output -> schema record ---------------------------------------------------------

MISSING_TEXT = {"na", "n/a", "null", "none", "not found", "not stated", "not available", "not mentioned"}


def _norm(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")


def decode_json(value: Any) -> Any:
    """Decode string-encoded JSON (also inside ``` fences or Unstract's §§§ markers); else return as-is."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if "§§§" in text:
        parts = [part.strip() for part in text.split("§§§") if part.strip()]
        text = parts[0] if parts else ""
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fence:
        text = fence.group(1).strip()
    if text[:1] in ("[", "{"):
        try:
            return json.loads(text)
        except ValueError:
            return value
    return value


def _clean(value: Any) -> Any:
    """Unstract-style 'NA' strings become null, recursively."""
    if isinstance(value, str):
        return None if value.strip().casefold() in MISSING_TEXT else value
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    return value


def _lookup(source: Mapping, name: str) -> tuple[bool, Any, str | None]:
    if name in source:
        return True, source[name], name
    wanted = _norm(name)
    for key, value in source.items():
        if _norm(key) == wanted:
            return True, value, key
    return False, None, None


def _shape(value: Any, field: SchemaField, path: str, notes: list[str]) -> Any:
    """Fit decoded Unstract JSON to one schema field (key spelling, wrappers, lists vs single items)."""
    value = _clean(decode_json(value))
    if value is None:
        return None
    if field.array:
        if isinstance(value, dict):
            lists = [item for item in value.values() if isinstance(item, list)]
            found, inner, _ = _lookup(value, field.name)
            if found and isinstance(inner, list):
                value = inner
            elif len(value) == 1 and lists:
                value = lists[0]
            elif field.type == "object":
                value = [value]
            else:
                value = list(value.values())
        if not isinstance(value, list):
            value = [value]
        single = field.model_copy(update={"array": False})
        return [_shape(item, single, path, notes) for item in value]
    if field.type == "object":
        if isinstance(value, list):
            dicts = [item for item in value if isinstance(item, dict)]
            if not dicts:
                return None
            if len(dicts) > 1:
                notes.append(f"Unstract returned {len(dicts)} objects for '{path}'; kept the first.")
            value = dicts[0]
        if not isinstance(value, dict):
            return None
        shaped = {}
        for child in field.fields:
            found, inner, _ = _lookup(value, child.name)
            shaped[child.name] = _shape(inner, child, f"{path}.{child.name}", notes) if found else None
        return shaped
    if isinstance(value, dict):
        found, inner, _ = _lookup(value, field.name)
        if found:
            return _shape(inner, field, path, notes)
        scalars = [item for item in value.values() if not isinstance(item, (dict, list))]
        return scalars[0] if len(scalars) == 1 else json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        scalars = [item for item in value if item is not None and not isinstance(item, (dict, list))]
        if len(scalars) > 1:
            notes.append(f"Unstract returned {len(scalars)} values for '{path}'; kept the first.")
        return scalars[0] if scalars else None
    return value


def output_to_record(output: Any, schema: ExtractionSchema | list[SchemaField]) -> tuple[dict, list[str]]:
    """Map Unstract's structured output ({prompt_key: value}) onto a plain schema-shaped record.

    Handles prompt keys that differ in case/spacing, string-encoded JSON values, "NA" answers, a single
    JSON prompt holding the whole record, and list/object wrapper mismatches. Unmapped keys are noted.
    """
    fields = schema.fields if isinstance(schema, ExtractionSchema) else schema
    notes: list[str] = []
    output = decode_json(output)
    if not isinstance(output, dict):
        notes.append("Unstract output was not a JSON object; nothing could be mapped.")
        return {field.name: None for field in fields}, notes
    source = dict(output)
    used: set[str] = set()
    record: dict = {}
    for field in fields:
        found, value, key = _lookup(source, field.name)
        if found:
            used.add(key)
            record[field.name] = _shape(value, field, field.name, notes)
    # A single JSON prompt may carry the whole record (or several fields) as one object.
    names = {_norm(field.name) for field in fields}
    for key, value in source.items():
        if key in used:
            continue
        decoded = decode_json(value)
        if isinstance(decoded, dict) and names & {_norm(inner) for inner in decoded}:
            filled = []
            for field in fields:
                if record.get(field.name) not in (None, [], {}):
                    continue
                found, inner, _ = _lookup(decoded, field.name)
                if found:
                    record[field.name] = _shape(inner, field, field.name, notes)
                    filled.append(field.name)
            used.add(key)
            if filled:
                notes.append(f"Read {', '.join(filled)} from the JSON object in Unstract prompt '{key}'.")
    unmapped = [key for key in source if key not in used]
    if unmapped:
        notes.append(f"Ignored Unstract output key(s) not in the schema: {', '.join(map(str, unmapped))}.")
    missing = [field.name for field in fields if field.name not in record]
    if missing and len(missing) < len(fields):
        notes.append(f"Unstract returned no prompt for: {', '.join(missing)}.")
    elif missing:
        notes.append("None of the Unstract prompt keys match the schema field names.")
    for field in fields:
        record.setdefault(field.name, None)
    return record, list(dict.fromkeys(notes))


# --- Confidence --------------------------------------------------------------------------------

def _numbers(value: Any, highlight: bool = False) -> list[float]:
    found: list[float] = []

    def visit(item):
        if isinstance(item, bool):
            return
        if isinstance(item, (int, float)) and not highlight:
            if 0 <= item <= 1:
                found.append(float(item))
        elif isinstance(item, list):
            # Highlight rows are [page, x, y, width/height..., confidence]: confidence is index 4.
            if highlight and len(item) >= 5 and all(isinstance(part, (int, float)) for part in item[:5]):
                if 0 <= item[4] <= 1 and not isinstance(item[4], bool):
                    found.append(float(item[4]))
                return
            for part in item:
                visit(part)
        elif isinstance(item, dict):
            for part in item.values():
                visit(part)
    visit(value)
    return found


def prompt_confidences(metadata: Any, keys: Iterable[str]) -> dict[str, Any]:
    """Confidence structures per prompt key from Unstract result metadata, if the edition provides them.

    Precedence matches Unstract's own rule engine: word_confidence_data, then the 5th element of
    highlight_data rows, then confidence_data. Community edition returns none of these.
    """
    if not isinstance(metadata, dict):
        return {}
    result: dict[str, Any] = {}
    for key in keys:
        for source in ("word_confidence_data", "highlight_data", "confidence_data"):
            block = metadata.get(source)
            if not isinstance(block, dict) or key not in block:
                continue
            value = block[key]
            if source == "highlight_data":
                numbers = _numbers(value, highlight=True)
                if numbers:
                    result[key] = sum(numbers) / len(numbers)
                    break
            elif _numbers(value):
                result[key] = value
                break
    return result


# --- Results files -----------------------------------------------------------------------------

def _stem(name: str) -> str:
    base = re.split(r"[\\/]", str(name))[-1]
    base = re.sub(r"\.(pdf|txt|docx?|json|md|html?)$", "", base, flags=re.I)
    return re.sub(r"[^a-z0-9]+", "", base.casefold())


def file_items(payload: Any) -> list[dict]:
    """Per-file result items from any Unstract results JSON shape (see module docstring)."""
    payload = decode_json(payload)
    if isinstance(payload, dict) and "message" in payload and isinstance(payload["message"], (dict, list)):
        payload = payload["message"]
    if isinstance(payload, dict) and isinstance(payload.get("result"), list):
        payload = payload["result"]
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        payload = payload["results"]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if "output" in payload or "file" in payload:
            return [payload]
        # {"<file name>": <per-file output or item>}
        if payload and all(isinstance(value, dict) for value in payload.values()) and any(
                re.search(r"\.\w{2,5}$", str(key)) for key in payload):
            return [{"file": key, **(value if "output" in value or "result" in value else {"result": {"output": value}})}
                    for key, value in payload.items()]
        return [{"result": {"output": payload}}]
    return []


def item_file_name(item: dict) -> str | None:
    result = decode_json(item.get("result"))
    candidates = [item.get("file"), item.get("file_name"), (item.get("metadata") or {}).get("source_name")
                  if isinstance(item.get("metadata"), dict) else None]
    if isinstance(result, dict) and isinstance(result.get("metadata"), dict):
        candidates += [result["metadata"].get("file_name"), result["metadata"].get("source_name")]
    if isinstance(item.get("metadata"), dict):
        candidates.append(item["metadata"].get("file_name"))
    return next((str(name) for name in candidates if name), None)


def item_output(item: dict) -> tuple[Any, dict, str | None]:
    """(structured output, result metadata, error) of one per-file item."""
    error = item.get("error") or None
    if str(item.get("status", "")).casefold() in ("failed", "error") and not error:
        error = "Unstract reported this file as failed."
    result = decode_json(item.get("result")) if "result" in item else {k: v for k, v in item.items() if k not in (
        "file", "file_name", "file_execution_id", "status", "error", "metadata")}
    if isinstance(result, dict) and "output" in result:
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        return decode_json(result.get("output")), metadata, error
    return result, {}, error


def _names(documents: Iterable[Any]) -> list[str]:
    return [document if isinstance(document, str) else getattr(document, "name", None) or document["name"]
            for document in documents]


def match_document(file_name: str | None, documents: Iterable[Any]) -> str | None:
    names = _names(documents)
    if not file_name:
        return names[0] if len(names) == 1 else None
    base = re.split(r"[\\/]", file_name)[-1]
    for name in names:
        if name == base:
            return name
    for name in names:
        if name.casefold() == base.casefold():
            return name
    stems = [name for name in names if _stem(name) == _stem(base)]
    return stems[0] if len(stems) == 1 else None


def parse_results_file(payload: Any, documents: Iterable[Any], file_name: str | None = None) -> dict[str, Any]:
    """{document name: Unstract structured output} from a results JSON the user downloaded from Unstract.

    Accepts an API-deployment POST/status response, its "result" list, a single per-file item, a per-file
    output JSON ({"output": ..., "metadata": {"file_name": ...}}), a plain {prompt_key: value} object (matched
    via file_name, e.g. the uploaded JSON's name, or the only document), or {"<file name>": output}.
    Files that cannot be matched to a document are skipped; failed files map to {"error": "..."}.
    """
    documents = list(documents)
    parsed: dict[str, Any] = {}
    for item in file_items(payload):
        output, _, error = item_output(item)
        name = match_document(item_file_name(item) or file_name, documents)
        if name is None:
            continue
        parsed[name] = {"error": error} if error else output
    return parsed


def import_records(payload: Any, documents: Iterable[Any], schema: ExtractionSchema,
                   file_name: str | None = None) -> tuple[list[dict], list[str]]:
    """ImportedRecord-shaped dicts ({document_id, data, error, raw}) for POST /api/runs/import, plus warnings.

    data is the plain mapped record (keys = schema field names); raw keeps Unstract's item and the mapping notes.
    """
    documents = list(documents)
    by_name = {getattr(document, "name", None) or document["name"]: document for document in documents}
    records, warnings, seen = [], [], set()
    for item in file_items(payload):
        source_name = item_file_name(item) or file_name
        name = match_document(source_name, documents)
        if name is None:
            warnings.append(f"No document matches Unstract result for '{source_name or 'unnamed file'}'.")
            continue
        if name in seen:
            warnings.append(f"Several Unstract results match '{name}'; kept the first.")
            continue
        seen.add(name)
        document = by_name[name]
        output, metadata, error = item_output(item)
        document_id = getattr(document, "id", None) or document.get("id")
        if error:
            records.append({"document_id": document_id, "data": None, "error": str(error), "raw": item})
            continue
        data, notes = output_to_record(output, schema)
        confidences = prompt_confidences(metadata, data)
        records.append({"document_id": document_id, "data": data, "error": None,
                        "raw": {"unstract": item, "notes": notes, "confidence": confidences or None}})
    if not records and not warnings:
        warnings.append("No Unstract results found in the file.")
    return records, warnings
