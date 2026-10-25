"""NuExtract3 adapter: schema -> NuExtract template, served locally by llama.cpp (CPU) or vLLM (GPU)."""
import os

from app import nuextract_engine as engine
from app.azure import ServiceError
from app.documents import DocumentError, ParsedDocument
from app.results import Leaf, Usage, annotate_plain, merge_annotated
from app.schema import ExtractionSchema
from app.schema_suggest import build_draft, merge_field_lists, sanitise_fields
from app.solutions.base import Outcome, RunContext, SettingSpec, Solution, SuggestionOutcome
from app.text_chunks import chunk_text


def engine_settings() -> engine.NuExtractSettings:
    return engine.NuExtractSettings(
        base_url=os.environ.get("NUEXTRACT_BASE_URL", "http://127.0.0.1:8090/v1").strip(),
        model=os.environ.get("NUEXTRACT_MODEL", "numind/NuExtract3").strip() or "numind/NuExtract3",
        api_key=os.environ.get("NUEXTRACT_API_KEY", "EMPTY").strip() or "EMPTY",
        timeout=float(os.environ.get("NUEXTRACT_TIMEOUT", "1800")),
    )


ERROR_STATUS = {"invalid_input": 422, "context_length": 413, "timeout": 504, "connection": 503, "unavailable": 503}


def attach_confidences(record: dict, confidences: dict[tuple, float], offset: tuple = ()) -> dict:
    """Copy per-leaf logprob confidences (keyed by JSON path) onto Leaf objects."""
    def visit(value, path):
        if isinstance(value, Leaf):
            confidence = confidences.get(path)
            if confidence is not None:
                value.confidence = confidence
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, path + (index,))
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, path + (key,))
    visit(record, offset)
    return record


class NuExtractSolution(Solution):
    id = "nuextract"
    name = "NuExtract3"
    tagline = "Open 4B extraction model running on your machine"
    description = ("NuMind's NuExtract3 (Qwen3.5-4B based, Apache-2.0) fills a JSON template generated from "
                   "the agreed schema. It runs locally through an OpenAI-compatible server: llama.cpp on CPU "
                   "or vLLM on GPU. Token log-probabilities give per-value confidence.")
    technology = "NuExtract3 · llama.cpp / vLLM"
    runtime = "local"
    capabilities = ["Runs locally (CPU or GPU)", "Template from schema", "Confidence from token log-probs",
                    "Optional page-image input"]
    confidence_method = "Geometric mean of the model's token probabilities for each value."
    route = "/apps/nuextract"
    accent = "#72d6bb"
    document_concurrency = 1
    schema_suggestion_method = (
        "Native template generation: NuExtract3's template-generation mode drafts a JSON extraction template "
        "from each exemplar's text (with the guidance as its description), one call per exemplar since a 4B "
        "model handles one document best; the templates are converted to schema fields and merged by name.")

    def settings_spec(self) -> list[SettingSpec]:
        return [
            SettingSpec("input_mode", "Input", "select", "text", options=["text", "page images"],
                        help="Text uses the extracted text layer. Page images render PDF pages for the vision "
                             "encoder (slow on CPU; values are still highlighted via the text layer)."),
            SettingSpec("enable_thinking", "Reasoning mode", "boolean", False,
                        help="NuMind recommends non-reasoning mode first; reasoning is much slower on CPU."),
            SettingSpec("temperature", "Temperature", "number", 0.2, min=0, max=1,
                        help="0.2 recommended for extraction."),
            SettingSpec("max_tokens", "Maximum output tokens", "integer", 4096, min=256, max=32768,
                        help="Upper bound on the generated JSON per chunk."),
            SettingSpec("max_chunk_chars", "Chunk size (characters)", "integer", 24000, min=2000, max=200000,
                        help="Longer documents are split on paragraph boundaries and merged."),
            SettingSpec("use_instructions", "Send field descriptions", "boolean", True,
                        help="Pass schema descriptions as NuExtract instructions."),
            SettingSpec("extra_instructions", "Extra instructions", "text", "",
                        help="Optional guidance appended to the instructions."),
            SettingSpec("image_dpi", "Page image DPI", "integer", 120, min=60, max=200,
                        help="Only used for page-image input."),
        ]

    def suggestion_settings_spec(self) -> list[SettingSpec]:
        return [
            SettingSpec("enable_thinking", "Reasoning mode", "boolean", False,
                        help="Think before writing the template. Much slower on CPU; NuMind's own chat template "
                             "disables it for template generation."),
            SettingSpec("max_tokens", "Maximum output tokens", "integer", 4096, min=256, max=32768,
                        help="Upper bound on each generated template (and reasoning, if enabled)."),
        ]

    def suggest_schema(self, documents: list[ParsedDocument], guidance: str, settings: dict,
                       context: RunContext) -> SuggestionOutcome:
        generate = getattr(engine, "generate_template", None)
        to_fields = getattr(engine, "template_to_fields", None)
        if generate is None or to_fields is None:
            raise ServiceError("NuExtract template generation is not available in this engine build yet.", 501)
        options = self.resolve_suggestion_settings(settings)
        fields: list[dict] = []
        notes, conflicts, outputs, errors = [], [], [], []
        usage = Usage()
        for document in documents:
            context.check()
            try:
                output = generate(engine_settings(), text=document.text, description=guidance.strip(),
                                  enable_thinking=options["enable_thinking"], max_tokens=options["max_tokens"])
            except engine.NuExtractError as error:
                errors.append(error)
                notes.append(f"No template for {document.name}: {error}")
                continue
            usage = usage.add(Usage(calls=1, prompt_tokens=output.usage.get("prompt_tokens", 0),
                                    completion_tokens=output.usage.get("completion_tokens", 0)))
            outputs.append({"document": document.name, "template": output.template, "raw": output.raw,
                            "reasoning": output.reasoning, "elapsed_seconds": output.elapsed_seconds})
            try:
                proposed = sanitise_fields(to_fields(output.template)) if output.template else []
            except ValueError as error:
                proposed = []
                notes.append(f"Could not convert the template for {document.name}: {error}")
            if not proposed:
                notes.append(f"NuExtract returned no usable template for {document.name}.")
                continue
            fields, found = merge_field_lists(fields, proposed)
            conflicts += found
        if not fields:
            if errors:
                raise ServiceError(str(errors[0]), ERROR_STATUS.get(errors[0].kind, 502))
            raise ServiceError("NuExtract did not return a usable template for any exemplar.")
        calls = len(documents)
        summary = (f"NuExtract template generation ran once per exemplar ({calls} call{'s' if calls != 1 else ''})"
                   + ("; the field lists were merged by name (the first type wins)." if usage.calls > 1 else "."))
        notes = [summary] + [f"Conflict at {conflict}" for conflict in conflicts[:15]] + notes
        draft = build_draft("NuExtract3 template schema",
                            f"Merged from NuExtract3 templates for: {', '.join(d.name for d in documents)}.",
                            fields)
        return SuggestionOutcome(draft=draft, notes=notes, usage=usage,
                                 raw={"description": guidance.strip(), "outputs": outputs})

    def status(self) -> dict:
        settings = engine_settings()
        status = engine.server_status(settings)
        detail = status.get("detail") or ""
        if not status.get("available"):
            detail = f"NuExtract server not reachable at {settings.base_url}. {detail}".strip()
        return {"ready": bool(status.get("available")), "detail": detail,
                "model": status.get("model") or settings.model}

    def model_label(self, settings: dict) -> str:
        mode = "images" if settings.get("input_mode") == "page images" else "text"
        return f"{self.status().get('model', 'NuExtract3')} · {mode}{' · thinking' if settings.get('enable_thinking') else ''}"

    def prepare(self, document: ParsedDocument, settings: dict, context: RunContext) -> list[str]:
        if settings.get("input_mode") != "page images":
            return []
        if not document.name.lower().endswith(".pdf"):
            return ["Page-image input applies only to PDFs; using the text layer."]
        images = context.store.page_images(document.id, settings.get("image_dpi", 120))
        return [f"Rendered {len(images)} page image(s) at {settings.get('image_dpi', 120)} dpi."]

    def preview(self, schema: ExtractionSchema, settings: dict) -> dict[str, str]:
        import json
        fields = [field.model_dump() for field in schema.fields]
        instructions = engine.schema_instructions(fields, settings.get("extra_instructions", "")) \
            if settings.get("use_instructions", True) else settings.get("extra_instructions", "")
        return {"NuExtract template": json.dumps(engine.schema_to_template(fields), indent=2),
                "Instructions": instructions or "(none)"}

    def extract(self, document: ParsedDocument, schema: ExtractionSchema, settings: dict,
                context: RunContext) -> Outcome:
        fields = [field.model_dump() for field in schema.fields]
        template = engine.schema_to_template(fields)
        instructions = engine.schema_instructions(fields, settings.get("extra_instructions", "")) \
            if settings.get("use_instructions", True) else settings.get("extra_instructions", "")
        options = dict(enable_thinking=settings.get("enable_thinking", False),
                       temperature=settings.get("temperature", 0.2), max_tokens=settings.get("max_tokens", 4096))
        notes: list[str] = []
        usage = Usage()
        records, raws = [], []
        image_mode = settings.get("input_mode") == "page images" and document.name.lower().endswith(".pdf")
        try:
            if image_mode:
                images = context.store.page_images(document.id, settings.get("image_dpi", 120))
                parts = [None]
            else:
                images = None
                parts = chunk_text(document.text, settings.get("max_chunk_chars", 24000))
                if len(parts) > 1:
                    notes.append(f"Split into {len(parts)} chunks and merged.")
            for part in parts:
                context.check()
                output = engine.extract(engine_settings(), text=part, images=images, template=template,
                                        instructions=instructions, **options)
                usage = usage.add(Usage(calls=1, prompt_tokens=output.usage.get("prompt_tokens", 0),
                                        completion_tokens=output.usage.get("completion_tokens", 0)))
                if output.data is None:
                    raise ServiceError("NuExtract returned output that is not valid JSON.")
                record = annotate_plain(output.data, schema.fields)
                records.append(attach_confidences(record, output.confidences))
                raws.append({"answer": output.raw, "reasoning": output.reasoning,
                             "finish_reason": output.finish_reason, "elapsed_seconds": output.elapsed_seconds})
                if not output.confidences:
                    notes.append("The server returned no token log-probabilities, so confidence is unavailable.")
        except engine.NuExtractError as error:
            raise ServiceError(str(error)) from error
        except DocumentError as error:
            raise ServiceError(str(error), error.status) from error
        record = records[0] if len(records) == 1 else merge_annotated(schema.fields, records)
        return Outcome(record=record, usage=usage, notes=list(dict.fromkeys(notes)),
                       raw={"template": template, "instructions": instructions, "outputs": raws})
