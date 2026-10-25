"""NuExtract3 adapter: schema -> NuExtract template, served locally by llama.cpp (CPU) or vLLM (GPU)."""
import os

from app import nuextract_engine as engine
from app.azure import ServiceError
from app.documents import DocumentError, ParsedDocument
from app.results import Leaf, Usage, annotate_plain, merge_annotated
from app.schema import ExtractionSchema
from app.solutions.base import Outcome, RunContext, SettingSpec, Solution
from app.text_chunks import chunk_text


def engine_settings() -> engine.NuExtractSettings:
    return engine.NuExtractSettings(
        base_url=os.environ.get("NUEXTRACT_BASE_URL", "http://127.0.0.1:8090/v1").strip(),
        model=os.environ.get("NUEXTRACT_MODEL", "numind/NuExtract3").strip() or "numind/NuExtract3",
        api_key=os.environ.get("NUEXTRACT_API_KEY", "EMPTY").strip() or "EMPTY",
        timeout=float(os.environ.get("NUEXTRACT_TIMEOUT", "1800")),
    )


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
