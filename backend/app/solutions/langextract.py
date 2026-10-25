"""LangExtract adapter: schema -> extraction classes, grounded findings -> schema records."""
from dataclasses import dataclass, field as dataclass_field
import re

from pydantic import JsonValue, ValidationError

from app.azure import REASONING_EFFORTS, ServiceError
from app.cognition import CognitionError, extract_document, generate_recipe
from app.documents import ParsedDocument
from app.grounding import normalise
from app.models import ExtractionItem, ExtractionOptions, Recipe
from app.results import Leaf, Usage, plain
from app.schema import ExtractionSchema, FieldType, SchemaField, TYPE_LABELS, coerce_scalar, is_empty, parse_date
from app.schema_suggest import build_draft, snake_name, unique_name
from app.settings import load_settings
from app.solutions.base import Outcome, RunContext, SettingSpec, Solution, SuggestionOutcome


SEPARATOR = "__"
# LangExtract reads any key ending in this suffix as the attributes of another class.
ATTRIBUTE_SUFFIX = "_attributes"


@dataclass
class ClassSpec:
    """One LangExtract extraction class and where its findings land in the schema record."""
    name: str
    path: list[str]
    field: SchemaField
    text_field: SchemaField | None = None
    attributes: list[SchemaField] = dataclass_field(default_factory=list)
    parent: str | None = None

    @property
    def is_object(self) -> bool:
        return self.field.type == "object"


PRIMARY_NAMES = ("name", "title", "description", "action", "decision", "obligation", "requirement", "summary",
                 "text", "subject", "label", "item")


def _primary(field: SchemaField) -> SchemaField | None:
    """The single-value child whose literal span best identifies an item; it becomes extraction_text.
    None when there is no such child (the item is then identified by a short phrase)."""
    leaves = [child for child in field.fields if child.type in ("verbatim", "text") and not child.array]
    for wanted in PRIMARY_NAMES:
        for child in leaves:
            if child.name == wanted or child.name.endswith(f"_{wanted}"):
                return child
    if leaves:
        return next((child for child in leaves if child.type == "verbatim"), leaves[0])
    scalars = [child for child in field.fields if child.type != "object" and not child.array]
    return scalars[0] if scalars else None


def class_name(path: list[str]) -> str:
    """The class for a field path. A name ending in the attribute suffix would not parse, so it gets a
    "__1" suffix (no child field can be named "1", so it is not the class of a nested field)."""
    name = SEPARATOR.join(path)
    return f"{name}{SEPARATOR}1" if name.endswith(ATTRIBUTE_SUFFIX) else name


def class_plan(fields: list[SchemaField], path: list[str] | None = None,
               parent: str | None = None) -> list[ClassSpec]:
    """Object fields become one class per item (primary child = extraction_text, other scalars =
    attributes). Leaves become their own class; nested arrays become child classes with a
    `parent` attribute naming the containing item."""
    path = path or []
    specs: list[ClassSpec] = []
    for field in fields:
        current = path + [field.name]
        name = class_name(current)
        if field.type == "object":
            primary = _primary(field)
            attributes = [child for child in field.fields if child is not primary
                          and child.type != "object" and (not child.array or child.type in ("enum", "boolean"))]
            specs.append(ClassSpec(name, current, field, primary, attributes, parent))
            nested = [child for child in field.fields if child is not primary and child not in attributes]
            specs.extend(class_plan(nested, current, name))
        else:
            specs.append(ClassSpec(name, current, field, parent=parent))
    return specs


SPAN_LABELS = {"text": "the relevant text", "verbatim": "the exact text", "date": "the date as written",
               "number": "the number as written", "integer": "the number as written",
               "boolean": "the phrase that states it", "enum": "the phrase that indicates it"}


def _describe(field: SchemaField) -> str:
    kind = TYPE_LABELS[field.type]
    if field.type == "enum":
        kind += " (" + ", ".join(field.options) + ")"
    if field.array and field.type != "object":
        kind = f"list of {kind}"
    return f"{kind}{' — ' + field.description if field.description else ''}"


def class_guide(schema: ExtractionSchema | list[SchemaField]) -> str:
    fields = schema.fields if isinstance(schema, ExtractionSchema) else schema
    lines = ["Agreed schema classes. Use only these extraction_class names; extraction_text is always "
             "a literal source span; give attributes only with the listed keys."]
    for spec in class_plan(fields):
        parent_rule = (f' attribute "parent" = the exact extraction_text of the {spec.parent} item it belongs to;'
                       if spec.parent else "")
        if spec.is_object:
            subject = f"the {spec.text_field.name}" if spec.text_field else "a short identifying phrase"
            many = ("one extraction per item" if spec.field.array
                    else "at most one extraction" + (f" per {spec.parent} item" if spec.parent else ""))
            line = (f"- {spec.name}: {many}"
                    f"{' (' + spec.field.description + ')' if spec.field.description else ''}; "
                    f"extraction_text = {subject}"
                    f"{' (' + spec.text_field.description.rstrip('.') + ')' if spec.text_field and spec.text_field.description else ''};"
                    f"{parent_rule}")
            if spec.attributes:
                line += " attributes: " + "; ".join(f'"{child.name}" = {_describe(child)}' for child in spec.attributes)
        else:
            many = "one extraction per value" if spec.field.array else "at most one extraction"
            normalised = (f' attribute "value" = {TYPE_LABELS[spec.field.type]}'
                          + (f" ({', '.join(spec.field.options)})" if spec.field.type == "enum" else "") + ";"
                          if spec.field.type not in ("text", "verbatim") else "")
            line = (f"- {spec.name}: {many}{' — ' + spec.field.description if spec.field.description else ''}; "
                    f"extraction_text = {SPAN_LABELS[spec.field.type]};{parent_rule}{normalised}")
        lines.append(line.rstrip(";"))
    return "\n".join(lines)


def _key(text: str) -> str:
    return normalise(text)[0].strip()


def _leaf(field: SchemaField, item: ExtractionItem, from_attribute: str | None = None,
          normalised: bool = True) -> Leaf:
    """A value from a finding: an attribute, or extraction_text (or its normalised "value" attribute,
    unless "value" is another field's attribute)."""
    attributes = item.attributes or {}
    if from_attribute is not None:
        return Leaf(coerce_scalar(attributes.get(from_attribute), field))
    raw = (attributes.get("value") if normalised and field.type not in ("text", "verbatim") and "value" in attributes
           else item.extraction_text)
    return Leaf(coerce_scalar(raw, field), quote=item.extraction_text, start=item.start, end=item.end)


def assemble(fields: list[SchemaField], items: list[ExtractionItem]) -> tuple[dict, list[str]]:
    """Rebuild a nested Leaf record from flat LangExtract findings (in document order)."""
    plan = {spec.name: spec for spec in class_plan(fields)}
    notes: list[str] = []
    unknown: dict[str, int] = {}
    # containers[class name] -> list of (start, item dict) for object classes
    containers: dict[str, list[tuple[int, dict]]] = {name: [] for name, spec in plan.items() if spec.is_object}
    root: dict = {}

    def empty_item(spec: ClassSpec) -> dict:
        return {child.name: ([] if child.array else None) for child in spec.field.fields}

    def target(spec: ClassSpec, item: ExtractionItem) -> dict:
        if spec.parent is None:
            return root
        parents = containers[spec.parent]
        reference = (item.attributes or {}).get("parent")
        parent_spec = plan[spec.parent]
        if isinstance(reference, str) and parent_spec.text_field is not None:
            wanted = _key(reference)
            for _, candidate in parents:
                value = candidate.get(parent_spec.text_field.name)
                if isinstance(value, Leaf) and isinstance(value.quote, str) and _key(value.quote) == wanted:
                    return candidate
        position = item.start if item.start is not None else None
        if parents:
            if position is None:
                return parents[-1][1]
            preceding = [entry for entry in parents if entry[0] <= position]
            return (preceding[-1] if preceding else parents[0])[1]
        holder = attach(parent_spec, empty_item(parent_spec), item.start or 0)
        notes.append(f"Created an unnamed {spec.parent} item to hold {spec.name} findings.")
        return holder

    def place(spec: ClassSpec, container: dict, record: dict, start: int) -> dict:
        """Add an object record to its container. A repeat of a single-value object fills the gaps of
        the first one, and later findings that belong to it (by position) attach to that first one."""
        name = spec.path[-1]
        if spec.field.array:
            if not isinstance(container.get(name), list):
                container[name] = []
            container[name].append(record)
        elif isinstance(container.get(name), dict):
            existing = container[name]
            for key, value in record.items():
                if is_empty(plain(existing.get(key))) and not is_empty(plain(value)):
                    existing[key] = value
            notes.append(f"Merged repeated {spec.name} findings into one item (the schema allows one).")
            record = existing
        else:
            container[name] = record
        containers[spec.name].append((start, record))
        return record

    def attach(spec: ClassSpec, record: dict, start: int) -> dict:
        container = target(spec, ExtractionItem(id="", extraction_class=spec.name, extraction_text="",
                                                attributes={}, start=start, end=None, alignment_status=None,
                                                grounding="ungrounded")) if spec.parent else root
        return place(spec, container, record, start)

    ordered = sorted(items, key=lambda item: (item.start if item.start is not None else 10**9))
    for item in ordered:
        spec = plan.get(item.extraction_class)
        if spec is None:
            unknown[item.extraction_class] = unknown.get(item.extraction_class, 0) + 1
            continue
        start = item.start if item.start is not None else 10**9
        if spec.is_object:
            record = empty_item(spec)
            if spec.text_field is not None:
                record[spec.text_field.name] = _leaf(
                    spec.text_field, item, normalised=all(child.name != "value" for child in spec.attributes))
            for child in spec.attributes:
                raw = (item.attributes or {}).get(child.name)
                if child.array:
                    values = raw if isinstance(raw, list) else ([] if raw is None else [raw])
                    single = child.model_copy(update={"array": False})
                    record[child.name] = [Leaf(coerce_scalar(value, single)) for value in values]
                else:
                    record[child.name] = _leaf(child, item, from_attribute=child.name)
            place(spec, root if spec.parent is None else target(spec, item), record, start)
        else:
            leaf = _leaf(spec.field.model_copy(update={"array": False}), item)
            if leaf.value is None:
                continue
            container = root if spec.parent is None else target(spec, item)
            name = spec.path[-1]
            if spec.field.array:
                container.setdefault(name, [])
                container[name].append(leaf)
            elif not isinstance(container.get(name), Leaf) or container[name].value is None:
                container[name] = leaf
            else:
                notes.append(f"Kept the first {spec.name} finding; ignored a later one.")
    for name, count in unknown.items():
        notes.append(f"Ignored {count} finding(s) of class '{name}', which is not in the schema.")
    return root, list(dict.fromkeys(notes))


# --- Derived schema suggestion: recipe classes and attributes -> schema fields -------------

DEFAULT_SUGGESTION_GOAL = (
    "Extract the recurring, structured information in documents of this type: people and organisations with "
    "their roles, dates, decisions, actions or obligations with owners and due dates, amounts and other key "
    "facts. Record the details of each item as attributes.")
# Numbers that survive a round trip: no leading zeros ("007") and no trailing fractional zeros ("4.10").
NUMBER = re.compile(r"-?(?:[1-9]\d{0,2}(?:,\d{3})+|0|[1-9]\d*)(?:\.\d*[1-9])?")
# Digits with separators that are not such numbers: item numbers (4.10, 4.1.2) and references (007).
NUMBER_LIKE = re.compile(r"-?\d[\d,]*(?:\.\d+)*")
BOOLEAN_WORDS = ("true", "false", "yes", "no")
MAX_CHILDREN = 40


def _field_name(name: str, seen: set[str]) -> str:
    return unique_name(snake_name(re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))), seen)


def _number(value: JsonValue) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and NUMBER.fullmatch(value.strip()):
        return float(value.strip().replace(",", ""))
    return None


def infer_type(values: list[JsonValue]) -> tuple[FieldType, bool]:
    """(type, array) for an attribute from the values seen in the recipe's annotations."""
    present = [value for value in values if value is not None and value != "" and value != []]
    if any(isinstance(value, list) for value in present):
        return "text", True
    if not present:
        return "text", False
    if all(isinstance(value, bool) or (isinstance(value, str) and value.strip().casefold() in BOOLEAN_WORDS)
           for value in present):
        return "boolean", False
    numbers = [_number(value) for value in present]
    if all(number is not None for number in numbers):
        return ("integer" if all(number.is_integer() for number in numbers) else "number"), False
    if all(isinstance(value, str) and parse_date(value) for value in present):
        return "date", False
    if all(_number(value) is not None or (isinstance(value, str) and NUMBER_LIKE.fullmatch(value.strip()))
           for value in present):
        return "verbatim", False  # numbering or references: a number type would lose 4.10 vs 4.1, or 007
    return "text", False


def _examples(values: list[JsonValue]) -> str:
    seen = list(dict.fromkeys(str(value)[:40] for value in values
                              if value not in (None, "", []) and not isinstance(value, (dict, list))))[:3]
    return f"; e.g. {', '.join(repr(value) for value in seen)}" if seen else ""


def schema_from_recipe(recipe: Recipe) -> tuple[list[dict], list[str]]:
    """Schema fields inferred from a recipe's extraction classes (in first-seen order).

    A class with attributes becomes an object (child "text" = the annotated span, one child per attribute
    key); a class without attributes a verbatim value. Either is a list when the class occurs more than
    once in any example."""
    classes: dict[str, dict] = {}
    for example in recipe.examples:
        per_example: dict[str, int] = {}
        for extraction in example.extractions:
            name = extraction.extraction_class
            entry = classes.setdefault(name, {"count": 0, "most": 0, "attributes": {}})
            entry["count"] += 1
            per_example[name] = per_example.get(name, 0) + 1
            for key, value in (extraction.attributes or {}).items():
                entry["attributes"].setdefault(key, []).append(value)
        for name, count in per_example.items():
            classes[name]["most"] = max(classes[name]["most"], count)
    fields, notes, seen = [], [], set()
    for name, entry in classes.items():
        plural = "s" if entry["count"] != 1 else ""
        field = {"name": _field_name(name, seen), "array": entry["most"] > 1, "options": [], "fields": [],
                 "description": f"From LangExtract class {name} ({entry['count']} annotation{plural})."}
        if not entry["attributes"]:
            fields.append({**field, "type": "verbatim"})
            continue
        children, child_names = [{"name": "text", "type": "verbatim", "array": False, "options": [], "fields": [],
                                  "description": f"The exact text annotated as {name}."}], {"text"}
        for key, values in entry["attributes"].items():
            kind, array = infer_type(values)
            count = sum(value not in (None, "", []) for value in values)
            children.append({"name": _field_name(key, child_names), "type": kind, "array": array, "options": [],
                             "fields": [], "description": f"From attribute '{key}' of {name} "
                                                          f"({count} value{'s' if count != 1 else ''}{_examples(values)})."[:1_000]})
        if len(children) > MAX_CHILDREN:
            notes.append(f"Class {name} has {len(children) - 1} attributes; kept the first {MAX_CHILDREN - 1}.")
        fields.append({**field, "type": "object", "fields": children[:MAX_CHILDREN]})
    singles = [name for name, entry in classes.items() if entry["most"] == 1 and entry["count"] > 1]
    if singles:
        notes.append(f"{', '.join(singles)} occurred at most once per example, so became single values; make them "
                     "lists if a document can contain several (recipe examples are short passages).")
    if len(fields) > 60:
        notes.append(f"The recipe has {len(fields)} classes; kept the first 60.")
    annotations = sum(entry["count"] for entry in classes.values())
    notes.append(f"The recipe has {len(recipe.examples)} example(s) with {annotations} annotations "
                 f"in {len(classes)} class{'es' if len(classes) != 1 else ''}.")
    return fields[:60], notes


OPTION_SPECS = [
    SettingSpec("chunk_size", "Target chunk size (characters)", "integer", 6000, min=500, max=20000,
                help="Smaller chunks mean more requests and repeated examples; larger chunks risk missed items."),
    SettingSpec("max_workers", "Maximum concurrent LLM requests", "integer", 4, min=1, max=16,
                help="Parallel chunk requests per document, within Azure quota."),
    SettingSpec("extraction_passes", "Extraction passes", "integer", 1, min=1, max=5,
                help="Extra sequential passes can improve recall at extra cost."),
    SettingSpec("context_window_chars", "Previous-chunk context (characters)", "integer", 0, min=0, max=6000,
                help="Trailing source text from the previous chunk; 0 disables."),
]


class LangExtractSolution(Solution):
    id = "langextract"
    name = "LangExtract"
    tagline = "Few-shot recipes with exact source grounding"
    description = ("Google's LangExtract library drives Azure OpenAI with instructions and annotated "
                   "exemplar documents (a recipe), chunks the target, and aligns every finding to an exact "
                   "character span. Findings are mapped onto the agreed schema.")
    technology = "Google LangExtract + Azure OpenAI"
    runtime = "cloud"
    capabilities = ["Exact span alignment", "Few-shot recipe from exemplars", "Recipe optimisation",
                    "Chunked long-document processing"]
    confidence_method = "Not provided by LangExtract; grounding status (exact / fuzzy / none) is the quality signal."
    route = "/apps/langextract"
    accent = "#aa98ff"
    document_concurrency = 2
    schema_suggestion_method = (
        "Derived (LangExtract has no schema-suggestion feature): a LangExtract recipe is generated from the "
        "exemplars without a schema, using the guidance as its goal, and a schema is inferred from the recipe's "
        "extraction classes (objects or verbatim values) and their attributes (child fields typed from the "
        "annotated values).")

    def settings_spec(self) -> list[SettingSpec]:
        return OPTION_SPECS + [
            SettingSpec("reasoning_effort", "Reasoning effort", "select", "low", options=list(REASONING_EFFORTS),
                        help="Passed to the Azure deployment for every chunk request."),
            SettingSpec("drop_ungrounded", "Drop findings LangExtract could not align", "boolean", True,
                        help="Unaligned findings are often copied from the few-shot examples rather than the target "
                             "document. They stay visible in the raw output either way."),
        ]

    def validate_settings(self, settings: dict) -> dict:
        settings = dict(settings)
        recipe = settings.pop("recipe", None)
        if recipe is None:
            raise ServiceError("LangExtract runs need a recipe. Generate or import one first.", 422)
        try:
            recipe = Recipe.model_validate(recipe)
        except ValidationError as error:
            raise ServiceError("The recipe is invalid. Regenerate or re-import it.", 422) from error
        validated = super().validate_settings(settings)
        return {**validated, "recipe": recipe.model_dump()}

    def status(self) -> dict:
        try:
            settings = load_settings()
        except ValueError as error:
            return {"ready": False, "detail": str(error), "model": ""}
        if not settings.configured:
            return {"ready": False, "detail": "Azure OpenAI is not configured.", "model": ""}
        return {"ready": True, "detail": "Azure OpenAI configured.", "model": settings.deployment}

    def model_label(self, settings: dict) -> str:
        return f"{self.status().get('model', '')} · reasoning {settings.get('reasoning_effort', 'default')}"

    def preview(self, schema: ExtractionSchema, settings: dict) -> dict[str, str]:
        return {"Schema class guide (appended to the recipe instructions)": class_guide(schema)}

    def suggest_schema(self, documents: list[ParsedDocument], guidance: str, settings: dict,
                       context: RunContext) -> SuggestionOutcome:
        goal = guidance.strip() or DEFAULT_SUGGESTION_GOAL
        try:
            recipe = generate_recipe(documents, goal, load_settings())
        except CognitionError as error:
            raise ServiceError(str(error), error.status) from error
        fields, notes = schema_from_recipe(recipe)
        draft = build_draft("LangExtract recipe schema",
                            f"Derived from the classes of a LangExtract recipe generated for: {goal}", fields)
        notes = ["LangExtract has no schema-suggestion feature: this schema is derived from the extraction classes "
                 "and attributes of a recipe generated from the exemplars.",
                 "LangExtract classes only nest one level (a class and its attributes), so deeper structure "
                 "is not represented."] + notes + [
                 "LangExtract recipe generation does not report token usage, so usage shows zero."]
        return SuggestionOutcome(draft=draft, notes=notes, usage=Usage(),
                                 raw={"goal": goal, "recipe": recipe.model_dump(mode="json")})

    def extract(self, document: ParsedDocument, schema: ExtractionSchema, settings: dict,
                context: RunContext) -> Outcome:
        recipe = Recipe.model_validate(settings["recipe"])
        options = ExtractionOptions(**{spec.key: settings[spec.key] for spec in OPTION_SPECS})
        try:
            result = extract_document(document, recipe, load_settings(), options,
                                      class_guide=class_guide(schema), reasoning_effort=settings.get("reasoning_effort"))
        except CognitionError as error:
            raise ServiceError(str(error), error.status) from error
        items = result.extractions
        notes: list[str] = []
        if settings.get("drop_ungrounded", True):
            items = [item for item in items if item.start is not None]
            dropped = len(result.extractions) - len(items)
            if dropped:
                notes.append(f"Dropped {dropped} finding(s) LangExtract could not align to this document "
                             "(often copied from the examples); see the raw output.")
        record, assembly_notes = assemble(schema.fields, items)
        notes += assembly_notes
        usage = Usage(**(result.usage or {}))
        return Outcome(record=record, usage=usage, notes=notes, raw=result.model_dump(mode="json", exclude={"text"}))
