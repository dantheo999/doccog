"""Cross-run comparison: align values per document/field and score agreement.

Without ground truth, agreement is measured against a chosen reference run, or against the other
runs (consensus). Cells where every compared value is empty are excluded from scores.
"""
from functools import lru_cache
from statistics import mean

from pydantic import JsonValue
from rapidfuzz import fuzz

from app.grounding import TRANSLATE
from app.results import FieldValue, Run
from app.schema import SchemaField, is_empty, parse_date


MATCH = 0.8          # similarity counted as agreement in summaries
PAIR_THRESHOLD = 0.5  # minimum similarity to align two list items


TYPOGRAPHY = str.maketrans(TRANSLATE)


@lru_cache(maxsize=65_536)
def _normalised(value: str) -> str:
    """Case, quote/dash and whitespace insensitive form (same rules as the grounder, no offset map)."""
    return " ".join(value.translate(TYPOGRAPHY).lower().split())


def _text(value) -> str:
    return _normalised(str(value))


def scalar_similarity(a: JsonValue, b: JsonValue, field: SchemaField) -> float:
    if is_empty(a) and is_empty(b):
        return 1.0
    if is_empty(a) or is_empty(b):
        return 0.0
    if field.type == "date":
        left, right = parse_date(str(a)), parse_date(str(b))
        if left and right:
            return 1.0 if left == right else 0.0
    if field.type in ("number", "integer") and isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return 1.0 if abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a))) else 0.0
    if field.type in ("enum", "boolean"):
        return 1.0 if _text(a) == _text(b) else 0.0
    left, right = _text(a), _text(b)
    if left == right:
        return 1.0
    score = max(fuzz.ratio(left, right), 0.95 * fuzz.token_set_ratio(left, right))
    return round(score / 100, 3)


def _primary(field: SchemaField) -> str | None:
    for child in field.fields:
        if child.type in ("verbatim", "text") and not child.array:
            return child.name
    return None


def item_similarity(a: dict, b: dict, field: SchemaField) -> float:
    """Weighted mean over children with a value on either side (the first text child counts double)."""
    primary = _primary(field)
    total = weight = 0.0
    for child in field.fields:
        left, right = (a or {}).get(child.name), (b or {}).get(child.name)
        if is_empty(left) and is_empty(right):
            continue
        child_weight = 2.0 if child.name == primary else 1.0
        total += child_weight * value_similarity(left, right, child)
        weight += child_weight
    return total / weight if weight else 1.0


def align(left: list, right: list, similarity) -> list[tuple[int, int, float]]:
    """Greedy one-to-one alignment by descending similarity."""
    pairs = sorted(((similarity(a, b), i, j) for i, a in enumerate(left) for j, b in enumerate(right)),
                   reverse=True)
    used_left, used_right, matches = set(), set(), []
    for score, i, j in pairs:
        if score < PAIR_THRESHOLD:
            break
        if i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)
        matches.append((i, j, score))
    return sorted(matches)


def value_similarity(a: JsonValue, b: JsonValue, field: SchemaField) -> float:
    if field.array:
        left, right = list(a or []), list(b or [])
        if not left and not right:
            return 1.0
        single = field.model_copy(update={"array": False})
        matches = align(left, right, lambda x, y: value_similarity(x, y, single))
        return round(2 * sum(score for _, _, score in matches) / (len(left) + len(right)), 3)
    if field.type == "object":
        return round(item_similarity(a or {}, b or {}, field), 3)
    return scalar_similarity(a, b, field)


def conforms(value: JsonValue, field: SchemaField, array: bool | None = None) -> bool:
    """Whether a value has the field's shape (a list for arrays, a dict for objects, else a scalar),
    recursively. Runs of another schema version may hold a field under the same name in another shape."""
    if value is None:
        return True
    if field.array if array is None else array:
        return isinstance(value, list) and all(conforms(item, field, False) for item in value)
    if field.type == "object":
        return isinstance(value, dict) and all(conforms(value.get(child.name), child) for child in field.fields)
    return not isinstance(value, (list, dict))


def _display(value: JsonValue, field: SchemaField) -> str:
    if is_empty(value):
        return ""
    if field.array:
        return f"{len(value)} item{'s' if len(value) != 1 else ''}"
    if field.type == "object":
        return " · ".join(str(item) for item in value.values() if not is_empty(item) and not isinstance(item, (list, dict)))
    return str(value)


def _leaf_stats(values: list[FieldValue], name: str) -> dict:
    leaves = [value for value in values if value.path and value.path[0] == name]
    confidences = [value.confidence for value in leaves if value.confidence is not None]
    first = leaves[0] if len(leaves) == 1 and len(leaves[0].path) == 1 else None
    return {
        "confidence": round(mean(confidences), 3) if confidences else None,
        "grounded": sum(value.grounding != "none" for value in leaves), "leaves": len(leaves),
        "grounding": first.grounding if first else None,
        "start": first.start if first else None, "end": first.end if first else None,
    }


def aligned_rows(field: SchemaField, items: dict[str, list], pivot: str) -> list[dict]:
    """Rows of matched object/list items across runs, pivot run's order first."""
    single = field.model_copy(update={"array": False})
    rows = [{"cells": {pivot: item}, "indexes": {pivot: index}, "scores": {}}
            for index, item in enumerate(items.get(pivot, []))]
    for run_id, run_items in items.items():
        if run_id == pivot:
            continue
        matches = align(items.get(pivot, []), run_items, lambda x, y: value_similarity(x, y, single))
        matched = set()
        for i, j, score in matches:
            rows[i]["cells"][run_id] = run_items[j]
            rows[i]["indexes"][run_id] = j
            rows[i]["scores"][run_id] = score
            matched.add(j)
        for j, item in enumerate(run_items):
            if j not in matched:
                rows.append({"cells": {run_id: item}, "indexes": {run_id: j}, "scores": {}})
    return rows


def compare_runs(runs: list[Run], reference_id: str | None = None) -> dict:
    if not runs:
        return {"runs": [], "documents": [], "fields": [], "cells": [], "warnings": []}
    reference = next((run for run in runs if run.id == reference_id), None)
    base = reference or runs[0]
    fields = base.extraction_schema.fields
    warnings = []
    schemas = {(run.extraction_schema.id, run.extraction_schema.version) for run in runs}
    if len(schemas) > 1:
        warnings.append("Runs use different schema versions; fields are compared by name using "
                        f"{'the reference' if reference else 'the first'} run's schema.")
    documents: dict[str, dict] = {}
    for run in runs:
        for document in run.documents:
            documents.setdefault(document.document_id, {"id": document.document_id,
                                                        "name": document.document_name, "status": {}})
            documents[document.document_id]["status"][run.id] = document.status
    by_run = {run.id: {document.document_id: document for document in run.documents} for run in runs}
    reshaped: dict[str, dict[str, str]] = {}  # field -> run id -> label, for values of another shape

    cells = []
    for document_id in documents:
        for field in fields:
            entry = {"document_id": document_id, "field": field.name, "values": {}, "agreement": {}}
            present = {}
            for run in runs:
                result = by_run[run.id].get(document_id)
                if result is None or result.status != "done" or result.data is None:
                    continue
                value = result.data.get(field.name)
                if not conforms(value, field):
                    reshaped.setdefault(field.name, {})[run.id] = f"{run.label} (v{run.extraction_schema.version})"
                    entry.setdefault("incomparable", []).append(run.id)
                    continue
                present[run.id] = value
                entry["values"][run.id] = {"value": value, "display": _display(value, field),
                                           "count": len(value) if field.array and isinstance(value, list) else None,
                                           **_leaf_stats(result.values, field.name)}
            for run_id, value in present.items():
                others = ([reference.id] if reference and run_id != reference.id and reference.id in present
                          else [] if reference else [other for other in present if other != run_id])
                scores = [value_similarity(value, present[other], field) for other in others
                          if not (is_empty(value) and is_empty(present[other]))]
                entry["agreement"][run_id] = round(mean(scores), 3) if scores else None
            if field.array:
                pivot = reference.id if reference and reference.id in present else max(
                    present, key=lambda key: len(present[key] or []), default=None)
                if pivot is not None:
                    entry["pivot"] = pivot
                    entry["rows"] = aligned_rows(field, {key: list(value or []) for key, value in present.items()}, pivot)
            cells.append(entry)
    for name, labels in reshaped.items():
        warnings.append(f"“{name}” has a different shape in {', '.join(labels.values())}; "
                        "those values are not compared.")

    run_stats = []
    for run in runs:
        scored = [cell["agreement"].get(run.id) for cell in cells if cell["agreement"].get(run.id) is not None]
        filled = [cell for cell in cells if run.id in cell["values"]]
        field_scores = {}
        for field in fields:
            items = [cell for cell in cells if cell["field"] == field.name and run.id in cell["values"]]
            agreements = [cell["agreement"][run.id] for cell in items if cell["agreement"].get(run.id) is not None]
            field_scores[field.name] = {
                "filled": sum(not is_empty(cell["values"][run.id]["value"]) for cell in items),
                "documents": len(items),
                "agreement": round(mean(agreements), 3) if agreements else None,
                "agree_count": sum(score >= MATCH for score in agreements), "scored": len(agreements),
            }
        run_stats.append({
            **run.summary(),
            "agreement": round(mean(scored), 3) if scored else None,
            "agree_rate": round(sum(score >= MATCH for score in scored) / len(scored), 3) if scored else None,
            "fill_rate": round(sum(not is_empty(cell["values"][run.id]["value"]) for cell in filled)
                               / len(filled), 3) if filled else None,
            "fields": field_scores,
        })
    return {
        "reference": reference.id if reference else None,
        "runs": run_stats,
        "documents": sorted(documents.values(), key=lambda item: item["name"].casefold()),
        "fields": [field.model_dump() for field in fields],
        "cells": cells,
        "warnings": warnings,
        "thresholds": {"match": MATCH, "pair": PAIR_THRESHOLD},
    }
