/** Pure helpers for schema suggestions: shared constants, stats, settings storage and the field-path comparison. */
import type { SchemaDraft, SchemaField, SchemaSuggestion, SettingSpec, SharedSuggestionResult, SolutionInfo, Usage } from "../types";
import { defaults, storedValueFits } from "./SettingsForm";

/** The solution id the backend stores the shared, solution-neutral suggestion under. */
export const SHARED_ID = "shared";
export const SHARED_NAME = "Shared suggestion";
export const SHARED_METHOD = "Solution-neutral: one Azure OpenAI call reads the exemplars' full text and returns a schema as strict JSON.";
/** WorkspacePage's working draft (a SchemaDraft); the Solution page writes it for "Open in Workspace". */
export const SCHEMA_DRAFT_KEY = "doc-cognition.schema-draft.v1";
export const MAX_EXEMPLARS = 5;
export const MAX_GUIDANCE = 4000;

const TYPE_LABELS: Record<SchemaField["type"], string> = {
  text: "Text", verbatim: "Verbatim", date: "Date", number: "Number", integer: "Integer",
  boolean: "Yes / no", enum: "Choice", object: "Object",
};

export function typeLabel(field: SchemaField): string {
  return TYPE_LABELS[field.type] ?? field.type;
}

/** Type identity used to decide whether two suggestions agree on a field. */
export function typeKey(field: SchemaField): string {
  return `${field.type}${field.array ? "[]" : ""}`;
}

const children = (field: SchemaField): SchemaField[] => Array.isArray(field.fields) ? field.fields : [];

/** Levels of nesting: 1 for a flat schema, 2 when objects hold scalars, and so on. 0 for no fields. */
export function schemaDepth(fields: SchemaField[]): number {
  return fields.reduce((deepest, field) => Math.max(deepest, 1 + (field.type === "object" ? schemaDepth(children(field)) : 0)), 0);
}

export function totalFields(fields: SchemaField[]): number {
  return fields.reduce((total, field) => total + 1 + totalFields(children(field)), 0);
}

export function tokens(usage: Partial<Usage> | null | undefined): number {
  return (usage?.prompt_tokens ?? 0) + (usage?.completion_tokens ?? 0);
}

export function formatTokens(usage: Partial<Usage> | null | undefined): string {
  const total = tokens(usage);
  if (!total) return "not reported";
  const calls = usage?.calls ?? 0;
  return `${total.toLocaleString()} tokens${calls ? ` · ${calls} call${calls === 1 ? "" : "s"}` : ""}`;
}

export function formatSeconds(seconds: number | null | undefined): string {
  if (typeof seconds !== "number" || !Number.isFinite(seconds)) return "—";
  if (seconds < 10) return `${seconds.toFixed(1)} s`;
  const total = Math.round(seconds); // round once, so 119.6 s reads "2 min 0 s", not "1 min 60 s"
  return total < 90 ? `${total} s` : `${Math.floor(total / 60)} min ${total % 60} s`;
}

export function formatWhen(iso: string, withSeconds = false): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso
    : date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: withSeconds ? "medium" : "short" });
}

/**
 * When each suggestion ran (by id), precise enough to tell apart suggestions from the same solution: to the
 * minute, to the second when two share a minute, then numbered in creation order when they share a second.
 */
export function whenLabels(suggestions: SchemaSuggestion[]): Record<string, string> {
  const clashes = (key: (item: SchemaSuggestion) => string) => {
    const counts = new Map<string, number>();
    for (const item of suggestions) counts.set(key(item), (counts.get(key(item)) ?? 0) + 1);
    return (item: SchemaSuggestion) => (counts.get(key(item)) ?? 0) > 1;
  };
  const sameMinute = clashes(item => `${item.solution_name}\u0000${formatWhen(item.created_at)}`);
  const precise = (item: SchemaSuggestion) => formatWhen(item.created_at, sameMinute(item));
  const sameSecond = clashes(item => `${item.solution_name}\u0000${precise(item)}`);
  const ordinals = new Map<string, number>();
  const labels: Record<string, string> = {};
  for (const item of [...suggestions].sort((a, b) => a.created_at.localeCompare(b.created_at) || a.id.localeCompare(b.id))) {
    const text = precise(item);
    if (!sameSecond(item)) { labels[item.id] = text; continue; }
    const key = `${item.solution_name}\u0000${text}`;
    ordinals.set(key, (ordinals.get(key) ?? 0) + 1);
    labels[item.id] = `${text} #${ordinals.get(key)}`;
  }
  return labels;
}

export function sameDraft(a: SchemaDraft | null | undefined, b: SchemaDraft | null | undefined): boolean {
  if (!a || !b) return !a && !b;
  return JSON.stringify({ name: a.name, description: a.description, fields: a.fields })
    === JSON.stringify({ name: b.name, description: b.description, fields: b.fields });
}

/** Newest first; suggestions whose exemplars all belong to the given document set. */
export function suggestionsFor(suggestions: SchemaSuggestion[], documentIds: Iterable<string>): SchemaSuggestion[] {
  const ids = new Set(documentIds);
  return suggestions
    .filter(item => Array.isArray(item.document_ids) && item.document_ids.every(id => ids.has(id)))
    .sort((a, b) => b.created_at.localeCompare(a.created_at));
}

/** The persisted record the backend keeps for a shared suggestion, rebuilt from the POST response. */
export function sharedSuggestion(result: SharedSuggestionResult & { id: string }, documentIds: string[], guidance: string,
  reasoningEffort: string): SchemaSuggestion {
  return {
    id: result.id, solution: SHARED_ID, solution_name: SHARED_NAME, method: SHARED_METHOD, draft: result.draft,
    notes: result.notes ?? [], usage: { calls: result.usage?.calls ?? 0, prompt_tokens: result.usage?.prompt_tokens ?? 0,
      completion_tokens: result.usage?.completion_tokens ?? 0 },
    elapsed_seconds: result.elapsed_seconds ?? 0, exemplars: result.exemplars ?? [], document_ids: documentIds, guidance,
    settings: { reasoning_effort: reasoningEffort }, created_at: new Date().toISOString(), raw: null,
  };
}

// --- Suggestion settings (per solution, per browser) -----------------------------------------

const settingsKey = (solutionId: string) => `doc-cognition.suggest-settings.${solutionId}`;

export function suggestionSpecs(solution: SolutionInfo): SettingSpec[] {
  return solution.schema_suggestion?.settings ?? [];
}

export function loadSuggestionSettings(solution: SolutionInfo): Record<string, unknown> {
  const specs = suggestionSpecs(solution);
  const base = defaults(specs);
  try {
    const stored = JSON.parse(localStorage.getItem(settingsKey(solution.id)) ?? "{}");
    if (!stored || typeof stored !== "object") return base;
    return Object.fromEntries(specs.map(spec => [spec.key, storedValueFits(spec, stored[spec.key]) ? stored[spec.key] : base[spec.key]]));
  } catch { return base; }
}

export function storeSuggestionSettings(solutionId: string, values: Record<string, unknown>) {
  try { localStorage.setItem(settingsKey(solutionId), JSON.stringify(values)); } catch { /* storage unavailable */ }
}

/** Write a draft where WorkspacePage picks it up on its next mount. Returns false when storage is unavailable. */
export function storeWorkspaceDraft(draft: SchemaDraft): boolean {
  try { localStorage.setItem(SCHEMA_DRAFT_KEY, JSON.stringify(draft)); return true; } catch { return false; }
}

export function loadWorkspaceDraft(): SchemaDraft | null {
  try {
    const value = JSON.parse(localStorage.getItem(SCHEMA_DRAFT_KEY) ?? "null");
    return value && typeof value === "object" && Array.isArray(value.fields) ? value as SchemaDraft : null;
  } catch { return null; }
}

// --- Comparison -------------------------------------------------------------------------------

export type RowStatus = "same" | "partial" | "unique" | "differs";

export interface ComparisonRow {
  /** Name chain, unique per row (e.g. "attendees.name"). */
  key: string;
  /** Display path with list markers (e.g. "attendees[].name"). */
  path: string;
  /** The display prefix before the last segment (e.g. "attendees[]."). */
  parent: string;
  name: string;
  depth: number;
  /** One entry per compared suggestion, null when the suggestion has no field at this path. */
  cells: (SchemaField | null)[];
  present: number;
  status: RowStatus;
}

interface Node { name: string; cells: (SchemaField | null)[]; children: Map<string, Node> }

/**
 * Union of field paths across drafts: per level, fields shared by more drafts first, then first-seen order;
 * children follow their parent.
 * A row is "unique" when only one draft has it, "differs" when drafts that have it disagree on type or list-ness,
 * "partial" when some (but not all, and more than one) drafts have it, otherwise "same".
 */
export function compareDrafts(drafts: SchemaDraft[]): ComparisonRow[] {
  const root = new Map<string, Node>();
  drafts.forEach((draft, column) => {
    const walk = (fields: SchemaField[], level: Map<string, Node>) => {
      for (const field of fields) {
        let node = level.get(field.name);
        if (!node) {
          node = { name: field.name, cells: drafts.map(() => null), children: new Map() };
          level.set(field.name, node);
        }
        if (!node.cells[column]) node.cells[column] = field;
        walk(field.type === "object" ? children(field) : [], node.children);
      }
    };
    walk(Array.isArray(draft.fields) ? draft.fields : [], root);
  });
  const rows: ComparisonRow[] = [];
  const presence = (node: Node) => node.cells.filter(Boolean).length;
  const flatten = (level: Map<string, Node>, keyPrefix: string, pathPrefix: string, depth: number) => {
    // Fields more suggestions share come first (stable, so ties keep first-seen order); children stay under their parent.
    for (const node of [...level.values()].sort((a, b) => presence(b) - presence(a))) {
      const present = node.cells.filter(Boolean) as SchemaField[];
      const kinds = new Set(present.map(typeKey));
      const status: RowStatus = kinds.size > 1 ? "differs"
        : drafts.length > 1 && present.length === 1 ? "unique"
          : present.length < drafts.length ? "partial" : "same";
      const key = keyPrefix + node.name;
      const path = pathPrefix + node.name;
      rows.push({ key, path, parent: pathPrefix, name: node.name, depth, cells: node.cells, present: present.length, status });
      // Children only exist under object cells, so their list marker comes from those; "[?]" when they disagree.
      const lists = new Set(present.filter(field => field.type === "object").map(field => field.array));
      flatten(node.children, `${key}.`, `${path}${lists.size > 1 ? "[?]" : lists.has(true) ? "[]" : ""}.`, depth + 1);
    }
  };
  flatten(root, "", "", 1);
  return rows;
}
