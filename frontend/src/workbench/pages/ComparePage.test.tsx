import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import ComparePage from "./ComparePage";
import { parseHash } from "../router";
import type { Comparison, CompareRunStats, Run, RunSummary, SchemaField } from "../types";

const usage = { calls: 0, prompt_tokens: 0, completion_tokens: 0 };
const summary = (overrides: Partial<RunSummary>): RunSummary => ({
  id: "run", solution: "x", solution_name: "X", label: "", model: "m", status: "done", error: null,
  schema_id: "minutes", schema_version: 2, schema_name: "Meeting minutes", created_at: "2026-10-20T10:00:00Z",
  started_at: null, finished_at: null, documents_total: 2, documents_done: 2, documents_failed: 0,
  document_ids: ["doc-1", "doc-2"], elapsed_seconds: 10, values: 0, grounded: 0, mean_confidence: null, usage,
  confidence_method: "not reported", ...overrides,
});

const runA = summary({ id: "run-a", solution: "langextract", solution_name: "LangExtract", label: "baseline", model: "gpt-4.1",
  values: 60, grounded: 57, mean_confidence: 0.9, confidence_method: "token logprobs",
  usage: { calls: 4, prompt_tokens: 1200, completion_tokens: 300 } });
const runB = summary({ id: "run-b", solution: "nuextract", solution_name: "NuExtract3", label: "local", model: "nuextract-3",
  documents_done: 1, documents_failed: 1, elapsed_seconds: 30, values: 20, grounded: 10 });
const runC = summary({ id: "run-c", solution: "llm", solution_name: "Direct LLM", label: "draft", status: "running",
  schema_version: 1, documents_done: 0 });

const attendee = (name: string, role: string, attendance: string) => ({ name, role, attendance });
const fields: SchemaField[] = [
  { name: "chair", type: "verbatim", description: "", array: false, options: [], fields: [] },
  { name: "meeting_date", type: "date", description: "", array: false, options: [], fields: [] },
  { name: "attendees", type: "object", description: "", array: true, options: [], fields: [
    { name: "name", type: "verbatim", description: "", array: false, options: [], fields: [] },
    { name: "role", type: "verbatim", description: "", array: false, options: [], fields: [] },
    { name: "attendance", type: "enum", description: "", array: false, options: ["present", "apologies"], fields: [] },
  ] },
];
const value = (raw: unknown, extra: Record<string, unknown> = {}) => ({
  value: raw, display: Array.isArray(raw) ? `${raw.length} items` : String(raw), count: Array.isArray(raw) ? raw.length : null,
  confidence: null, grounded: 1, leaves: 1, grounding: null, start: null, end: null, ...extra,
});

function comparison(runIds: string[], reference: string | null): Comparison {
  const scored = (score: number) => reference ? { "run-a": null, "run-b": score } : { "run-a": score, "run-b": score };
  const fieldScores = (agreement: number) => ({
    chair: { filled: 2, documents: 2, agreement: reference === "run-a" ? null : 0.7, agree_count: 0, scored: 1 },
    meeting_date: { filled: 2, documents: 2, agreement: reference === "run-a" ? null : 1, agree_count: 1, scored: 1 },
    attendees: { filled: 2, documents: 2, agreement: reference === "run-a" ? null : agreement, agree_count: 0, scored: 1 },
  });
  const stats: Record<string, CompareRunStats> = {
    "run-a": { ...runA, agreement: reference === "run-a" ? null : 0.79, agree_rate: reference === "run-a" ? null : 0.5,
      fill_rate: 1, fields: fieldScores(0.667) },
    "run-b": { ...runB, agreement: 0.79, agree_rate: 0.5, fill_rate: 0.9, fields: fieldScores(0.667) },
  };
  return {
    reference,
    runs: runIds.filter(id => stats[id]).map(id => stats[id]),
    documents: [
      { id: "doc-1", name: "Minutes A", status: { "run-a": "done", "run-b": "done" } },
      { id: "doc-2", name: "Minutes B", status: { "run-a": "done", "run-b": "error" } },
    ],
    fields,
    cells: [
      { document_id: "doc-1", field: "chair", agreement: scored(0.7), values: {
        "run-a": value("Jane Smith", { confidence: 0.92, grounding: "exact", start: 7, end: 17 }),
        "run-b": value("J. Smith", { grounding: "fuzzy" }) } },
      { document_id: "doc-1", field: "meeting_date", agreement: scored(1), values: {
        "run-a": value("2026-03-01"), "run-b": value("2026-03-01") } },
      { document_id: "doc-1", field: "attendees", agreement: scored(0.667), values: {
        "run-a": value([attendee("Jane Smith", "Chair", "present"), attendee("Tom Lee", "Treasurer", "apologies")], { leaves: 6, grounded: 6 }),
        "run-b": value([attendee("Jane Smith", "Chair", "present")], { leaves: 3, grounded: 2 }) },
      rows: [
        { cells: { "run-a": attendee("Jane Smith", "Chair", "present"), "run-b": attendee("Jane Smith", "Chair", "present") }, scores: { "run-b": 0.92 } },
        { cells: { "run-a": attendee("Tom Lee", "Treasurer", "apologies") } },
      ] },
      { document_id: "doc-2", field: "chair", agreement: { "run-a": null }, values: { "run-a": value("Ann Bell") } },
      { document_id: "doc-2", field: "meeting_date", agreement: { "run-a": null }, values: { "run-a": value("") } },
      { document_id: "doc-2", field: "attendees", agreement: { "run-a": null }, values: { "run-a": value([]) } },
    ],
    warnings: [],
    thresholds: { match: 0.8, pair: 0.5 },
  };
}

const text = "Chair: Jane Smith\nPresent: Jane Smith, Tom Lee";
const fullRun: Run = {
  id: "run-a", solution: "langextract", solution_name: "LangExtract", label: "baseline", model: "gpt-4.1",
  confidence_method: "token logprobs", status: "done", error: null, settings: {}, created_at: runA.created_at, active: false,
  extraction_schema: { id: "minutes", version: 2, name: "Meeting minutes", description: "", fields, created_at: "", exemplars: [] },
  summary: runA,
  documents: [{
    document_id: "doc-1", document_name: "Minutes A", document_sha256: "x", status: "done", error: null, data: null,
    elapsed_seconds: 5, usage, notes: [], raw: null, started_at: null, finished_at: null,
    values: [
      { path: ["chair"], field: "chair", value: "Jane Smith", quote: "Jane Smith", confidence: 0.92, grounding: "exact", start: 7, end: 17 },
      { path: ["attendees", 0, "name"], field: "attendees.name", value: "Jane Smith", quote: null, confidence: 0.9, grounding: "exact", start: 27, end: 37 },
      { path: ["attendees", 1, "name"], field: "attendees.name", value: "Tom Lee", quote: null, confidence: 0.8, grounding: "exact", start: 39, end: 46 },
      { path: ["meeting_date"], field: "meeting_date", value: "2026-03-01", quote: null, confidence: 0.7, grounding: "none", start: null, end: null },
    ],
  }],
};

let fetchMock: ReturnType<typeof vi.fn>;
const compareBodies = () => fetchMock.mock.calls.filter(([input]) => input === "/api/compare")
  .map(([, init]) => JSON.parse((init as RequestInit).body as string));

beforeEach(() => {
  window.location.hash = "";
  fetchMock = vi.fn(async (input: string, init?: RequestInit) => {
    if (input === "/api/runs") return Response.json([runA, runB, runC]);
    if (input === "/api/compare") {
      const body = JSON.parse(init?.body as string);
      return Response.json(comparison(body.run_ids, body.reference_run_id));
    }
    if (input === "/api/documents/doc-1") return Response.json({ id: "doc-1", name: "Minutes A", text, characters: text.length,
      pages: null, page_starts: [0], sha256: "x" });
    if (input === "/api/runs/run-a") return Response.json(fullRun);
    return Response.json({ detail: "Unexpected endpoint" }, { status: 500 });
  });
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => { vi.unstubAllGlobals(); });

const query = () => parseHash(window.location.hash).query;

it("shows guidance when nothing is selected and no runs exist", async () => {
  fetchMock.mockImplementation(async () => Response.json([]));
  window.location.hash = "#/compare";
  render(<ComparePage />);
  expect(await screen.findByText("Run at least one solution, then select runs to compare.")).toBeInTheDocument();
});

it("renders the scoreboard and field matrix from a comparison", async () => {
  window.location.hash = "#/compare?runs=run-a,run-b";
  render(<ComparePage />);
  const board = await screen.findByRole("table", { name: "Scoreboard" });
  expect(within(board).getAllByRole("columnheader").map(cell => cell.textContent)).toEqual(
    expect.arrayContaining([expect.stringContaining("LangExtract"), expect.stringContaining("NuExtract3")]));
  const time = within(board).getByRole("row", { name: /Time \/ document/ });
  expect(within(time).getByText("5.0 s")).toBeInTheDocument();
  expect(within(time).getByText("15 s")).toBeInTheDocument();
  expect(within(time).getAllByText("best")).toHaveLength(1);
  expect(within(board).getByRole("row", { name: /Grounded/ })).toHaveTextContent("95%");
  expect(within(board).getByRole("row", { name: /Tokens/ })).toHaveTextContent("not metered");
  expect(within(board).getByRole("row", { name: /Mean confidence/ })).toHaveTextContent("n/a — not reported");

  const matrix = screen.getByRole("table", { name: "Field matrix" });
  expect(within(matrix).getByLabelText("LangExtract — baseline on chair: 70% agreement, partial; filled 2 of 2 documents"))
    .toHaveClass("cmp-s2");
  expect(within(matrix).getByLabelText(/NuExtract3 — local on meeting_date: 100% agreement, strong agreement/)).toHaveClass("cmp-s4");
  expect(compareBodies()).toEqual([{ run_ids: ["run-a", "run-b"], reference_run_id: null }]);

  const documents = screen.getByRole("table", { name: "Document matrix" });
  expect(within(documents).getByLabelText("NuExtract3 — local: error")).toBeInTheDocument();
});

it("selecting runs updates the query and requests the comparison", async () => {
  window.location.hash = "#/compare";
  render(<ComparePage />);
  expect(await screen.findByText(/Select two or more runs/)).toBeInTheDocument();
  fireEvent.click(await screen.findByRole("checkbox", { name: /LangExtract/ }));
  await waitFor(() => expect(query().runs).toBe("run-a"));
  await waitFor(() => expect(compareBodies()).toEqual([{ run_ids: ["run-a"], reference_run_id: null }]));

  fireEvent.click(screen.getByRole("checkbox", { name: /NuExtract3/ }));
  await waitFor(() => expect(query().runs).toBe("run-a,run-b"));
  await waitFor(() => expect(compareBodies().at(-1)).toEqual({ run_ids: ["run-a", "run-b"], reference_run_id: null }));
  expect(await screen.findByRole("table", { name: "Scoreboard" })).toBeInTheDocument();

  fireEvent.click(screen.getByRole("checkbox", { name: /Direct LLM/ }));
  await waitFor(() => expect(query().runs).toBe("run-a,run-b,run-c"));
  expect(screen.getByText(/Selected runs use different schemas/)).toBeInTheDocument();
  expect(screen.getByText(/Direct LLM \(running\) not finished/)).toBeInTheDocument();
});

it("choosing a reference re-scores against that run", async () => {
  window.location.hash = "#/compare?runs=run-a,run-b";
  render(<ComparePage />);
  await screen.findByRole("table", { name: "Scoreboard" });
  fireEvent.change(screen.getByLabelText(/Agreement measured against/i), { target: { value: "run-a" } });
  await waitFor(() => expect(query().ref).toBe("run-a"));
  await waitFor(() => expect(compareBodies().at(-1)).toEqual({ run_ids: ["run-a", "run-b"], reference_run_id: "run-a" }));
  const board = screen.getByRole("table", { name: "Scoreboard" });
  await waitFor(() => expect(within(board).getByRole("row", { name: /^Agreement/ })).toHaveTextContent("reference"));
  expect(screen.getByText(/as if it were the answer key/)).toBeInTheDocument();

  fireEvent.change(screen.getByLabelText(/Agreement measured against/i), { target: { value: "" } });
  await waitFor(() => expect(query().ref).toBeUndefined());
});

it("opens a document drill-down and expands aligned list items", async () => {
  window.location.hash = "#/compare?runs=run-a,run-b";
  render(<ComparePage />);
  fireEvent.click(await screen.findByRole("button", { name: "Minutes A" }));
  await waitFor(() => expect(query().doc).toBe("doc-1"));
  const drill = await screen.findByRole("table", { name: "Values in Minutes A" });
  const chair = within(drill).getByRole("row", { name: /^chair/ });
  expect(within(chair).getByRole("button", { name: "Jane Smith" })).toBeInTheDocument();
  expect(within(chair).getByText("J. Smith")).toBeInTheDocument();
  expect(within(chair).getByText("conf 92%")).toBeInTheDocument();
  expect(within(chair).getByText("fuzzy")).toBeInTheDocument();

  const expand = within(drill).getByRole("button", { name: /attendees/ });
  expect(expand).toHaveAttribute("aria-expanded", "false");
  expect(within(drill).getByRole("button", { name: "2 items" })).toBeInTheDocument();
  fireEvent.click(expand);
  expect(expand).toHaveAttribute("aria-expanded", "true");
  const first = within(drill).getByRole("row", { name: /attendees #1/ });
  expect(within(first).getAllByRole("button", { name: "Jane Smith · Chair · present" })).toHaveLength(2);
  expect(within(first).getByText("pivot")).toBeInTheDocument();
  expect(within(first).getByText("92%")).toBeInTheDocument();
  const second = within(drill).getByRole("row", { name: /attendees #2/ });
  expect(within(second).getByRole("button", { name: "Tom Lee · Treasurer · apologies" })).toBeInTheDocument();
  expect(within(second).getByLabelText("NuExtract3 — local: no matching item")).toBeInTheDocument();

  fireEvent.click(screen.getByRole("button", { name: "Close drill-down" }));
  await waitFor(() => expect(query().doc).toBeUndefined());
});

it("shows a clicked value in the source text", async () => {
  window.location.hash = "#/compare?runs=run-a,run-b&doc=doc-1";
  render(<ComparePage />);
  const drill = await screen.findByRole("table", { name: "Values in Minutes A" });
  fireEvent.click(within(within(drill).getByRole("row", { name: /^chair/ })).getByRole("button", { name: "Jane Smith" }));
  const panel = await screen.findByRole("complementary", { name: "Source panel" });
  const source = await within(panel).findByLabelText("Document source text");
  expect(within(source).getByRole("button", { name: "Jane Smith" })).toHaveAttribute("aria-pressed", "true");
  expect(within(source).queryByRole("button", { name: "Tom Lee" })).not.toBeInTheDocument(); // filtered to the field
  expect(fetchMock).toHaveBeenCalledWith("/api/documents/doc-1", undefined);
  expect(fetchMock).toHaveBeenCalledWith("/api/runs/run-a", undefined);

  // An aligned list item selects that item's first located leaf; document and run come from the cache.
  fireEvent.click(within(drill).getByRole("button", { name: /attendees/ }));
  const second = within(drill).getByRole("row", { name: /attendees #2/ });
  fireEvent.click(within(second).getByRole("button", { name: "Tom Lee · Treasurer · apologies" }));
  await waitFor(() => expect(within(screen.getByLabelText("Document source text")).getByRole("button", { name: "Tom Lee" }))
    .toHaveAttribute("aria-pressed", "true"));
  expect(within(screen.getByLabelText("Document source text")).getByRole("button", { name: "Jane Smith" }))
    .toHaveAttribute("aria-pressed", "false");
  expect(fetchMock.mock.calls.filter(([input]) => input === "/api/runs/run-a")).toHaveLength(1);
});

it("exports the field matrix as CSV", async () => {
  const created: Blob[] = [];
  const original = { create: URL.createObjectURL, revoke: URL.revokeObjectURL };
  URL.createObjectURL = vi.fn((blob: Blob) => { created.push(blob); return "blob:test"; });
  URL.revokeObjectURL = vi.fn();
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
  try {
    window.location.hash = "#/compare?runs=run-a,run-b";
    render(<ComparePage />);
    fireEvent.click(await screen.findByRole("button", { name: /Field matrix CSV/ }));
    expect(created).toHaveLength(1);
    const csv = await new Promise<string>(resolve => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.readAsText(created[0]);
    });
    expect(csv.split("\r\n")[0]).toContain("LangExtract — baseline agreement");
    expect(csv).toContain("chair,verbatim,0.7,0,1,2,2");
    await waitFor(() => expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:test"));
  } finally {
    URL.createObjectURL = original.create;
    URL.revokeObjectURL = original.revoke;
  }
});
