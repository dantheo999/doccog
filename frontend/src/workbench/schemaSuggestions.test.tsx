import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { SuggestionComparison } from "./components/SchemaSuggestions";
import { compareDrafts, formatSeconds, formatWhen, schemaDepth, whenLabels } from "./components/schemaSuggestionModel";
import type { SchemaDraft, SchemaField, SchemaSuggestion, SettingSpec, SolutionInfo } from "./types";

const leaf = (name: string, type: SchemaField["type"], array = false): SchemaField =>
  ({ name, type, description: "", array, options: [], fields: [] });
const object = (name: string, fields: SchemaField[], array = true): SchemaField =>
  ({ name, type: "object", description: "", array, options: [], fields });

const sharedDraft: SchemaDraft = { name: "Meeting minutes", description: "Board minutes", fields: [
  leaf("meeting_date", "date"), object("attendees", [leaf("name", "verbatim"), leaf("role", "text")]),
] };
const llmDraft: SchemaDraft = { name: "Minutes (LLM)", description: "Board minutes", fields: [
  leaf("meeting_date", "text"), object("attendees", [leaf("name", "verbatim")]), leaf("decisions", "verbatim", true),
] };
const savedSchema = { ...sharedDraft, id: "s1", version: 1, created_at: "2026-10-25T00:00:00Z", exemplars: [] };
const collection = { id: "c1", name: "minutes", created_at: "2026-10-25", updated_at: "2026-10-25", documents: [
  { id: "d1", path: "minutes/a.txt", name: "a.txt", characters: 33, pages: null, sha256: "x" },
  { id: "d2", path: "minutes/b.txt", name: "b.txt", characters: 44, pages: null, sha256: "y" },
] };
const usage = { calls: 1, prompt_tokens: 1200, completion_tokens: 300 };
const UNSTRACT_REASON = "Unstract's open-source edition has no prompt or schema generation.";
const NUEXTRACT_SPECS: SettingSpec[] = [
  { key: "enable_thinking", label: "Reasoning mode", type: "boolean", default: false, help: "", min: null, max: null, options: null },
  { key: "max_tokens", label: "Maximum output tokens", type: "integer", default: 4096, help: "", min: 256, max: 32768, options: null },
];

const solution = (overrides: Partial<SolutionInfo>): SolutionInfo => ({
  id: "llm", name: "Direct LLM", tagline: "Prompted", description: "Calls the model.", technology: "Azure OpenAI",
  runtime: "cloud", capabilities: [], confidence_method: "Model-reported.", route: "/apps/llm", accent: "#7cc4ff",
  external_url: null, status: { ready: true, detail: "", model: "gpt" },
  settings: [{ key: "reasoning_effort", label: "Reasoning effort", type: "select", default: "low", help: "", min: null, max: null, options: ["low", "high"] }],
  schema_suggestion: { supported: true, method: "Same approach as the shared suggestion, with this solution's own settings.",
    settings: [{ key: "reasoning_effort", label: "Reasoning effort", type: "select", default: "medium", help: "", min: null, max: null,
      options: ["low", "medium", "high"] }] },
  ...overrides,
});

const suggestion = (overrides: Partial<SchemaSuggestion>): SchemaSuggestion => ({
  id: "x", solution: "llm", solution_name: "Direct LLM", method: "Same approach as the shared suggestion.", draft: llmDraft,
  notes: [], usage, elapsed_seconds: 21.5, exemplars: ["a.txt"], document_ids: ["d1"], guidance: "", settings: {},
  created_at: "2026-10-25T10:00:00Z", raw: null, ...overrides,
});

let requests: { url: string; method: string; body: unknown }[] = [];
let stored: SchemaSuggestion[] = [];
let solutions: SolutionInfo[] = [];
let gate: Promise<void> | null = null;
let failing: Record<string, { status: number; detail: string }> = {};
let confirm: ReturnType<typeof vi.fn>;

beforeEach(() => {
  localStorage.clear();
  window.location.hash = "";
  requests = [];
  stored = [];
  gate = null;
  failing = {};
  solutions = [
    solution({}),
    solution({ id: "nuextract", name: "NuExtract3", route: "/apps/nuextract", runtime: "local", accent: "#72d6bb",
      status: { ready: false, detail: "Start the NuExtract model server.", model: "" },
      schema_suggestion: { supported: true, method: "Native template generation.", settings: [] } }),
    solution({ id: "unstract", name: "Unstract", route: "/apps/unstract", runtime: "local",
      schema_suggestion: { supported: false, method: UNSTRACT_REASON, settings: [] } }),
    solution({ id: "langextract", name: "LangExtract", route: "/apps/langextract",
      schema_suggestion: { supported: true, method: "Derived from a generated recipe's classes.", settings: [] } }),
  ];
  confirm = vi.fn(() => true);
  vi.stubGlobal("confirm", confirm);
  vi.stubGlobal("fetch", vi.fn(async (input: string, init?: RequestInit) => {
    const method = init?.method ?? "GET";
    const body = typeof init?.body === "string" ? JSON.parse(init.body) : init?.body;
    requests.push({ url: input, method, body });
    if (input === "/api/solutions") return Response.json(solutions);
    if (input === "/api/runs") return Response.json([]);
    if (input === "/api/collections") return Response.json([]);
    if (input === "/api/collections/c1") return Response.json(collection);
    if (input === "/api/schemas/s1") return Response.json(savedSchema);
    if (input === "/api/schema-suggestions") return Response.json([...stored].reverse());
    if (input.startsWith("/api/schema-suggestions/") && method === "DELETE") {
      const id = input.split("/").pop()!;
      stored = stored.filter(item => item.id !== id);
      return Response.json({ deleted: id });
    }
    if (input === "/api/schemas/suggest") {
      if (gate) await gate;
      const exemplars = collection.documents.filter(item => body.document_ids.includes(item.id)).map(item => item.name);
      stored.push(suggestion({ id: "sh1", solution: "shared", solution_name: "Shared suggestion", method: "Solution-neutral.",
        draft: sharedDraft, notes: ["Attendance varies."], document_ids: body.document_ids, exemplars, created_at: "2026-10-25T11:00:00Z" }));
      return Response.json({ draft: sharedDraft, notes: ["Attendance varies."], usage, exemplars, id: "sh1", elapsed_seconds: 12 });
    }
    const match = /^\/api\/solutions\/(\w+)\/suggest-schema$/.exec(input);
    if (match && method === "POST") {
      if (gate) await gate;
      if (failing[match[1]]) return Response.json({ detail: failing[match[1]].detail }, { status: failing[match[1]].status });
      const exemplars = collection.documents.filter(item => body.document_ids.includes(item.id)).map(item => item.name);
      const created = suggestion({ id: `${match[1]}1`, document_ids: body.document_ids, exemplars, guidance: body.guidance,
        settings: body.settings, notes: ["Decisions are listed per item."], created_at: "2026-10-25T11:00:01Z" });
      stored.push(created);
      return Response.json(created);
    }
    return Response.json({ detail: `Unexpected ${method} ${input}` }, { status: 500 });
  }));
});

function openWorkspace(schemaId: string | null = null) {
  localStorage.setItem("doc-cognition.workspace.v1", JSON.stringify({ collectionId: "c1", schemaId }));
  window.location.hash = "#/workspace";
  render(<App />);
}

describe("workspace schema suggestions", () => {
  it("keeps the shared suggestion as the default and still posts it to /api/schemas/suggest", async () => {
    openWorkspace();
    const shared = await screen.findByRole("checkbox", { name: "Shared suggestion" });
    expect(shared).toBeChecked();
    expect(shared).toHaveAccessibleDescription(/Solution-neutral/);
    fireEvent.click(await screen.findByLabelText("Use minutes/b.txt as exemplar"));
    fireEvent.change(screen.getByLabelText("Guidance for the suggestion (optional)"), { target: { value: "Governance" } });
    fireEvent.click(screen.getByRole("button", { name: "Suggest from 1 exemplar" }));
    await within(await screen.findByRole("region", { name: "Model notes" })).findByText("Attendance varies.");
    expect(requests.find(item => item.url === "/api/schemas/suggest")!.body)
      .toEqual({ document_ids: ["d2"], guidance: "Governance", reasoning_effort: "medium" });
    expect(requests.some(item => item.url.endsWith("/suggest-schema"))).toBe(false);
    expect(screen.getByLabelText("Field name attendees.role")).toHaveValue("role");
    // The shared result is also kept as a suggestion card for later comparison.
    expect(await screen.findByRole("article", { name: /^Shared suggestion/ })).toBeInTheDocument();
  });

  it("lists the solutions that can suggest after the shared option and disables the rest with their reason", async () => {
    openWorkspace();
    const llm = await screen.findByRole("checkbox", { name: "Direct LLM" });
    const group = screen.getByRole("group", { name: "Suggest with" });
    const names = within(group).getAllByRole("checkbox").map(box => box.getAttribute("id"));
    expect(names).toEqual(["suggest-with-shared", "suggest-with-llm", "suggest-with-nuextract", "suggest-with-langextract", "suggest-with-unstract"]);
    expect(llm).toBeEnabled();
    expect(llm).not.toBeChecked();
    expect(llm).toHaveAccessibleDescription(/Same approach as the shared suggestion/);
    const unstract = screen.getByRole("checkbox", { name: "Unstract" });
    expect(unstract).toBeDisabled();
    expect(unstract).toHaveAccessibleDescription(UNSTRACT_REASON);
    const nuextract = screen.getByRole("checkbox", { name: "NuExtract3" });
    expect(nuextract).toBeDisabled();
    expect(nuextract).toHaveAccessibleDescription("Not ready: Start the NuExtract model server.");
    expect(screen.getByRole("checkbox", { name: "LangExtract" })).toBeEnabled();
  });

  it("runs the shared and a solution suggestion in parallel and shows both as cards side by side", async () => {
    let open!: () => void;
    gate = new Promise(resolve => { open = resolve; });
    openWorkspace();
    fireEvent.click(await screen.findByLabelText("Use minutes/a.txt as exemplar"));
    fireEvent.click(screen.getByLabelText("Use minutes/b.txt as exemplar"));
    fireEvent.click(await screen.findByRole("checkbox", { name: "Direct LLM" }));
    fireEvent.click(screen.getByRole("button", { name: "Suggest from 2 exemplars · 2 methods" }));
    // Both requests are in flight at once, each with its own loading state.
    await waitFor(() => expect(requests.filter(item => item.method === "POST")).toHaveLength(2));
    expect(screen.getAllByText(/Suggesting…/)).toHaveLength(2);
    expect(requests.find(item => item.url === "/api/schemas/suggest")!.body)
      .toEqual({ document_ids: ["d1", "d2"], guidance: "", reasoning_effort: "medium" });
    expect(requests.find(item => item.url === "/api/solutions/llm/suggest-schema")!.body)
      .toEqual({ document_ids: ["d1", "d2"], guidance: "", settings: { reasoning_effort: "medium" } });
    open();
    await screen.findByText("Loaded into the editor", { exact: false });
    await screen.findByText(/3 fields · in Suggestions/);
    const cards = await screen.findAllByRole("article");
    expect(cards.map(card => card.getAttribute("aria-label"))).toEqual([
      expect.stringMatching(/^Direct LLM/), expect.stringMatching(/^Shared suggestion/)]);
    expect(within(cards[0]).getByText("a.txt, b.txt")).toBeInTheDocument();
    expect(within(cards[0]).getByText("1,500 tokens · 1 call")).toBeInTheDocument();
    expect(within(cards[0]).getByText("3 fields · depth 2")).toBeInTheDocument();
    // A multi-method batch is ticked for comparison automatically.
    const comparison = screen.getByRole("region", { name: "Comparison" });
    expect(within(comparison).getAllByRole("columnheader").map(cell => cell.textContent)).toEqual([
      "Field path", "Shared suggestion2 top-level · depth 2", "Direct LLM3 top-level · depth 2", "Agreement"]);
  });

  it("shows a failing method's error in its own row without blocking the others", async () => {
    failing = { langextract: { status: 503, detail: "Azure OpenAI is not configured." } };
    openWorkspace();
    fireEvent.click(await screen.findByLabelText("Use minutes/a.txt as exemplar"));
    fireEvent.click(await screen.findByRole("checkbox", { name: "LangExtract" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Direct LLM" }));
    fireEvent.click(screen.getByRole("button", { name: "Suggest from 1 exemplar · 3 methods" }));
    const row = screen.getByRole("checkbox", { name: "LangExtract" }).closest(".suggest-option") as HTMLElement;
    expect(await within(row).findByRole("alert")).toHaveTextContent("Azure OpenAI is not configured.");
    await screen.findByText("Loaded into the editor", { exact: false });
    expect(screen.getByLabelText("Field name attendees.role")).toBeInTheDocument();
    expect(await screen.findAllByRole("article")).toHaveLength(2);
    // Only the failed method can be retried on its own: the button reflects the methods not running.
    fireEvent.click(screen.getByRole("checkbox", { name: "Shared suggestion" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Direct LLM" }));
    failing = {};
    fireEvent.click(screen.getByRole("button", { name: "Suggest from 1 exemplar" }));
    await waitFor(() => expect(within(row).queryByRole("alert")).not.toBeInTheDocument());
    expect(requests.filter(item => item.url === "/api/solutions/langextract/suggest-schema")).toHaveLength(2);
  });

  it("loads a saved suggestion into the editor, confirming over unsaved edits, and deletes suggestions", async () => {
    stored = [
      suggestion({ id: "sh0", solution: "shared", solution_name: "Shared suggestion", draft: sharedDraft, notes: ["Attendance varies."],
        created_at: "2026-10-25T09:00:00Z" }),
      suggestion({ id: "llm0", notes: ["Decisions are listed per item."] }),
      suggestion({ id: "other", document_ids: ["elsewhere"], solution_name: "Other set" }),
    ];
    openWorkspace("s1");
    const llmCard = await screen.findByRole("article", { name: /^Direct LLM/ });
    expect(screen.queryByRole("article", { name: /^Other set/ })).not.toBeInTheDocument();
    fireEvent.change(await screen.findByLabelText("Field name meeting_date"), { target: { value: "held_on" } });
    fireEvent.click(within(llmCard).getByRole("button", { name: "Use as draft" }));
    expect(confirm).toHaveBeenCalledWith(expect.stringContaining("Replace the current draft"));
    expect(await screen.findByLabelText("Field name decisions")).toHaveValue("decisions");
    const notes = screen.getByRole("region", { name: "Model notes" });
    expect(within(notes).getByText("Decisions are listed per item.")).toBeInTheDocument();
    expect(within(notes).getByText("· from Direct LLM's suggestion")).toBeInTheDocument();
    expect(within(llmCard).getByText("in editor")).toBeInTheDocument();
    // The editor sits below the cards, so "Use as draft" brings it into view and moves focus to it.
    await waitFor(() => expect(screen.getByRole("heading", { name: "Agreed schema" })).toHaveFocus());
    fireEvent.click(within(await screen.findByRole("article", { name: /^Shared suggestion/ })).getByRole("button", { name: "Use as draft" }));
    expect(within(screen.getByRole("region", { name: "Model notes" })).getByText("· from the shared suggestion")).toBeInTheDocument();
    fireEvent.click(within(llmCard).getByRole("button", { name: "Use as draft" }));
    fireEvent.click(within(llmCard).getByRole("button", { name: /^Delete Direct LLM suggestion/ }));
    await waitFor(() => expect(screen.queryByRole("article", { name: /^Direct LLM/ })).not.toBeInTheDocument());
    expect(requests.some(item => item.method === "DELETE" && item.url === "/api/schema-suggestions/llm0")).toBe(true);
  });

  it("keeps the user's draft when the shared result arrives after they loaded or edited another one", async () => {
    let openShared!: () => void;
    const sharedGate = new Promise<void>(resolve => { openShared = resolve; });
    const mocked = globalThis.fetch as unknown as (input: string, init?: RequestInit) => Promise<Response>;
    vi.stubGlobal("fetch", vi.fn(async (input: string, init?: RequestInit) => {
      if (input === "/api/schemas/suggest") await sharedGate;
      return mocked(input, init);
    }));
    openWorkspace();
    fireEvent.click(await screen.findByLabelText("Use minutes/a.txt as exemplar"));
    fireEvent.click(await screen.findByRole("checkbox", { name: "Direct LLM" }));
    fireEvent.click(screen.getByRole("button", { name: "Suggest from 1 exemplar · 2 methods" }));
    const llmCard = await screen.findByRole("article", { name: /^Direct LLM/ });
    fireEvent.click(within(llmCard).getByRole("button", { name: "Use as draft" }));
    fireEvent.change(await screen.findByLabelText("Field name decisions"), { target: { value: "resolutions" } });
    confirm.mockClear();
    openShared();
    await screen.findByText(/Kept under Suggestions; the editor draft changed while it ran/);
    expect(confirm).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Field name resolutions")).toHaveValue("resolutions");
    expect(screen.queryByLabelText("Field name attendees.role")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Show in editor" })).not.toBeInTheDocument();
    // The shared result is still a card, so it can be loaded later.
    expect(await screen.findByRole("article", { name: /^Shared suggestion/ })).toBeInTheDocument();
  });

  it("loads a shared-only result into the editor and brings the editor into view", async () => {
    const scroll = vi.fn();
    Element.prototype.scrollIntoView = scroll;
    try {
      openWorkspace();
      fireEvent.click(await screen.findByLabelText("Use minutes/a.txt as exemplar"));
      fireEvent.click(screen.getByRole("button", { name: "Suggest from 1 exemplar" }));
      expect(await screen.findByLabelText("Field name attendees.role")).toHaveValue("role");
      await waitFor(() => expect(scroll).toHaveBeenCalledTimes(1));
      expect(scroll.mock.contexts[0]).toBe(screen.getByRole("heading", { name: "Agreed schema" }).closest("section"));
      fireEvent.click(screen.getByRole("button", { name: "Show in editor" }));
      await waitFor(() => expect(scroll).toHaveBeenCalledTimes(2));
      expect(screen.getByRole("heading", { name: "Agreed schema" })).toHaveFocus();
    } finally { delete (Element.prototype as { scrollIntoView?: unknown }).scrollIntoView; }
  });

  it("validates a solution's suggestion settings before suggesting, like the Solution page", async () => {
    solutions[1] = { ...solutions[1], status: { ready: true, detail: "", model: "nu" },
      schema_suggestion: { supported: true, method: "Native template generation.", settings: NUEXTRACT_SPECS } };
    openWorkspace();
    fireEvent.click(await screen.findByLabelText("Use minutes/a.txt as exemplar"));
    fireEvent.click(await screen.findByRole("checkbox", { name: "NuExtract3" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Shared suggestion" }));
    fireEvent.click(screen.getByText("NuExtract3 suggestion settings"));
    fireEvent.change(screen.getByLabelText("Maximum output tokens"), { target: { value: "100" } });
    const row = screen.getByRole("checkbox", { name: "NuExtract3" }).closest(".suggest-option") as HTMLElement;
    expect(within(row).getByRole("list", { name: "NuExtract3 suggestion settings errors" }))
      .toHaveTextContent("Maximum output tokens: must be between 256 and 32768.");
    const button = screen.getByRole("button", { name: "Suggest from 1 exemplar" });
    expect(button).toBeDisabled();
    expect(screen.getByText(/Fix the suggestion settings of NuExtract3/)).toBeInTheDocument();
    // Unticking the method lifts the block; ticking it again brings it back.
    fireEvent.click(screen.getByRole("checkbox", { name: "NuExtract3" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Shared suggestion" }));
    expect(button).toBeEnabled();
    fireEvent.click(screen.getByRole("checkbox", { name: "Shared suggestion" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "NuExtract3" }));
    expect(button).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Maximum output tokens"), { target: { value: "2048" } });
    expect(within(row).queryByRole("list", { name: /settings errors/ })).not.toBeInTheDocument();
    fireEvent.click(button);
    await waitFor(() => expect(requests.some(item => item.url === "/api/solutions/nuextract/suggest-schema")).toBe(true));
    expect(requests.find(item => item.url === "/api/solutions/nuextract/suggest-schema")!.body)
      .toEqual({ document_ids: ["d1"], guidance: "", settings: { enable_thinking: false, max_tokens: 2048 } });
  });

  it("tells apart suggestions from the same solution made in the same minute", async () => {
    stored = [
      suggestion({ id: "llmA", created_at: "2026-10-25T18:47:05Z" }),
      suggestion({ id: "llmB", created_at: "2026-10-25T18:47:40Z", draft: sharedDraft }),
    ];
    openWorkspace();
    const cards = await screen.findAllByRole("article", { name: /^Direct LLM/ });
    const labels = cards.map(card => card.getAttribute("aria-label"));
    expect(new Set(labels).size).toBe(2);
    const boxes = screen.getAllByRole("checkbox", { name: /^Compare Direct LLM suggestion from/ });
    expect(new Set(boxes.map(box => box.getAttribute("aria-label"))).size).toBe(2);
    boxes.forEach(box => fireEvent.click(box));
    const comparison = await screen.findByRole("region", { name: "Comparison" });
    // Columns follow the order ticked (newest card first) and use the cards' "when" labels.
    const headers = Array.from(comparison.querySelectorAll(".compare-col-name"), cell => cell.textContent);
    expect(headers).toEqual(labels.map(label => label!.replace(/^Direct LLM, (.*)$/, "Direct LLM ($1)")));
    expect(new Set(headers).size).toBe(2);
    const row = (path: string) => within(comparison).getByRole("rowheader", { name: path }).closest("tr")!;
    expect(within(row("decisions")).getByText(`Only in ${headers[1]}`)).toBeInTheDocument();
    expect(within(row("attendees[].role")).getByText(`Only in ${headers[0]}`)).toBeInTheDocument();
  });
});

describe("suggestion comparison", () => {
  const shared = suggestion({ id: "a", solution: "shared", solution_name: "Shared suggestion", draft: sharedDraft });
  const llm = suggestion({ id: "b", draft: llmDraft });

  it("builds the union of field paths with list markers, flagging unique and differing rows", () => {
    const rows = compareDrafts([sharedDraft, llmDraft]);
    expect(rows.map(row => [row.path, row.status])).toEqual([
      ["meeting_date", "differs"], ["attendees", "same"], ["attendees[].name", "same"], ["attendees[].role", "unique"],
      ["decisions", "unique"],
    ]);
    // Shared fields sort ahead of fields only one suggestion has, keeping children under their parent.
    const ordered = compareDrafts([
      { name: "a", description: "", fields: [leaf("only_here", "text"), object("items", [leaf("solo", "text"), leaf("title", "text")])] },
      { name: "b", description: "", fields: [object("items", [leaf("title", "text")])] },
    ]);
    expect(ordered.map(row => row.path)).toEqual(["items", "items[].title", "items[].solo", "only_here"]);
    expect(schemaDepth(sharedDraft.fields)).toBe(2);
    expect(schemaDepth([leaf("a", "text")])).toBe(1);
  });

  it("takes child list markers from the object cells and flags disagreement", () => {
    const owner = [leaf("owner", "text")];
    expect(compareDrafts([
      { name: "a", description: "", fields: [leaf("action", "verbatim", true)] },
      { name: "b", description: "", fields: [object("action", owner, false)] },
    ]).map(row => row.path)).toEqual(["action", "action.owner"]);
    expect(compareDrafts([
      { name: "a", description: "", fields: [object("action", owner, true)] },
      { name: "b", description: "", fields: [object("action", owner, false)] },
    ]).map(row => row.path)).toEqual(["action", "action[?].owner"]);
  });

  it("formats durations without a 60-second remainder", () => {
    expect([4.25, 59.6, 89.6, 104.8, 119.6, 179.5, 3600].map(formatSeconds))
      .toEqual(["4.3 s", "60 s", "1 min 30 s", "1 min 45 s", "2 min 0 s", "3 min 0 s", "60 min 0 s"]);
  });

  it("labels suggestion times to the minute, then the second, then with an ordinal", () => {
    const at = (id: string, created_at: string, solution_name = "Direct LLM") => suggestion({ id, created_at, solution_name });
    const labels = whenLabels([at("a", "2026-10-25T18:47:05Z"), at("b", "2026-10-25T18:47:40Z"), at("c", "2026-10-25T18:47:40Z"),
      at("d", "2026-10-25T18:47:40Z", "LangExtract"), at("e", "2026-10-25T19:10:00Z")]);
    expect(labels).toEqual({
      a: formatWhen("2026-10-25T18:47:05Z", true), // shares its minute with b and c: to the second
      b: `${formatWhen("2026-10-25T18:47:40Z", true)} #1`, // b and c share the second: numbered in creation order
      c: `${formatWhen("2026-10-25T18:47:40Z", true)} #2`,
      d: formatWhen("2026-10-25T18:47:40Z"), // another solution does not clash
      e: formatWhen("2026-10-25T19:10:00Z"),
    });
    expect(formatWhen("2026-10-25T18:47:05Z", true)).not.toBe(formatWhen("2026-10-25T18:47:40Z", true));
  });

  it("renders a compact table that marks unique and different rows", () => {
    render(<SuggestionComparison suggestions={[shared, llm]} />);
    const table = screen.getByRole("table");
    const row = (path: string) => within(table).getByRole("rowheader", { name: path }).closest("tr")!;
    expect(row("meeting_date")).toHaveAttribute("data-status", "differs");
    expect(within(row("meeting_date")).getByText("Types differ")).toBeInTheDocument();
    expect(within(row("meeting_date")).getByText("Date")).toBeInTheDocument();
    expect(within(row("meeting_date")).getByText("Text")).toBeInTheDocument();
    expect(row("attendees[].role")).toHaveAttribute("data-status", "unique");
    expect(within(row("attendees[].role")).getByText("Only in Shared suggestion")).toBeInTheDocument();
    expect(within(row("attendees[].role")).getByText("—")).toBeInTheDocument();
    expect(within(row("decisions")).getByText("Only in Direct LLM")).toBeInTheDocument();
    expect(within(row("decisions")).getByText("list")).toBeInTheDocument();
    expect(within(row("attendees[].name")).getByText("Same")).toBeInTheDocument();
    expect(screen.getByText("5 field paths · 2 only in one · 1 with different types")).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("Only differences"));
    expect(within(table).queryByRole("rowheader", { name: "attendees[].name" })).not.toBeInTheDocument();
    expect(within(table).getByRole("rowheader", { name: "decisions" })).toBeInTheDocument();
  });
});

describe("solution page schema suggestion", () => {
  function openSolution(route: string) {
    localStorage.setItem("doc-cognition.workspace.v1", JSON.stringify({ collectionId: "c1", schemaId: "s1" }));
    window.location.hash = `#${route}`;
    render(<App />);
  }

  it("tests one solution's suggestion on its own and opens the result in the Workspace", async () => {
    openSolution("/apps/llm");
    const panel = await screen.findByRole("region", { name: "Schema suggestion" });
    expect(within(panel).getByText(/Same approach as the shared suggestion/)).toBeInTheDocument();
    const suggest = within(panel).getByRole("button", { name: /Suggest schema/ });
    expect(suggest).toBeDisabled();
    fireEvent.click(await within(panel).findByLabelText("Use minutes/b.txt as suggestion exemplar"));
    fireEvent.change(within(panel).getByLabelText("Guidance (optional)"), { target: { value: "Decisions only" } });
    fireEvent.change(within(panel).getByLabelText("Reasoning effort"), { target: { value: "high" } });
    fireEvent.click(suggest);
    const result = await within(panel).findByRole("region", { name: "Suggested schema" });
    expect(requests.find(item => item.url === "/api/solutions/llm/suggest-schema")!.body)
      .toEqual({ document_ids: ["d2"], guidance: "Decisions only", settings: { reasoning_effort: "high" } });
    expect(within(result).getByText("Decisions are listed per item.")).toBeInTheDocument();
    expect(within(result).getByText("decisions")).toBeInTheDocument();
    expect(within(result).getByText(/1,500 tokens · 1 call/)).toBeInTheDocument();
    // The run settings keep their own control (unique ids, separate values).
    expect(screen.getAllByLabelText("Reasoning effort").map(item => (item as HTMLSelectElement).value)).toEqual(["low", "high"]);
    fireEvent.click(within(result).getByRole("button", { name: "Open in Workspace" }));
    expect(JSON.parse(localStorage.getItem("doc-cognition.schema-draft.v1")!)).toEqual(llmDraft);
    await waitFor(() => expect(window.location.hash).toBe("#/workspace"));
    expect(await screen.findByLabelText("Field name decisions")).toHaveValue("decisions");
  });

  it("asks before replacing an unsaved Workspace draft", async () => {
    localStorage.setItem("doc-cognition.schema-draft.v1", JSON.stringify({ ...sharedDraft, name: "Edited by hand" }));
    confirm.mockReturnValue(false);
    openSolution("/apps/llm");
    const panel = await screen.findByRole("region", { name: "Schema suggestion" });
    fireEvent.click(await within(panel).findByLabelText("Use minutes/a.txt as suggestion exemplar"));
    fireEvent.click(within(panel).getByRole("button", { name: /Suggest schema/ }));
    fireEvent.click(await within(panel).findByRole("button", { name: "Open in Workspace" }));
    expect(confirm).toHaveBeenCalled();
    expect(JSON.parse(localStorage.getItem("doc-cognition.schema-draft.v1")!).name).toBe("Edited by hand");
    expect(window.location.hash).toBe("#/apps/llm");
  });

  it("falls back to the default for a cleared or mistyped stored setting", async () => {
    solutions[1] = { ...solutions[1], status: { ready: true, detail: "", model: "nu" },
      schema_suggestion: { supported: true, method: "Native template generation.", settings: NUEXTRACT_SPECS } };
    localStorage.setItem("doc-cognition.suggest-settings.nuextract", JSON.stringify({ enable_thinking: "yes", max_tokens: null }));
    openSolution("/apps/nuextract");
    const panel = await screen.findByRole("region", { name: "Schema suggestion" });
    fireEvent.click(await within(panel).findByLabelText("Use minutes/a.txt as suggestion exemplar"));
    expect(within(panel).getByLabelText("Maximum output tokens")).toHaveValue(4096);
    expect(within(panel).queryByText(/enter a number/)).not.toBeInTheDocument();
    const suggest = within(panel).getByRole("button", { name: /Suggest schema/ });
    expect(suggest).toBeEnabled();
    fireEvent.click(suggest);
    await waitFor(() => expect(requests.some(item => item.url === "/api/solutions/nuextract/suggest-schema")).toBe(true));
    expect(requests.find(item => item.url === "/api/solutions/nuextract/suggest-schema")!.body)
      .toEqual({ document_ids: ["d1"], guidance: "", settings: { enable_thinking: false, max_tokens: 4096 } });
  });

  it("shows the reason instead of a panel when the solution cannot suggest", async () => {
    openSolution("/apps/unstract");
    const note = await screen.findByRole("region", { name: "Schema suggestion" });
    expect(within(note).getByText(UNSTRACT_REASON)).toBeInTheDocument();
    expect(within(note).queryByRole("button")).not.toBeInTheDocument();
  });

  it("puts LangExtract's panel on the Batch runs tab", async () => {
    openSolution("/apps/langextract?tab=runs");
    const panel = await screen.findByRole("region", { name: "Schema suggestion" });
    expect(within(panel).getByText(/Derived from a generated recipe/)).toBeInTheDocument();
  });
});
