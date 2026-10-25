import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import App from "../App";
import ValueHighlighter from "./components/ValueHighlighter";
import type { SchemaDraft, SolutionInfo } from "./types";

const draft: SchemaDraft = {
  name: "Meeting minutes", description: "Board minutes",
  fields: [{ name: "meeting_date", type: "date", description: "", array: false, options: [], fields: [] }],
};
const schema = { ...draft, id: "s1", version: 1, created_at: "2026-10-25T00:00:00Z", exemplars: [] };
const policies = { name: "Policies", description: "", id: "s2", version: 3, created_at: "2026-10-24T00:00:00Z", exemplars: [],
  fields: [{ name: "policy_title", type: "verbatim", description: "", array: false, options: [], fields: [] }] };
const collection = { id: "c1", name: "minutes", created_at: "2026-10-25", updated_at: "2026-10-25", documents: [
  { id: "d1", path: "minutes/a.txt", name: "a.txt", characters: 30, pages: null, sha256: "x" },
  { id: "d2", path: "minutes/b.txt", name: "b.txt", characters: 30, pages: null, sha256: "y" },
] };
const solution: SolutionInfo = {
  id: "llm", name: "Direct LLM", tagline: "Prompted", description: "Calls the model.", technology: "Azure OpenAI",
  runtime: "cloud", capabilities: [], confidence_method: "Model-reported.", route: "/apps/llm", accent: "#7dbfff",
  external_url: null, status: { ready: true, detail: "", model: "gpt" }, settings: [],
};
const texts: Record<string, string> = { d1: "First document. Simon Dow chaired.", d2: "Second document. Liz Butler chaired." };
const documentResult = (id: string, name: string, value: string) => ({
  document_id: id, document_name: name, document_sha256: id === "d1" ? "x" : "y", status: "done", error: null,
  data: { chair: value }, elapsed_seconds: 1, usage: { calls: 1, prompt_tokens: 1, completion_tokens: 1 }, notes: [], raw: null,
  started_at: null, finished_at: null,
  values: [{ path: ["chair"], field: "chair", value, quote: value, confidence: 0.9, grounding: "exact",
    start: texts[id].indexOf(value), end: texts[id].indexOf(value) + value.length }],
});
const summary = { id: "r1", solution: "llm", solution_name: "Direct LLM", label: "Run", model: "gpt", status: "done", error: null,
  schema_id: "s1", schema_version: 1, schema_name: "Meeting minutes", created_at: "2026-10-25T10:00:00Z", started_at: null,
  finished_at: null, documents_total: 2, documents_done: 2, documents_failed: 0, document_ids: ["d1", "d2"], elapsed_seconds: 2,
  values: 2, grounded: 2, mean_confidence: 0.9, usage: { calls: 2, prompt_tokens: 2, completion_tokens: 2 }, confidence_method: "Model-reported." };
const run = { ...summary, confidence_method: "Model-reported.", extraction_schema: schema, settings: {}, active: false, summary,
  documents: [documentResult("d1", "a.txt", "Simon Dow"), documentResult("d2", "b.txt", "Liz Butler")] };
const storedDocument = (id: string) => ({ id, name: `${id}.txt`, text: texts[id], characters: texts[id].length, pages: null,
  page_starts: [], sha256: id === "d1" ? "x" : "y" });

type Handler = (input: string, init?: RequestInit) => Promise<Response> | Response | undefined;
let override: Handler = () => undefined;
let requests: { url: string; body: unknown }[] = [];

beforeEach(() => {
  localStorage.clear();
  window.location.hash = "";
  requests = [];
  override = () => undefined;
  vi.stubGlobal("confirm", () => true);
  vi.stubGlobal("fetch", vi.fn(async (input: string, init?: RequestInit) => {
    const body = typeof init?.body === "string" ? JSON.parse(init.body) : init?.body;
    requests.push({ url: input, body });
    const custom = await override(input, init);
    if (custom) return custom;
    if (input === "/api/solutions") return Response.json([solution]);
    if (input === "/api/runs" && !init?.method) return Response.json([summary]);
    if (input === "/api/runs/r1") return Response.json(run);
    if (input.startsWith("/api/documents/")) return Response.json(storedDocument(input.split("/").pop()!));
    if (input === "/api/collections") return Response.json([{ ...collection, documents: 2 }]);
    if (input === "/api/collections/c1") return Response.json(collection);
    if (input === "/api/schemas" && !init?.method) return Response.json([
      { id: "s1", name: "Meeting minutes", version: 1, created_at: schema.created_at, fields: 1 },
      { id: "s2", name: "Policies", version: 3, created_at: policies.created_at, fields: 1 }]);
    if (input === "/api/schemas/s1") return Response.json(schema);
    if (input === "/api/schemas/s2") return Response.json(policies);
    if (input === "/api/schema-suggestions") return Response.json([]);
    return Response.json({ detail: `Unexpected ${input}` }, { status: 500 });
  }));
});

afterEach(() => { vi.useRealTimers(); });

function withWorkspace(schemaId: string | null = "s1") {
  localStorage.setItem("doc-cognition.workspace.v1", JSON.stringify({ collectionId: "c1", schemaId }));
}

/** jsdom's File has no text(). */
const jsonFile = (content: string) => Object.assign(new File([content], "schema.json", { type: "application/json" }),
  { text: async () => content });

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>(done => { resolve = done; });
  return { promise, resolve };
}

it("keeps polling a running run after a failed refresh and says so", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  let polls = 0;
  override = input => {
    if (input !== "/api/runs/r1") return undefined;
    polls += 1;
    if (polls === 2) return Response.json({ detail: "Bad gateway" }, { status: 502 });
    return Response.json(polls < 3 ? { ...run, status: "running", active: true } : run);
  };
  withWorkspace();
  window.location.hash = "#/apps/llm?run=r1";
  const { container } = render(<App />);
  await waitFor(() => expect(container.querySelector(".run-status")?.textContent).toBe("running"));
  await act(() => vi.advanceTimersByTimeAsync(2_100));
  expect(await screen.findByText(/Could not refresh this run: Bad gateway Retrying/)).toBeInTheDocument();
  await act(() => vi.advanceTimersByTimeAsync(4_100));
  await waitFor(() => expect(container.querySelector(".run-status")?.textContent).toBe("done"));
  expect(polls).toBe(3);
  expect(screen.queryByText(/Could not refresh this run/)).not.toBeInTheDocument();
});

it("shows a document that failed to load and never highlights the previous one with the next one's values", async () => {
  const second = deferred<Response>();
  let failFirst = true;
  override = input => {
    if (input === "/api/documents/d1" && failFirst) return Response.json({ detail: "Document expired." }, { status: 410 });
    if (input === "/api/documents/d2") return second.promise;
    return undefined;
  };
  withWorkspace();
  window.location.hash = "#/apps/llm?run=r1";
  const { container } = render(<App />);
  const source = () => container.querySelector(".run-source")!;
  expect((await screen.findByText("Document expired.")).closest(".run-source")).not.toBeNull();
  failFirst = false;
  fireEvent.click(screen.getByRole("button", { name: "Retry loading the document" }));
  expect(await screen.findByRole("button", { name: "Simon Dow" })).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: /b\.txt/ }));
  await waitFor(() => expect(source().textContent).toContain("Loading document…"));
  expect(source().textContent).not.toContain("Simon Dow");
  expect(screen.queryByText(/stored document text changed/)).not.toBeInTheDocument();
  second.resolve(Response.json(storedDocument("d2")));
  expect(await screen.findByRole("button", { name: "Liz Butler" })).toBeInTheDocument();
});

it("imports hand-written schema JSON with missing lists and reports what is wrong", async () => {
  withWorkspace(null);
  window.location.hash = "#/workspace";
  render(<App />);
  const file = jsonFile(JSON.stringify({ name: "Hand-written", fields: [
    { name: "title", type: "text" }, { name: "status", type: "enum" }, { name: "parties", type: "object" }] }));
  fireEvent.change(await screen.findByLabelText("Import schema file"), { target: { files: [file] } });
  expect(await screen.findByText("status: a choice field needs options.")).toBeInTheDocument();
  expect(screen.getByText("parties: an object needs child fields.")).toBeInTheDocument();
  expect(screen.getByLabelText("Field name title")).toHaveValue("title");
  fireEvent.change(screen.getByLabelText("Import schema file"), {
    target: { files: [jsonFile('{"name": "x", "fields": {}}')] } });
  expect(await screen.findByText(/no "fields" list/)).toBeInTheDocument();
});

it("picks a saved schema and saves a draft as a new schema or a new version", async () => {
  override = (input, init) => {
    if (input !== "/api/schemas" || init?.method !== "POST") return undefined;
    const body = JSON.parse(init.body as string);
    // The backend trims the name; the editor must follow, or the draft stays "unsaved" forever.
    return Response.json({ ...body.draft, name: body.draft.name.trim(), id: body.schema_id ?? "s9",
      version: body.schema_id ? 2 : 1, created_at: "2026-10-26T00:00:00Z", exemplars: [] });
  };
  withWorkspace();
  window.location.hash = "#/workspace";
  render(<App />);
  const picker = await screen.findByLabelText("Saved schema");
  await waitFor(() => expect(picker).toHaveValue("s1"));
  expect(within(picker).getAllByRole("option").map(option => option.textContent)).toEqual([
    "Meeting minutes · v1 · 1 fields", "Policies · v3 · 1 fields"]);
  fireEvent.change(picker, { target: { value: "s2" } });
  await waitFor(() => expect(screen.getByLabelText("Schema name")).toHaveValue("Policies"));
  expect(screen.getByText("Policies", { selector: "b" })).toBeInTheDocument();

  fireEvent.change(screen.getByLabelText("Schema name"), { target: { value: "Policies and agreements " } });
  fireEvent.click(screen.getByRole("button", { name: "Save as new version" }));
  await screen.findByText(/Saved “Policies and agreements” as version 2/);
  expect(requests.filter(item => item.url === "/api/schemas" && item.body).at(-1)!.body).toMatchObject({ schema_id: "s2" });
  expect(screen.getByRole("button", { name: "Saved" })).toBeDisabled();
  expect(screen.queryByText(/Unsaved changes/)).not.toBeInTheDocument();

  fireEvent.click(screen.getByRole("button", { name: "Save as new schema" }));
  await screen.findByText(/Saved “Policies and agreements” as version 1/);
  expect(requests.filter(item => item.url === "/api/schemas" && item.body).at(-1)!.body).toMatchObject({ schema_id: null });
});

it("offers the only document set when none is selected and keeps the last preview clicked", async () => {
  const first = deferred<Response>();
  override = input => input === "/api/documents/d1" ? first.promise : undefined;
  window.location.hash = "#/workspace";
  render(<App />);
  const select = await screen.findByRole("combobox", { name: "Document set" });
  fireEvent.change(select, { target: { value: "c1" } });
  fireEvent.click(await screen.findByRole("button", { name: /a\.txt/ }));
  fireEvent.click(screen.getByRole("button", { name: /b\.txt/ }));
  expect(await screen.findByRole("dialog", { name: "Preview of d2.txt" })).toBeInTheDocument();
  first.resolve(Response.json(storedDocument("d1")));
  await new Promise(resolve => setTimeout(resolve, 20));
  expect(screen.getByRole("dialog", { name: "Preview of d2.txt" })).toBeInTheDocument();
  expect(screen.queryByRole("dialog", { name: "Preview of d1.txt" })).not.toBeInTheDocument();
});

it("shows why a solution page cannot load and retries", async () => {
  let down = true;
  override = input => input === "/api/solutions" && down ? Response.json({ detail: "Bad gateway" }, { status: 502 }) : undefined;
  window.location.hash = "#/apps/llm";
  render(<App />);
  expect(await screen.findByText("Could not load the solutions: Bad gateway")).toBeInTheDocument();
  expect(screen.queryByText(/No solution called/)).not.toBeInTheDocument();
  down = false;
  fireEvent.click(screen.getByRole("button", { name: "Retry" }));
  expect(await screen.findByRole("heading", { name: "Direct LLM", level: 1 })).toBeInTheDocument();
});

it("marks every page start in the source text, with or without values", () => {
  const text = "Page one text.Page two text.Page three.";
  const { container, rerender } = render(<ValueHighlighter text={text} values={[]} selected={null} onSelect={() => {}}
    pageStarts={[0, 14, 28]} />);
  expect([...container.querySelectorAll(".page-break")].map(node => node.textContent)).toEqual(["page 2", "page 3"]);
  rerender(<ValueHighlighter text={text} selected={null} onSelect={() => {}} pageStarts={[0, 14, 28]} values={[
    { path: ["a"], field: "a", value: "one text.Page two", quote: null, confidence: null, grounding: "exact", start: 5, end: 22 }]} />);
  expect([...container.querySelectorAll(".page-break")].map(node => node.textContent)).toEqual(["page 2", "page 3"]);
});
