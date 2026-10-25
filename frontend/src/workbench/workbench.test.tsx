import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import App from "../App";
import { schemaErrors } from "./components/SchemaEditor";
import type { SchemaDraft, SolutionInfo } from "./types";

const draft: SchemaDraft = {
  name: "Meeting minutes", description: "Board minutes",
  fields: [
    { name: "meeting_date", type: "date", description: "", array: false, options: [], fields: [] },
    { name: "attendees", type: "object", description: "", array: true, options: [], fields: [
      { name: "name", type: "verbatim", description: "", array: false, options: [], fields: [] },
    ] },
  ],
};
const schema = { ...draft, id: "s1", version: 1, created_at: "2026-10-25T00:00:00Z", exemplars: [] };
const collection = { id: "c1", name: "minutes", created_at: "2026-10-25", updated_at: "2026-10-25", documents: [
  { id: "d1", path: "minutes/a.txt", name: "a.txt", characters: 33, pages: null, sha256: "x" },
] };
const solution = (overrides: Partial<SolutionInfo> = {}): SolutionInfo => ({
  id: "llm", name: "Direct LLM", tagline: "Prompted", description: "Calls the model.", technology: "Azure OpenAI",
  runtime: "cloud", capabilities: ["Quotes"], confidence_method: "Model-reported.", route: "/apps/llm", accent: "#7dbfff",
  external_url: null, status: { ready: true, detail: "", model: "gpt" },
  settings: [{ key: "reasoning_effort", label: "Reasoning effort", type: "select", default: "low", help: "", min: null, max: null, options: ["low", "high"] }],
  ...overrides,
});
const text = "Minutes of 22 October 2019. Present: Simon Dow.";
const run = {
  id: "r1", solution: "llm", solution_name: "Direct LLM", label: "First run", model: "gpt", confidence_method: "Model-reported.",
  status: "done", error: null, extraction_schema: schema, settings: {}, created_at: "2026-10-25T10:00:00Z", active: false,
  documents: [{ document_id: "d1", document_name: "a.txt", document_sha256: "x", status: "done", error: null,
    data: { meeting_date: "2019-10-22", attendees: [{ name: "Simon Dow" }] },
    values: [
      { path: ["meeting_date"], field: "meeting_date", value: "2019-10-22", quote: "22 October 2019", confidence: 0.92, grounding: "exact", start: text.indexOf("22 October"), end: text.indexOf("22 October") + 15 },
      { path: ["attendees", 0, "name"], field: "attendees[].name", value: "Simon Dow", quote: "Simon Dow", confidence: 0.55, grounding: "exact", start: text.indexOf("Simon"), end: text.indexOf("Simon") + 9 },
    ],
    elapsed_seconds: 1.5, usage: { calls: 1, prompt_tokens: 10, completion_tokens: 5 }, notes: ["A note"], raw: null,
    started_at: null, finished_at: null }],
  summary: { id: "r1", solution: "llm", solution_name: "Direct LLM", label: "First run", model: "gpt", status: "done", error: null,
    schema_id: "s1", schema_version: 1, schema_name: "Meeting minutes", created_at: "2026-10-25T10:00:00Z", started_at: null,
    finished_at: null, documents_total: 1, documents_done: 1, documents_failed: 0, document_ids: ["d1"], elapsed_seconds: 1.5,
    values: 2, grounded: 2, mean_confidence: 0.74, usage: { calls: 1, prompt_tokens: 10, completion_tokens: 5 }, confidence_method: "Model-reported." },
};

let requests: { url: string; body: unknown }[] = [];
let solutions: SolutionInfo[] = [];

beforeEach(() => {
  localStorage.clear();
  window.location.hash = "";
  requests = [];
  solutions = [solution(), solution({ id: "unstract", name: "Unstract", route: "/apps/unstract", runtime: "local",
    external_url: "http://frontend.unstract.localhost", status: { ready: false, detail: "Set UNSTRACT_API_URL.", model: "" } })];
  vi.stubGlobal("confirm", () => true);
  vi.stubGlobal("fetch", vi.fn(async (input: string, init?: RequestInit) => {
    const body = typeof init?.body === "string" ? JSON.parse(init.body) : init?.body;
    requests.push({ url: input, body });
    if (input === "/api/solutions") return Response.json(solutions);
    if (input === "/api/runs" && !init?.method) return Response.json([]);
    if (input === "/api/runs" && init?.method === "POST") return Response.json(run.summary);
    if (input === "/api/runs/r1") return Response.json(run);
    if (input === "/api/documents/d1") return Response.json({ id: "d1", name: "a.txt", text, characters: text.length, pages: null, page_starts: [], sha256: "x" });
    if (input === "/api/collections" && init?.method === "POST") return Response.json({ ...collection, documents: [] });
    if (input === "/api/collections") return Response.json([]);
    if (input === "/api/collections/c1") return Response.json(collection);
    if (input === "/api/documents" && init?.method === "POST") return Response.json({ id: "d1", name: "a.txt", text, characters: 33, pages: null });
    if (input === "/api/schemas/s1") return Response.json(schema);
    if (input === "/api/schemas/suggest") return Response.json({ draft, notes: ["Attendance varies."], usage: {}, exemplars: ["a.txt"] });
    if (input === "/api/schema-suggestions") return Response.json([]);
    if (input === "/api/schemas" && init?.method === "POST") return Response.json(schema);
    return Response.json({ detail: `Unexpected ${input}` }, { status: 500 });
  }));
});

function withWorkspace() {
  localStorage.setItem("doc-cognition.workspace.v1", JSON.stringify({ collectionId: "c1", schemaId: "s1" }));
}

it("lists registered solutions and launches packaged ones externally", async () => {
  render(<App />);
  const cards = await screen.findAllByRole("article");
  expect(cards).toHaveLength(2);
  expect(within(cards[0]).getByRole("heading", { name: "Direct LLM" })).toBeInTheDocument();
  expect(within(cards[0]).getByText("Ready")).toBeInTheDocument();
  const external = within(cards[1]).getByRole("link", { name: /Open Unstract/ });
  expect(external).toHaveAttribute("href", "http://frontend.unstract.localhost");
  expect(external).toHaveAttribute("target", "_blank");
  expect(within(cards[1]).getByText("Set UNSTRACT_API_URL.")).toBeInTheDocument();
});

it("processes a chosen folder once into a shared document set", async () => {
  window.location.hash = "#/workspace";
  render(<App />);
  fireEvent.change(await screen.findByLabelText("Choose workspace folder"), {
    target: { files: [new File(["Minutes"], "a.txt"), new File(["x"], "image.png")] },
  });
  await screen.findByText("Processed 1 of 1 documents");
  const upload = requests.find(item => item.url === "/api/documents")!;
  expect((upload.body as FormData).get("collection_id")).toBe("c1");
  expect(JSON.parse(localStorage.getItem("doc-cognition.workspace.v1")!).collectionId).toBe("c1");
});

it("suggests a schema from exemplars and saves it as the agreed schema", async () => {
  localStorage.setItem("doc-cognition.workspace.v1", JSON.stringify({ collectionId: "c1", schemaId: null }));
  window.location.hash = "#/workspace";
  render(<App />);
  fireEvent.click(await screen.findByLabelText("Use minutes/a.txt as exemplar"));
  fireEvent.click(screen.getByRole("button", { name: /Suggest from 1 exemplar/ }));
  await screen.findByText("Attendance varies.");
  expect(requests.find(item => item.url === "/api/schemas/suggest")!.body).toMatchObject({ document_ids: ["d1"] });
  expect(screen.getByLabelText("Field name attendees.name")).toHaveValue("name");
  fireEvent.click(screen.getByRole("button", { name: "Save as agreed schema" }));
  await screen.findByText(/Saved “Meeting minutes” as version 1/);
  expect(requests.find(item => item.url === "/api/schemas" && item.body)!.body).toMatchObject({ draft, exemplars: ["a.txt"] });
});

it("validates schema names, nesting and enum options", () => {
  expect(schemaErrors(draft)).toEqual([]);
  const broken: SchemaDraft = { ...draft, fields: [...draft.fields, { ...draft.fields[0] },
    { name: "Status", type: "enum", description: "", array: false, options: [], fields: [] }] };
  const errors = schemaErrors(broken);
  expect(errors.some(error => error.includes("duplicate"))).toBe(true);
  expect(errors.some(error => error.includes("snake_case"))).toBe(true);
  expect(errors.some(error => error.includes("options"))).toBe(true);
});

it("starts a run with the agreed schema and shows grounded, scored values", async () => {
  withWorkspace();
  window.location.hash = "#/apps/llm";
  render(<App />);
  const start = await screen.findByRole("button", { name: "Run on 1 document" });
  await waitFor(() => expect(start).toBeEnabled());
  fireEvent.change(screen.getByLabelText("Reasoning effort"), { target: { value: "high" } });
  fireEvent.click(start);
  await waitFor(() => expect(window.location.hash).toContain("run=r1"));
  expect(requests.find(item => item.url === "/api/runs" && item.body)!.body).toEqual({
    solution: "llm", label: "", schema_id: "s1", settings: { reasoning_effort: "high" }, document_ids: ["d1"],
  });
  const record = await screen.findByRole("button", { name: "meeting_date: 2019-10-22" });
  expect(within(record).getByText("92%")).toBeInTheDocument();
  expect(screen.getByText("A note")).toBeInTheDocument();
  fireEvent.click(record);
  expect(await screen.findByRole("button", { name: "22 October 2019" })).toHaveAttribute("aria-pressed", "true");
});

it("ignores stored run settings that no longer fit their type", async () => {
  withWorkspace();
  solutions = [solution({ settings: [
    { key: "max_tokens", label: "Maximum output tokens", type: "integer", default: 4096, help: "", min: 256, max: 32768, options: null },
    { key: "reasoning_effort", label: "Reasoning effort", type: "select", default: "low", help: "", min: null, max: null, options: ["low", "high"] },
  ] })];
  localStorage.setItem("doc-cognition.settings.llm", JSON.stringify({ max_tokens: null, reasoning_effort: "extreme" }));
  window.location.hash = "#/apps/llm";
  render(<App />);
  expect(await screen.findByLabelText("Maximum output tokens")).toHaveValue(4096);
  expect(screen.getByLabelText("Reasoning effort")).toHaveValue("low");
});
