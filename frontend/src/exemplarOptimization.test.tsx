import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import App from "./LangExtractStudio";
import { StrictMode } from "react";
import userEvent from "@testing-library/user-event";

const options = { chunk_size: 6000, max_workers: 2, extraction_passes: 1, context_window_chars: 0 };
const original = {
  id: "recipe-1", version: 1, goal: "Find people",
  instructions: "Extract people using exact source text.",
  source_name: "a.txt", created_at: "2026-10-12T00:00:00Z",
  source_characters_used: 13, source_characters_total: 13,
  examples: [{ text: "Alice signed.", extractions: [
    { extraction_class: "person", extraction_text: "Alice", attributes: { role: "signer" } },
  ] }],
};
const candidate = { ...original, version: 2, instructions: "Extract named people and their explicit roles." };
const proposal = { base_recipe_id: "recipe-1", base_recipe_version: 1, candidate,
  reasons: ["Clarify explicit role attributes; findings are not ground truth."] };
const twoSourceRecipe = { ...original, source_name: "nda.txt, arac.txt",
  source_characters_used: 26, source_characters_total: 26, examples: [
  { ...original.examples[0], source: {
    name: "nda.txt", document_id: "nda", document_sha256: "a".repeat(64),
    start: 0, end: 13, text_sha256: "b".repeat(64), verified: true,
  } },
  { text: "Bob approved.", extractions: [
    { extraction_class: "person", extraction_text: "Bob", attributes: {} },
  ], source: {
    name: "arac.txt", document_id: "arac", document_sha256: "c".repeat(64),
    start: 0, end: 13, text_sha256: "d".repeat(64), verified: true,
  } },
] };
let uploads: string[];
let generationBody: unknown;
let optimizationBody: any;
let extractionCount: number;
let responseProposal: unknown;
let generationRecipe: unknown;
let documentText: string;
let findings: any[];
let failure: string | null;

beforeEach(() => {
  localStorage.clear();
  localStorage.setItem("doc-cognition.recipe.v1", JSON.stringify(original));
  uploads = [];
  generationBody = null;
  optimizationBody = null;
  extractionCount = 0;
  responseProposal = proposal;
  generationRecipe = { ...original, id: "generated" };
  documentText = "Alice signed.\n😀 Full document tail.";
  findings = [
    { id: "0", extraction_class: "person", extraction_text: "Alice", attributes: { role: "signer" },
      start: 0, end: 5, alignment_status: "match_exact", grounding: "exact" },
    { id: "1", extraction_class: "action", extraction_text: "signed", attributes: {},
      start: 6, end: 12, alignment_status: "match_exact", grounding: "exact" },
  ];
  failure = null;
  vi.stubGlobal("fetch", vi.fn(async (input: string, init?: RequestInit) => {
    if (input === "/api/config") return Response.json({
      configured: true, deployment: "test-model", endpoint: "https://example.invalid", error: null,
      max_upload_mib: 20, max_characters: 500_000,
    });
    if (input === "/api/documents") {
      const file = (init?.body as FormData).get("file") as File;
      uploads.push(file.name);
      if (file.name === "broken.txt") return Response.json({ detail: "Cannot read selected exemplar." }, { status: 422 });
      return Response.json({ id: file.name, name: file.name, text: documentText,
        characters: Array.from(documentText).length, pages: null });
    }
    if (input === "/api/recipes/generate") {
      generationBody = JSON.parse(init!.body as string);
      return Response.json(generationRecipe);
    }
    if (input === "/api/extractions") {
      extractionCount++;
      const body = JSON.parse(init!.body as string);
      return Response.json({ document_id: body.document_id, text: documentText,
        recipe_id: body.recipe.id, recipe_version: body.recipe.version,
        elapsed_seconds: 0.2, extractions: findings, options: body.options });
    }
    if (input === "/api/recipes/optimize") {
      optimizationBody = JSON.parse(init!.body as string);
      return failure ? Response.json({ detail: failure }, { status: 422 }) : Response.json(responseProposal);
    }
    return Response.json({ detail: "Unexpected endpoint" }, { status: 500 });
  }));
});

async function folder(names = ["a.txt", "b.txt", "c.txt"]) {
  await screen.findByText("test-model");
  fireEvent.change(screen.getByLabelText("Choose document folder"), {
    target: { files: names.map(name => new File(["Alice signed."], name)) },
  });
}

async function preview(name = "a.txt") {
  fireEvent.click(screen.getByRole("button", { name: new RegExp(name.replace(".", "\\.")) }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Extract" })).toBeEnabled());
}

async function completed() {
  await folder();
  await preview();
  fireEvent.click(screen.getByRole("button", { name: "Extract" }));
  await screen.findByText(`${findings.length} findings`);
}

function beginDeferredImport(name = "delayed-recipe.json") {
  let resolve!: (text: string) => void;
  let reject!: (error: Error) => void;
  const file = new File(["delayed"], name);
  Object.defineProperty(file, "text", { value: () => new Promise<string>((finish, fail) => {
    resolve = finish;
    reject = fail;
  }) });
  fireEvent.change(screen.getByLabelText("Recipe JSON file"), { target: { files: [file] } });
  return { resolve, reject };
}

it.each([
  twoSourceRecipe.examples.slice(0, 1),
  [twoSourceRecipe.examples[0], twoSourceRecipe.examples[0]],
  [...twoSourceRecipe.examples, original.examples[0]],
  [twoSourceRecipe.examples[1], twoSourceRecipe.examples[0]],
  [twoSourceRecipe.examples[0], { ...twoSourceRecipe.examples[1], text: "Bob approved. Again.",
    source: { ...twoSourceRecipe.examples[1].source, verified: false } }],
  [twoSourceRecipe.examples[0], { ...twoSourceRecipe.examples[1],
    source: { ...twoSourceRecipe.examples[1].source, name: "replacement.txt" } }],
].map(examples => ({ examples })))("rejects optimization that changes the original exemplar set: %j", async ({ examples }) => {
  localStorage.setItem("doc-cognition.recipe.v1", JSON.stringify(twoSourceRecipe));
  responseProposal = { ...proposal, candidate: { ...twoSourceRecipe, version: 2, examples } };
  render(<App />);
  await completed();
  fireEvent.click(screen.getByRole("button", { name: "Optimize" }));
  await screen.findByRole("alert");
  expect(screen.queryByRole("dialog", { name: "Review optimization" })).not.toBeInTheDocument();
  expect(JSON.parse(localStorage.getItem("doc-cognition.recipe.v1")!)).toEqual(twoSourceRecipe);
  expect(screen.getByText("2 findings")).toBeInTheDocument();
});

it("shows both original exemplar texts and counts when reviewing and applying optimized annotations", async () => {
  localStorage.setItem("doc-cognition.recipe.v1", JSON.stringify(twoSourceRecipe));
  const revised = { ...twoSourceRecipe, version: 2, examples: twoSourceRecipe.examples.map(example => ({
    ...example, extractions: example.extractions.map(item => ({ ...item, attributes: { reviewed: true } })),
  })) };
  responseProposal = { ...proposal, candidate: revised };
  render(<App />);
  await completed();
  fireEvent.click(screen.getByRole("button", { name: "Optimize" }));
  const dialog = await screen.findByRole("dialog", { name: "Review optimization" });
  fireEvent.click(within(dialog).getByRole("tab", { name: "Example text" }));
  for (const name of ["Before", "After"]) {
    const side = within(dialog).getByRole("region", { name });
    expect(within(side).getByRole("heading")).toHaveTextContent("2 examples");
    expect(within(side).getByText("nda.txt")).toBeVisible();
    expect(within(side).getByText("arac.txt")).toBeVisible();
    expect(within(side).getByText("Alice signed.")).toBeVisible();
    expect(within(side).getByText("Bob approved.")).toBeVisible();
  }
  fireEvent.click(within(dialog).getByRole("button", { name: "Apply revision" }));
  const saved = JSON.parse(localStorage.getItem("doc-cognition.recipe.v1")!);
  expect(saved.examples.map((example: { text: string }) => example.text)).toEqual(["Alice signed.", "Bob approved."]);
  expect(saved.examples[1].extractions[0].attributes).toEqual({ reviewed: true });
  expect(extractionCount).toBe(1);
});

it("rejects proposals that revise unverified annotations instead of retaining them unchanged", async () => {
  responseProposal = { ...proposal, candidate: { ...candidate, examples: [{
    ...original.examples[0], extractions: [
      { ...original.examples[0].extractions[0], attributes: { role: "replacement" } },
    ],
  }] } };
  render(<App />);
  await completed();
  fireEvent.click(screen.getByRole("button", { name: "Optimize" }));
  await screen.findByRole("alert");
  expect(screen.queryByRole("dialog", { name: "Review optimization" })).not.toBeInTheDocument();
  expect(JSON.parse(localStorage.getItem("doc-cognition.recipe.v1")!)).toEqual(original);
});

it("selects exemplar checkboxes independently and uploads only checked sources without replacing preview or findings", async () => {
  render(<App />);
  await completed();
  fireEvent.click(screen.getByRole("checkbox", { name: "Use a.txt as exemplar" }));
  fireEvent.click(screen.getByRole("checkbox", { name: "Use c.txt as exemplar" }));
  expect(screen.getByRole("checkbox", { name: "Use c.txt as exemplar" }).closest("button")).toBeNull();
  expect(screen.getByText(/2 exemplar documents/)).toHaveTextContent(/a.txt.*c.txt/);
  fireEvent.click(screen.getByRole("button", { name: "Generate example" }));
  await waitFor(() => expect(generationBody).toEqual({ document_ids: ["a.txt", "c.txt"], goal: "Find people" }));
  expect(uploads).toEqual(["a.txt", "c.txt"]);
  expect(screen.getByRole("heading", { name: "a.txt" })).toBeInTheDocument();
  expect(screen.getByText("2 findings")).toBeInTheDocument();
});

it("generates, persists and edits complete long examples without trimming the text", async () => {
  documentText = "Alice signed.\r\n" + "Unlabelled body.\n".repeat(1500) + "😀 COMPLETE TAIL";
  generationRecipe = { ...original, id: "whole-document", examples: [
    { ...original.examples[0], text: documentText },
  ] };
  render(<App />);
  await folder();
  await preview();
  fireEvent.click(screen.getByRole("button", { name: "Generate example" }));
  await waitFor(() => expect(JSON.parse(localStorage.getItem("doc-cognition.recipe.v1")!).id).toBe("whole-document"));
  fireEvent.click(screen.getByRole("button", { name: "Expand Example text" }));
  const dialog = screen.getByRole("dialog", { name: "Active recipe" });
  expect(within(dialog).getByText((_, element) => element?.tagName === "P"
    && element.textContent === documentText)).toBeVisible();
  fireEvent.click(within(dialog).getByRole("button", { name: "Close full-page view" }));
  fireEvent.change(screen.getByLabelText("Generic extraction instructions"), { target: { value: "Find all people, not only example labels." } });
  fireEvent.click(screen.getByRole("button", { name: "Apply edits" }));
  const saved = JSON.parse(localStorage.getItem("doc-cognition.recipe.v1")!);
  expect(saved.version).toBe(2);
  expect(saved.examples[0].text).toBe(documentText);
  expect(saved.examples[0].extractions).toHaveLength(1);
});

it("reviews and applies a whole-document optimization candidate without automatic extraction", async () => {
  documentText = "Alice signed.\r\n" + "Unlabelled body.\n".repeat(1500) + "😀 COMPLETE TAIL";
  localStorage.setItem("doc-cognition.recipe.v1", JSON.stringify({ ...original, examples: [
    { ...original.examples[0], text: documentText },
  ] }));
  responseProposal = { ...proposal, candidate: { ...candidate, examples: [
    { ...candidate.examples[0], text: documentText },
  ] } };
  render(<App />);
  await completed();
  fireEvent.click(screen.getByRole("button", { name: "Optimize" }));
  const dialog = await screen.findByRole("dialog", { name: "Review optimization" });
  fireEvent.click(within(dialog).getByRole("tab", { name: "Example text" }));
  for (const name of ["Before", "After"]) {
    expect(within(within(dialog).getByRole("region", { name })).getByText((_, element) => element?.tagName === "P"
      && element.textContent === documentText)).toBeVisible();
  }
  fireEvent.click(within(dialog).getByRole("button", { name: "Apply revision" }));
  expect(JSON.parse(localStorage.getItem("doc-cognition.recipe.v1")!).examples[0].text).toBe(documentText);
  expect(extractionCount).toBe(1);
  expect(screen.getByText(/These findings use an earlier recipe/)).toBeInTheDocument();
});

it("keeps whole-document text out of both optimization annotation comparisons", async () => {
  render(<App />);
  await completed();
  fireEvent.click(screen.getByRole("button", { name: "Optimize" }));
  const dialog = await screen.findByRole("dialog", { name: "Review optimization" });
  fireEvent.click(within(dialog).getByRole("tab", { name: "Few-shot JSON" }));
  for (const name of ["Before", "After"]) {
    const section = within(dialog).getByRole("region", { name });
    expect(JSON.parse(section.querySelector("pre")!.textContent!)).toEqual([
      { extractions: [{ extraction_class: "person", extraction_text: "Alice", attributes: { role: "signer" } }] },
    ]);
    expect(section).not.toHaveTextContent("Alice signed.");
  }
});

it("allows five independent sources without a preview and clears only selection and documents on folder reset", async () => {
  render(<App />);
  await folder(["1.txt", "2.txt", "3.txt", "4.txt", "5.txt", "6.txt"]);
  for (const name of ["1.txt", "2.txt", "3.txt", "4.txt", "5.txt"]) {
    fireEvent.click(screen.getByRole("checkbox", { name: `Use ${name} as exemplar` }));
  }
  expect(screen.getByRole("checkbox", { name: "Use 6.txt as exemplar" })).toBeDisabled();
  expect(screen.getByRole("checkbox", { name: "Use 1.txt as exemplar" })).toBeEnabled();
  fireEvent.click(screen.getByRole("button", { name: "Generate example" }));
  await waitFor(() => expect(generationBody).toEqual({
    document_ids: ["1.txt", "2.txt", "3.txt", "4.txt", "5.txt"], goal: "Find people",
  }));
  expect(uploads).toEqual(["1.txt", "2.txt", "3.txt", "4.txt", "5.txt"]);
  expect(screen.getByRole("heading", { name: "Select a document" })).toBeInTheDocument();
  await folder(["new.txt"]);
  expect(screen.getByRole("checkbox", { name: "Use new.txt as exemplar" })).not.toBeChecked();
  expect(screen.getByText("Recipe ready")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Generate example" })).toBeDisabled();
});

it("falls back to the active preview when no exemplars are checked", async () => {
  render(<App />);
  await folder();
  await preview();
  fireEvent.click(screen.getByRole("button", { name: "Generate example" }));
  await waitFor(() => expect(generationBody).toEqual({ document_ids: ["a.txt"], goal: "Find people" }));
  expect(uploads).toEqual(["a.txt"]);
});

it("fails selected-source generation atomically without losing the active recipe", async () => {
  render(<App />);
  await folder(["a.txt", "broken.txt"]);
  await preview();
  fireEvent.click(screen.getByRole("checkbox", { name: "Use broken.txt as exemplar" }));
  fireEvent.click(screen.getByRole("button", { name: "Generate example" }));
  await screen.findByText("Cannot read selected exemplar.");
  expect(generationBody).toBeNull();
  expect(screen.getByLabelText("Generic extraction instructions")).toHaveValue(original.instructions);
  expect(JSON.parse(localStorage.getItem("doc-cognition.recipe.v1")!)).toEqual(original);
});

it("gates Optimize on a matching completed extraction, including zero findings", async () => {
  findings = [];
  render(<App />);
  const optimize = screen.getByRole("button", { name: "Optimize" });
  expect(optimize).toBeDisabled();
  await completed();
  expect(optimize).toBeEnabled();
  fireEvent.change(screen.getByLabelText("Generic extraction instructions"), { target: { value: "Unapplied draft." } });
  expect(optimize).toBeEnabled();
  fireEvent.click(screen.getByRole("button", { name: "Apply edits" }));
  expect(optimize).toBeDisabled();
  expect(screen.getByText(/These findings use an earlier recipe/)).toBeInTheDocument();
});

it("sends the full unfiltered extraction and applied recipe, not drafts or current settings", async () => {
  render(<App />);
  await completed();
  fireEvent.click(screen.getByRole("button", { name: "person" }));
  fireEvent.change(screen.getByLabelText("Generic extraction instructions"), { target: { value: "Unapplied draft." } });
  fireEvent.click(screen.getByRole("button", { name: "Extraction settings" }));
  fireEvent.change(screen.getByLabelText("Maximum concurrent LLM requests"), { target: { value: "8" } });
  fireEvent.click(screen.getByRole("button", { name: "Save settings" }));
  fireEvent.click(screen.getByRole("button", { name: "Optimize" }));
  await screen.findByRole("dialog", { name: "Review optimization" });
  expect(optimizationBody).toEqual({ document_id: "a.txt", recipe: original, result: {
    document_id: "a.txt", text: "Alice signed.\n😀 Full document tail.", recipe_id: "recipe-1", recipe_version: 1,
    elapsed_seconds: 0.2, extractions: findings, options,
  } });
});

it.each(["discard", "close", "escape", "native-close"])("preserves recipe, findings and drafts on optimization %s and returns focus", async mode => {
  render(<App />);
  await completed();
  fireEvent.change(screen.getByLabelText("Generic extraction instructions"), { target: { value: "Keep my draft." } });
  fireEvent.click(screen.getByRole("button", { name: "Optimize" }));
  const dialog = await screen.findByRole("dialog", { name: "Review optimization" });
  expect(within(dialog).getByText(original.instructions)).toBeVisible();
  expect(within(dialog).getByText(candidate.instructions)).toBeVisible();
  expect(within(dialog).getByText(proposal.reasons[0])).toBeVisible();
  expect(within(dialog).getByText(/replaces.*unapplied.*drafts/i)).toBeVisible();
  if (mode === "escape") fireEvent(dialog, new Event("cancel", { cancelable: true }));
  else if (mode === "native-close") await act(async () => (dialog as HTMLDialogElement).close());
  else fireEvent.click(within(dialog).getByRole("button", { name: mode === "discard" ? "Discard" : "Close review" }));
  expect(screen.queryByRole("dialog", { name: "Review optimization" })).not.toBeInTheDocument();
  expect(screen.getByLabelText("Generic extraction instructions")).toHaveValue("Keep my draft.");
  expect(screen.getByText("2 findings")).toBeInTheDocument();
  expect(JSON.parse(localStorage.getItem("doc-cognition.recipe.v1")!)).toEqual(original);
  expect(screen.getByRole("button", { name: "Optimize" })).toHaveFocus();
});

it("applies exactly the proposed next version, resets drafts and keeps findings stale without extracting", async () => {
  render(<App />);
  await completed();
  fireEvent.change(screen.getByLabelText("Generic extraction instructions"), { target: { value: "Replace my draft." } });
  fireEvent.click(screen.getByRole("button", { name: "Optimize" }));
  const dialog = await screen.findByRole("dialog", { name: "Review optimization" });
  for (const label of ["Few-shot JSON", "Example text", "Instructions"]) {
    fireEvent.click(within(dialog).getByRole("tab", { name: label }));
    expect(within(dialog).getByRole("tab", { name: label })).toHaveAttribute("aria-selected", "true");
  }
  fireEvent.click(within(dialog).getByRole("button", { name: "Apply revision" }));
  expect(screen.queryByRole("dialog", { name: "Review optimization" })).not.toBeInTheDocument();
  expect(screen.getByLabelText("Generic extraction instructions")).toHaveValue(candidate.instructions);
  expect(JSON.parse(localStorage.getItem("doc-cognition.recipe.v1")!)).toEqual(candidate);
  expect(screen.getByText("2 findings")).toBeInTheDocument();
  expect(screen.getByText(/These findings use an earlier recipe/)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Optimize" })).toBeDisabled();
  expect(extractionCount).toBe(1);
});

it.each([
  { ...proposal, base_recipe_id: "wrong" },
  { ...proposal, base_recipe_version: 2 },
  { ...proposal, candidate: { ...candidate, id: "wrong" } },
  { ...proposal, candidate: { ...candidate, version: 3 } },
  { ...proposal, candidate: { ...candidate, goal: "Different goal" } },
  { ...proposal, reasons: [] },
  { ...proposal, reasons: [" "] },
  { ...proposal, unknown: true },
])("rejects inconsistent or malformed proposals %j without replacing active state", async value => {
  responseProposal = value;
  render(<App />);
  await completed();
  fireEvent.click(screen.getByRole("button", { name: "Optimize" }));
  await screen.findByRole("alert");
  expect(screen.queryByRole("dialog", { name: "Review optimization" })).not.toBeInTheDocument();
  expect(screen.getByLabelText("Generic extraction instructions")).toHaveValue(original.instructions);
  expect(screen.getByText("2 findings")).toBeInTheDocument();
  expect(JSON.parse(localStorage.getItem("doc-cognition.recipe.v1")!)).toEqual(original);
});

it("retains active recipe, findings and drafts on optimization failure", async () => {
  failure = "Full input exceeds model context capacity; nothing was truncated.";
  render(<App />);
  await completed();
  fireEvent.change(screen.getByLabelText("Generic extraction instructions"), { target: { value: "Keep draft." } });
  fireEvent.click(screen.getByRole("button", { name: "Optimize" }));
  await screen.findByText(failure);
  expect(screen.getByLabelText("Generic extraction instructions")).toHaveValue("Keep draft.");
  expect(screen.getByText("2 findings")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Optimize" })).toBeEnabled();
});

it("blocks conflicting actions during optimization but permits settings for later runs", async () => {
  const baseFetch = fetch;
  let release!: (value: Response) => void;
  vi.stubGlobal("fetch", (input: string, init?: RequestInit) => input === "/api/recipes/optimize"
    ? new Promise<Response>(resolve => { release = resolve; }) : baseFetch(input, init));
  render(<App />);
  await completed();
  fireEvent.click(screen.getByRole("button", { name: "Optimize" }));
  expect(screen.getByRole("button", { name: "Optimizing…" })).toBeDisabled();
  for (const name of ["Choose folder", "Extract", "Generate example", "Apply edits", "Import recipe JSON",
    "Expand Instructions", "Expand Few-shot JSON", "Expand Example text", "b.txt"]) {
    expect(screen.getByRole("button", { name: name === "b.txt" ? /b.txt/ : name })).toBeDisabled();
  }
  expect(screen.getByLabelText("Generic extraction instructions")).toBeDisabled();
  expect(screen.getByRole("checkbox", { name: "Use b.txt as exemplar" })).toBeDisabled();
  expect(screen.getByRole("button", { name: "Extraction settings" })).toBeEnabled();
  await act(async () => release(Response.json(proposal)));
  await screen.findByRole("dialog", { name: "Review optimization" });
});

it("keeps optimization review open under the application's StrictMode effect replay", async () => {
  render(<StrictMode><App /></StrictMode>);
  await completed();
  fireEvent.click(screen.getByRole("button", { name: "Optimize" }));
  const dialog = await screen.findByRole("dialog", { name: "Review optimization" });
  expect(dialog).toBeVisible();
  fireEvent.click(within(dialog).getByRole("button", { name: "Discard" }));
  expect(screen.getByRole("button", { name: "Optimize" })).toHaveFocus();
});

it("finishes open settings before showing an asynchronously completed optimization review", async () => {
  const baseFetch = fetch;
  let release!: (value: Response) => void;
  vi.stubGlobal("fetch", (input: string, init?: RequestInit) => input === "/api/recipes/optimize"
    ? new Promise<Response>(resolve => { release = resolve; }) : baseFetch(input, init));
  render(<App />);
  await completed();
  fireEvent.click(screen.getByRole("button", { name: "Optimize" }));
  fireEvent.click(screen.getByRole("button", { name: "Extraction settings" }));
  fireEvent.change(screen.getByLabelText("Maximum concurrent LLM requests"), { target: { value: "8" } });
  await act(async () => release(Response.json(proposal)));
  expect(screen.queryByRole("dialog", { name: "Review optimization" })).not.toBeInTheDocument();
  expect(screen.getByRole("dialog", { name: "Extraction settings" })).toBeVisible();
  expect(screen.getByRole("button", { name: "Extract" })).toBeDisabled();
  expect(screen.getByRole("button", { name: "Generate example" })).toBeDisabled();
  expect(screen.getByRole("button", { name: "Expand Instructions" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Save settings" }));
  const review = await screen.findByRole("dialog", { name: "Review optimization" });
  fireEvent.click(within(review).getByRole("button", { name: "Discard" }));
  expect(screen.getByRole("button", { name: "Optimize" })).toHaveFocus();
  expect(JSON.parse(localStorage.getItem("doc-cognition.extraction-options.v1")!).max_workers).toBe(8);
});

it("does not commit an in-flight recipe import during model optimization", async () => {
  render(<App />);
  await completed();
  let finishImport!: (text: string) => void;
  const file = new File(["delayed"], "recipe.json");
  Object.defineProperty(file, "text", { value: () => new Promise<string>(resolve => { finishImport = resolve; }) });
  fireEvent.change(screen.getByLabelText("Recipe JSON file"), { target: { files: [file] } });
  fireEvent.click(screen.getByRole("button", { name: "Optimize" }));
  const dialog = await screen.findByRole("dialog", { name: "Review optimization" });
  await act(async () => finishImport(JSON.stringify({ ...original, instructions: "Imported different instructions." })));
  fireEvent.click(within(dialog).getByRole("button", { name: "Discard" }));
  expect(screen.getByLabelText("Generic extraction instructions")).toHaveValue(original.instructions);
  expect(JSON.parse(localStorage.getItem("doc-cognition.recipe.v1")!)).toEqual(original);
});

it("invalidates a pending import even when it completes after optimization Apply returns the UI to idle", async () => {
  render(<App />);
  await completed();
  fireEvent.change(screen.getByLabelText("Generic extraction instructions"), { target: { value: "Unapplied draft." } });
  const pending = beginDeferredImport();
  fireEvent.click(screen.getByRole("button", { name: "Optimize" }));
  const dialog = await screen.findByRole("dialog", { name: "Review optimization" });
  fireEvent.click(within(dialog).getByRole("button", { name: "Apply revision" }));
  await waitFor(() => expect(screen.getByLabelText("Generic extraction instructions")).toHaveValue(candidate.instructions));
  expect(screen.getByRole("button", { name: "Import recipe JSON" })).toBeEnabled();
  await act(async () => pending.resolve(JSON.stringify({ ...original, instructions: "Obsolete imported instructions." })));
  expect(screen.getByLabelText("Generic extraction instructions")).toHaveValue(candidate.instructions);
  expect(screen.getByLabelText("Few-shot extraction JSON")).toHaveValue(JSON.stringify([
    { extractions: [{ extraction_class: "person", extraction_text: "Alice", attributes: { role: "signer" } }] },
  ], null, 2));
  expect(JSON.parse(localStorage.getItem("doc-cognition.recipe.v1")!)).toEqual(candidate);
  expect(screen.getByText("2 findings")).toBeInTheDocument();
  expect(screen.getByText(/These findings use an earlier recipe/)).toBeInTheDocument();
  expect(extractionCount).toBe(1);
});

it.each(["discard", "close", "escape", "native-close"])(
  "invalidates an old import that completes after optimization %s without erasing drafts or focus", async mode => {
    render(<App />);
    await completed();
    fireEvent.change(screen.getByLabelText("Generic extraction instructions"), { target: { value: "Keep my instruction draft." } });
    fireEvent.click(screen.getByRole("tab", { name: "Few-shot JSON" }));
    fireEvent.change(screen.getByLabelText("Few-shot extraction JSON"), { target: { value: "Keep my JSON draft too." } });
    const pending = beginDeferredImport();
    fireEvent.click(screen.getByRole("button", { name: "Optimize" }));
    const dialog = await screen.findByRole("dialog", { name: "Review optimization" });
    if (mode === "escape") fireEvent(dialog, new Event("cancel", { cancelable: true }));
    else if (mode === "native-close") await act(async () => (dialog as HTMLDialogElement).close());
    else fireEvent.click(within(dialog).getByRole("button", { name: mode === "discard" ? "Discard" : "Close review" }));
    expect(screen.getByRole("button", { name: "Import recipe JSON" })).toBeEnabled();
    await act(async () => pending.resolve(JSON.stringify({ ...original, instructions: "Obsolete imported instructions." })));
    expect(screen.getByLabelText("Generic extraction instructions")).toHaveValue("Keep my instruction draft.");
    expect(screen.getByLabelText("Few-shot extraction JSON")).toHaveValue("Keep my JSON draft too.");
    expect(JSON.parse(localStorage.getItem("doc-cognition.recipe.v1")!)).toEqual(original);
    expect(screen.getByText("2 findings")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Optimize" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Optimize" })).toHaveFocus();
  });

it("invalidates pending imports on recipe replacement without a model operation", async () => {
  render(<App />);
  await screen.findByText("test-model");
  const pending = beginDeferredImport();
  fireEvent.change(screen.getByLabelText("Generic extraction instructions"), { target: { value: "Latest manual edit." } });
  fireEvent.click(screen.getByRole("button", { name: "Apply edits" }));
  await waitFor(() => expect(screen.getByText("v2 · from a.txt")).toBeInTheDocument());
  await act(async () => pending.resolve(JSON.stringify({ ...original, instructions: "Obsolete imported instructions." })));
  expect(screen.getByLabelText("Generic extraction instructions")).toHaveValue("Latest manual edit.");
  expect(JSON.parse(localStorage.getItem("doc-cognition.recipe.v1")!)).toEqual({
    ...original, version: 2, instructions: "Latest manual edit.",
  });
});

it.each(["older-first", "newer-first"])("only applies the latest import when reads finish %s", async order => {
  render(<App />);
  await screen.findByText("test-model");
  const older = beginDeferredImport("older.json");
  const newer = beginDeferredImport("newer.json");
  const oldRecipe = { ...original, instructions: "Obsolete older import." };
  const newRecipe = { ...original, goal: "Find named people", instructions: "Latest imported instructions." };
  if (order === "older-first") {
    await act(async () => older.resolve(JSON.stringify(oldRecipe)));
    expect(screen.getByLabelText("Generic extraction instructions")).toHaveValue(original.instructions);
    expect(JSON.parse(localStorage.getItem("doc-cognition.recipe.v1")!)).toEqual(original);
    expect((screen.getByLabelText("Recipe JSON file") as HTMLInputElement).files?.[0].name).toBe("newer.json");
    await act(async () => newer.resolve(JSON.stringify(newRecipe)));
  } else {
    await act(async () => newer.resolve(JSON.stringify(newRecipe)));
    await act(async () => older.resolve(JSON.stringify(oldRecipe)));
  }
  expect(screen.getByLabelText("Generic extraction instructions")).toHaveValue(newRecipe.instructions);
  expect(screen.getByLabelText("What do you want to extract?")).toHaveValue(newRecipe.goal);
  expect(JSON.parse(localStorage.getItem("doc-cognition.recipe.v1")!)).toMatchObject({
    version: 1, goal: "Find named people", instructions: "Latest imported instructions.",
  });
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
});

it("ignores an obsolete import read failure after a newer import succeeds", async () => {
  render(<App />);
  await screen.findByText("test-model");
  const older = beginDeferredImport("older.json");
  const newer = beginDeferredImport("newer.json");
  await act(async () => newer.resolve(JSON.stringify({ ...original, instructions: "Latest imported instructions." })));
  screen.getByRole("button", { name: "Import recipe JSON" }).focus();
  await act(async () => older.reject(new Error("Obsolete file read failed.")));
  expect(screen.getByLabelText("Generic extraction instructions")).toHaveValue("Latest imported instructions.");
  expect(screen.queryAllByText("Obsolete file read failed.")).toHaveLength(0);
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Import recipe JSON" })).toHaveFocus();
});

it("does not fall back to an older pending import when the latest import is invalid", async () => {
  render(<App />);
  await screen.findByText("test-model");
  const older = beginDeferredImport("older.json");
  const newer = beginDeferredImport("newer.json");
  await act(async () => newer.resolve("not JSON"));
  expect(screen.getAllByRole("alert")[0]).not.toBeEmptyDOMElement();
  await act(async () => older.resolve(JSON.stringify({ ...original, instructions: "Obsolete older import." })));
  expect(screen.getByLabelText("Generic extraction instructions")).toHaveValue(original.instructions);
  expect(JSON.parse(localStorage.getItem("doc-cognition.recipe.v1")!)).toEqual(original);
  expect(screen.getAllByRole("alert")[0]).not.toBeEmptyDOMElement();
});

it("allows choosing the same file again after its pending import is invalidated", async () => {
  const user = userEvent.setup();
  render(<App />);
  await completed();
  let finishOld!: (text: string) => void;
  let reads = 0;
  const file = new File(["delayed"], "repeat.json", { type: "application/json" });
  Object.defineProperty(file, "text", { value: () => ++reads === 1
    ? new Promise<string>(resolve => { finishOld = resolve; })
    : Promise.resolve(JSON.stringify({ ...original, instructions: "Re-imported same file." })) });
  const input = screen.getByLabelText("Recipe JSON file");
  await user.upload(input, file);
  fireEvent.click(screen.getByRole("button", { name: "Optimize" }));
  const dialog = await screen.findByRole("dialog", { name: "Review optimization" });
  fireEvent.click(within(dialog).getByRole("button", { name: "Discard" }));
  await user.upload(input, file);
  await waitFor(() => expect(screen.getByLabelText("Generic extraction instructions")).toHaveValue("Re-imported same file."));
  await act(async () => finishOld(JSON.stringify({ ...original, instructions: "Obsolete previous read." })));
  expect(screen.getByLabelText("Generic extraction instructions")).toHaveValue("Re-imported same file.");
  expect(JSON.parse(localStorage.getItem("doc-cognition.recipe.v1")!)).toMatchObject({ instructions: "Re-imported same file." });
});

it("navigates optimization comparison tabs with arrow keys and Home/End", async () => {
  render(<App />);
  await completed();
  fireEvent.click(screen.getByRole("button", { name: "Optimize" }));
  const dialog = await screen.findByRole("dialog", { name: "Review optimization" });
  const instructions = within(dialog).getByRole("tab", { name: "Instructions" });
  instructions.focus();
  fireEvent.keyDown(instructions, { key: "ArrowRight" });
  expect(within(dialog).getByRole("tab", { name: "Few-shot JSON" })).toHaveFocus();
  expect(within(dialog).getByRole("tab", { name: "Few-shot JSON" })).toHaveAttribute("aria-selected", "true");
  fireEvent.keyDown(within(dialog).getByRole("tab", { name: "Few-shot JSON" }), { key: "End" });
  expect(within(dialog).getByRole("tab", { name: "Example text" })).toHaveFocus();
  fireEvent.keyDown(within(dialog).getByRole("tab", { name: "Example text" }), { key: "Home" });
  expect(instructions).toHaveFocus();
});

it("allows replacing a checked source at the five-source limit, even when the list is filtered", async () => {
  render(<App />);
  await folder(["1.txt", "2.txt", "3.txt", "4.txt", "5.txt", "6.txt"]);
  for (const name of ["1.txt", "2.txt", "3.txt", "4.txt", "5.txt"]) {
    fireEvent.click(screen.getByRole("checkbox", { name: `Use ${name} as exemplar` }));
  }
  fireEvent.change(screen.getByLabelText("Filter documents"), { target: { value: "1.txt" } });
  fireEvent.click(screen.getByRole("checkbox", { name: "Use 1.txt as exemplar" }));
  fireEvent.change(screen.getByLabelText("Filter documents"), { target: { value: "6.txt" } });
  fireEvent.click(screen.getByRole("checkbox", { name: "Use 6.txt as exemplar" }));
  fireEvent.click(screen.getByRole("button", { name: "Generate example" }));
  await waitFor(() => expect(generationBody).toEqual({
    document_ids: ["2.txt", "3.txt", "4.txt", "5.txt", "6.txt"], goal: "Find people",
  }));
  expect(uploads).toEqual(["2.txt", "3.txt", "4.txt", "5.txt", "6.txt"]);
});

it("ignores an old folder's pending upload when a new folder is previewed", async () => {
  const baseFetch = fetch;
  let release!: (value: Response) => void;
  vi.stubGlobal("fetch", (input: string, init?: RequestInit) => input === "/api/documents"
    && ((init!.body as FormData).get("file") as File).name === "a.txt"
    ? new Promise<Response>(resolve => { release = resolve; }) : baseFetch(input, init));
  render(<App />);
  await folder();
  fireEvent.click(screen.getByRole("checkbox", { name: "Use a.txt as exemplar" }));
  fireEvent.click(screen.getByRole("button", { name: /a.txt/ }));
  await folder(["new.txt"]);
  await preview("new.txt");
  await act(async () => release(Response.json({ id: "old", name: "a.txt", text: "Stale text", characters: 10, pages: null })));
  expect(screen.getByRole("heading", { name: "new.txt" })).toBeInTheDocument();
  expect(screen.queryByText("Stale text")).not.toBeInTheDocument();
  expect(screen.getByRole("checkbox", { name: "Use new.txt as exemplar" })).not.toBeChecked();
  fireEvent.click(screen.getByRole("button", { name: "Generate example" }));
  await waitFor(() => expect(generationBody).toEqual({ document_ids: ["new.txt"], goal: "Find people" }));
});

it("rejects oversized selected sources before uploading them and retains the extraction", async () => {
  render(<App />);
  await completed();
  // The size-only fixture avoids allocating an oversized document to exercise the upload boundary.
  const oversized = new File(["small fixture"], "large.txt");
  Object.defineProperty(oversized, "size", { value: 20 * 1024 * 1024 + 1 });
  await folder(["a.txt"]);
  fireEvent.change(screen.getByLabelText("Choose document folder"), {
    target: { files: [new File(["Alice signed."], "a.txt"), oversized] },
  });
  await preview();
  fireEvent.click(screen.getByRole("button", { name: "Extract" }));
  await screen.findByText("2 findings");
  fireEvent.click(screen.getByRole("checkbox", { name: "Use large.txt as exemplar" }));
  fireEvent.click(screen.getByRole("button", { name: "Generate example" }));
  await screen.findByText("Document exceeds the 20 MiB upload limit.");
  expect(uploads).not.toContain("large.txt");
  expect(generationBody).toBeNull();
  expect(screen.getByText("2 findings")).toBeInTheDocument();
  expect(screen.getByLabelText("Generic extraction instructions")).toHaveValue(original.instructions);
});

it("disables optimization after a document switch, even with checked exemplars retained", async () => {
  render(<App />);
  await completed();
  fireEvent.click(screen.getByRole("checkbox", { name: "Use a.txt as exemplar" }));
  await preview("b.txt");
  expect(screen.getByRole("button", { name: "Optimize" })).toBeDisabled();
  expect(screen.getByRole("checkbox", { name: "Use a.txt as exemplar" })).toBeChecked();
});

it("renders retained legacy and verified source cards in both comparisons without persisting the candidate", async () => {
  const mixed = { ...twoSourceRecipe, examples: [original.examples[0], twoSourceRecipe.examples[1]] };
  localStorage.setItem("doc-cognition.recipe.v1", JSON.stringify(mixed));
  responseProposal = { ...proposal, candidate: { ...mixed, version: 2, instructions: candidate.instructions } };
  render(<App />);
  await completed();
  fireEvent.click(screen.getByRole("button", { name: "Optimize" }));
  const dialog = await screen.findByRole("dialog", { name: "Review optimization" });
  fireEvent.click(within(dialog).getByRole("tab", { name: "Example text" }));
  for (const name of ["Before", "After"]) {
    const side = within(within(dialog).getByRole("region", { name }));
    expect(side.getByText(/Legacy example.*unverified/)).toBeVisible();
    expect(side.getByText("Verified source text")).toBeVisible();
    expect(side.getByText(/Source span 0–13/)).toBeVisible();
  }
  expect(JSON.parse(localStorage.getItem("doc-cognition.recipe.v1")!)).toEqual(mixed);
  expect(Object.keys(localStorage)).toEqual(["doc-cognition.recipe.v1"]);
});

it("keeps an applied optimization active with a persistence warning when browser storage fails", async () => {
  render(<App />);
  await completed();
  fireEvent.click(screen.getByRole("button", { name: "Optimize" }));
  const dialog = await screen.findByRole("dialog", { name: "Review optimization" });
  const failingStorage = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
    throw new DOMException("Storage quota exceeded", "QuotaExceededError");
  });
  try {
    fireEvent.click(within(dialog).getByRole("button", { name: "Apply revision" }));
    expect(screen.getByText(/Recipe is active.*browser persistence failed/)).toBeVisible();
    expect(screen.getByLabelText("Generic extraction instructions")).toHaveValue(candidate.instructions);
    expect(screen.getByText(/These findings use an earlier recipe/)).toBeVisible();
    expect(extractionCount).toBe(1);
  } finally { failingStorage.mockRestore(); }
});
