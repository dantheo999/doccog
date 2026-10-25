import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import App from "./App";

const storedRecipe = {
  id: "recipe-1", version: 1, goal: "Find people",
  instructions: "Extract people using exact source text.",
  source_name: "a.txt", created_at: "2026-10-12T00:00:00Z",
  source_characters_used: 13, source_characters_total: 13,
  examples: [{ text: "Alice signed.", extractions: [
    { extraction_class: "person", extraction_text: "Alice", attributes: {} },
  ] }],
};

beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal("fetch", vi.fn(async (input: string, init?: RequestInit) => {
    if (input === "/api/config") return Response.json({
      configured: true, deployment: "my-deployment", endpoint: "https://example.openai.azure.com",
      error: null, max_upload_mib: 20, max_characters: 500_000,
    });
    if (input === "/api/documents") {
      const file = (init?.body as FormData).get("file") as File;
      return Response.json({ id: file.name, name: file.name, text: "Alice signed.", characters: 13, pages: null });
    }
    if (input === "/api/recipes/generate") {
      const body = JSON.parse(init?.body as string);
      if (JSON.stringify(body.document_ids) !== '["a.txt"]' || body.goal !== "Find people") {
        return Response.json({ detail: "Generation used the wrong source or goal." }, { status: 422 });
      }
      return Response.json(storedRecipe);
    }
    if (input === "/api/extractions") {
      const body = JSON.parse(init?.body as string);
      if (body.document_id !== "b.txt" || body.recipe.id !== "recipe-1"
        || body.recipe.instructions !== storedRecipe.instructions) {
        return Response.json({ detail: "Extraction did not reuse the active recipe on the selected document." }, { status: 422 });
      }
      return Response.json({
      document_id: "b.txt", text: "Alice signed.", recipe_id: "recipe-1", recipe_version: 1,
      options: body.options,
      elapsed_seconds: 0.5, extractions: [{
        id: "0", extraction_class: "person", extraction_text: "Alice", attributes: {},
        start: 0, end: 5, alignment_status: "match_exact", grounding: "exact",
      }],
      });
    }
    return Response.json({ detail: "Unexpected endpoint" }, { status: 500 });
  }));
});

async function chooseFolder() {
  await screen.findByText("my-deployment");
  fireEvent.change(screen.getByLabelText("Choose document folder"), {
    target: { files: [new File(["Alice"], "a.txt"), new File(["Alice"], "b.txt")] },
  });
  fireEvent.click(screen.getByRole("button", { name: /a.txt/ }));
  await screen.findByText("13 characters");
}

it("keeps the generated recipe when extracting a different document", async () => {
  render(<App />);
  await chooseFolder();
  fireEvent.change(screen.getByLabelText("What do you want to extract?"), { target: { value: "Find people" } });
  fireEvent.click(screen.getByRole("button", { name: "Generate example" }));
  await screen.findByText("Recipe ready");
  fireEvent.click(screen.getByRole("button", { name: /b.txt/ }));
  await waitFor(() => expect(screen.getByRole("heading", { name: "b.txt" })).toBeInTheDocument());
  fireEvent.click(screen.getByRole("button", { name: "Extract" }));
  await screen.findByText("1 finding");
  expect(screen.getByLabelText("Generic extraction instructions")).toHaveValue(storedRecipe.instructions);
  expect(screen.getByRole("button", { name: /Alice.*exact/i })).toBeInTheDocument();
  expect(JSON.parse(localStorage.getItem("doc-cognition.recipe.v1")!).source_name).toBe("a.txt");
  fireEvent.click(screen.getByRole("button", { name: /Alice.*exact/i }));
  expect(screen.getByRole("button", { name: "Alice" })).toHaveAttribute("aria-pressed", "true");
  fireEvent.change(screen.getByLabelText("What do you want to extract?"), { target: { value: "Find deadlines instead" } });
  expect(screen.getByLabelText("Generic extraction instructions")).toHaveValue(storedRecipe.instructions);
});

it("preserves the existing recipe when generation fails", async () => {
  localStorage.setItem("doc-cognition.recipe.v1", JSON.stringify(storedRecipe));
  const originalFetch = globalThis.fetch;
  vi.stubGlobal("fetch", async (input: string, init?: RequestInit) =>
    input === "/api/recipes/generate"
      ? Response.json({ detail: "Azure rate limit reached." }, { status: 429 })
      : originalFetch(input, init));
  render(<App />);
  await chooseFolder();
  fireEvent.click(screen.getByRole("button", { name: "Generate example" }));
  await screen.findByText("Azure rate limit reached.");
  expect(screen.getByLabelText("Generic extraction instructions")).toHaveValue(storedRecipe.instructions);
});

it("does not apply a stale preview after selecting another document", async () => {
  const originalFetch = globalThis.fetch;
  let release: (value: Response) => void = () => {};
  vi.stubGlobal("fetch", (input: string, init?: RequestInit) => {
    if (input === "/api/documents" && ((init?.body as FormData).get("file") as File).name === "a.txt") {
      return new Promise<Response>(resolve => { release = resolve; });
    }
    return originalFetch(input, init);
  });
  render(<App />);
  await screen.findByText("my-deployment");
  fireEvent.change(screen.getByLabelText("Choose document folder"), {
    target: { files: [new File(["A"], "a.txt"), new File(["B"], "b.txt")] },
  });
  fireEvent.click(screen.getByRole("button", { name: /a.txt/ }));
  fireEvent.click(screen.getByRole("button", { name: /b.txt/ }));
  await screen.findByText("13 characters");
  await act(async () => release(Response.json({ id: "a.txt", name: "a.txt", text: "Wrong stale text", characters: 16, pages: null })));
  expect(screen.getByRole("heading", { name: "b.txt" })).toBeInTheDocument();
  expect(screen.queryByText("Wrong stale text")).not.toBeInTheDocument();
});

it("saves extraction settings separately and sends them without changing the recipe", async () => {
  localStorage.setItem("doc-cognition.recipe.v1", JSON.stringify(storedRecipe));
  render(<App />);
  await chooseFolder();
  fireEvent.click(screen.getByRole("button", { name: "Extraction settings" }));
  fireEvent.change(screen.getByLabelText("Target chunk size (characters)"), { target: { value: "1200" } });
  fireEvent.change(screen.getByLabelText("Maximum concurrent LLM requests"), { target: { value: "8" } });
  fireEvent.change(screen.getByLabelText("Extraction passes"), { target: { value: "3" } });
  fireEvent.change(screen.getByLabelText("Previous-chunk context (characters)"), { target: { value: "300" } });
  fireEvent.click(screen.getByRole("button", { name: "Save settings" }));
  const options = { chunk_size: 1200, max_workers: 8, extraction_passes: 3, context_window_chars: 300 };
  expect(JSON.parse(localStorage.getItem("doc-cognition.extraction-options.v1")!)).toEqual(options);
  expect(JSON.parse(localStorage.getItem("doc-cognition.recipe.v1")!)).toEqual(storedRecipe);
  fireEvent.click(screen.getByRole("button", { name: /b.txt/ }));
  await waitFor(() => expect(screen.getByRole("heading", { name: "b.txt" })).toBeInTheDocument());
  fireEvent.click(screen.getByRole("button", { name: "Extract" }));
  await screen.findByText("1 finding");
  const calls = vi.mocked(fetch).mock.calls.filter(([input]) => input === "/api/extractions");
  expect(JSON.parse(calls[0][1]!.body as string).options).toEqual(options);
});

it("restores saved settings and warns when saved settings are corrupt", async () => {
  localStorage.setItem("doc-cognition.extraction-options.v1", "not json");
  render(<App />);
  expect(await screen.findByText(/Could not restore.*settings/i)).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Extraction settings" }));
  expect(screen.getByLabelText("Target chunk size (characters)")).toHaveValue(6000);
  expect(screen.getByLabelText("Maximum concurrent LLM requests")).toHaveValue(2);
});

it("restores valid settings without adding them to the recipe", () => {
  localStorage.setItem("doc-cognition.extraction-options.v1", JSON.stringify({
    chunk_size: 20000, max_workers: 16, extraction_passes: 5, context_window_chars: 6000,
  }));
  render(<App />);
  fireEvent.click(screen.getByRole("button", { name: "Extraction settings" }));
  expect(screen.getByLabelText("Target chunk size (characters)")).toHaveValue(20000);
  expect(screen.getByLabelText("Maximum concurrent LLM requests")).toHaveValue(16);
  expect(screen.getByLabelText("Extraction passes")).toHaveValue(5);
  expect(screen.getByLabelText("Previous-chunk context (characters)")).toHaveValue(6000);
  expect(localStorage.getItem("doc-cognition.recipe.v1")).toBeNull();
});

it.each(["cancel", "close", "escape"])("discards unsaved settings on %s and restores focus", mode => {
  render(<App />);
  const cog = screen.getByRole("button", { name: "Extraction settings" });
  fireEvent.click(cog);
  const dialog = screen.getByRole("dialog", { name: "Extraction settings" });
  fireEvent.change(within(dialog).getByLabelText("Maximum concurrent LLM requests"), { target: { value: "8" } });
  if (mode === "escape") fireEvent(dialog, new Event("cancel", { cancelable: true }));
  else fireEvent.click(within(dialog).getByRole("button", { name: mode === "cancel" ? "Cancel" : "Close settings" }));
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(cog).toHaveFocus();
  expect(localStorage.getItem("doc-cognition.extraction-options.v1")).toBeNull();
  fireEvent.click(cog);
  expect(screen.getByLabelText("Maximum concurrent LLM requests")).toHaveValue(2);
});

it("resets drafts only until Save applies defaults", () => {
  localStorage.setItem("doc-cognition.extraction-options.v1", JSON.stringify({
    chunk_size: 1000, max_workers: 8, extraction_passes: 2, context_window_chars: 500,
  }));
  render(<App />);
  fireEvent.click(screen.getByRole("button", { name: "Extraction settings" }));
  fireEvent.click(screen.getByRole("button", { name: "Reset to defaults" }));
  expect(screen.getByLabelText("Target chunk size (characters)")).toHaveValue(6000);
  expect(JSON.parse(localStorage.getItem("doc-cognition.extraction-options.v1")!).max_workers).toBe(8);
  fireEvent.click(screen.getByRole("button", { name: "Save settings" }));
  expect(JSON.parse(localStorage.getItem("doc-cognition.extraction-options.v1")!)).toEqual({
    chunk_size: 6000, max_workers: 2, extraction_passes: 1, context_window_chars: 0,
  });
});

it.each([
  ["Target chunk size (characters)", "499"], ["Target chunk size (characters)", "20001"],
  ["Maximum concurrent LLM requests", "0"], ["Maximum concurrent LLM requests", "17"],
  ["Extraction passes", "0"], ["Extraction passes", "6"],
  ["Previous-chunk context (characters)", "-1"], ["Previous-chunk context (characters)", "6001"],
  ["Maximum concurrent LLM requests", "2.5"], ["Previous-chunk context (characters)", ""],
])("rejects invalid %s value %s without applying it", (label, value) => {
  render(<App />);
  fireEvent.click(screen.getByRole("button", { name: "Extraction settings" }));
  const dialog = screen.getByRole("dialog", { name: "Extraction settings" });
  fireEvent.change(within(dialog).getByLabelText(label), { target: { value } });
  fireEvent.click(within(dialog).getByRole("button", { name: "Save settings" }));
  expect(within(dialog).getByRole("alert")).not.toBeEmptyDOMElement();
  expect(within(dialog).getByRole("alert")).toHaveFocus();
  expect(localStorage.getItem("doc-cognition.extraction-options.v1")).toBeNull();
});

it.each([true, "4", 4.5, null])("does not coerce invalid persisted concurrency %s", value => {
  localStorage.setItem("doc-cognition.extraction-options.v1", JSON.stringify({
    chunk_size: 6000, max_workers: value, extraction_passes: 1, context_window_chars: 0,
  }));
  render(<App />);
  expect(screen.getByText(/Could not restore.*settings/i)).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Extraction settings" }));
  expect(screen.getByLabelText("Maximum concurrent LLM requests")).toHaveValue(2);
});

it("keeps a settings snapshot for an in-flight extraction", async () => {
  localStorage.setItem("doc-cognition.recipe.v1", JSON.stringify(storedRecipe));
  const originalFetch = globalThis.fetch;
  let release: (value: Response) => void = () => {};
  vi.stubGlobal("fetch", (input: string, init?: RequestInit) => input === "/api/extractions"
    ? new Promise<Response>(resolve => { release = resolve; })
    : originalFetch(input, init));
  render(<App />);
  await chooseFolder();
  fireEvent.click(screen.getByRole("button", { name: "Extract" }));
  await screen.findByRole("button", { name: "Extracting…" });
  fireEvent.click(screen.getByRole("button", { name: "Extraction settings" }));
  fireEvent.change(screen.getByLabelText("Maximum concurrent LLM requests"), { target: { value: "8" } });
  fireEvent.click(screen.getByRole("button", { name: "Save settings" }));
  await act(async () => release(Response.json({
    document_id: "a.txt", text: "Alice signed.", recipe_id: "recipe-1", recipe_version: 1,
    elapsed_seconds: 0.5, extractions: [],
    options: { chunk_size: 6000, max_workers: 2, extraction_passes: 1, context_window_chars: 0 },
  })));
  expect(screen.getByLabelText("Extraction run settings")).toHaveTextContent(/2 concurrent/);
  expect(JSON.parse(localStorage.getItem("doc-cognition.extraction-options.v1")!).max_workers).toBe(8);
});

it("keeps settings active and reports failed browser persistence", () => {
  render(<App />);
  fireEvent.click(screen.getByRole("button", { name: "Extraction settings" }));
  fireEvent.change(screen.getByLabelText("Maximum concurrent LLM requests"), { target: { value: "8" } });
  const failingStorage = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
    throw new DOMException("Storage quota exceeded", "QuotaExceededError");
  });
  try {
    fireEvent.click(screen.getByRole("button", { name: "Save settings" }));
    expect(screen.getByText(/settings.*browser persistence failed/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Extraction settings" }));
    expect(screen.getByLabelText("Maximum concurrent LLM requests")).toHaveValue(8);
  } finally { failingStorage.mockRestore(); }
});
