import { useState } from "react";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { expect, it } from "vitest";
import type { Recipe } from "../types";
import RecipePanel from "./RecipePanel";

const sampleRecipe: Recipe = {
  id: "review-recipe", version: 1, goal: "Find people",
  instructions: "Extract people using exact source text.",
  source_name: "sample.txt", created_at: "2026-10-12T00:00:00Z",
  source_characters_used: 13, source_characters_total: 13,
  examples: [{ text: "Alice signed.", extractions: [
    { extraction_class: "person", extraction_text: "Alice", attributes: {} },
  ] }],
};

function Harness({ initialRecipe = sampleRecipe }: { initialRecipe?: Recipe | null }) {
  const [recipe, setRecipe] = useState<Recipe | null>(initialRecipe);
  const [goal, setGoal] = useState("Find people");
  const [error, setError] = useState<string | null>(null);
  return <>
    <output hidden aria-label="Applied recipe">{JSON.stringify(recipe)}</output>
    {error && <p role="alert">{error}</p>}
    <RecipePanel recipe={recipe} goal={goal} setGoal={setGoal} busy={false}
      generating={false} canGenerate={false} onGenerate={() => {}}
      onApply={setRecipe} onError={setError} />
  </>;
}

it("shows annotations only in Few-shot JSON while retaining the complete source separately", () => {
  render(<Harness />);
  fireEvent.click(screen.getByRole("tab", { name: "Few-shot JSON" }));
  expect(JSON.parse((screen.getByLabelText("Few-shot extraction JSON") as HTMLTextAreaElement).value))
    .toEqual([{ extractions: [{ extraction_class: "person", extraction_text: "Alice", attributes: {} }] }]);
  fireEvent.click(screen.getByRole("tab", { name: "Example text" }));
  expect(screen.getByText("Alice signed.")).toBeVisible();
});

it("applies annotation-only edits against each unchanged source and preserves its provenance", () => {
  const second = { text: "Bob approved.", extractions: [
    { extraction_class: "person", extraction_text: "Bob", attributes: {} },
  ] };
  const original = { ...sourcedRecipe, examples: [sourcedRecipe.examples[0], second] };
  render(<Harness initialRecipe={original} />);
  fireEvent.click(screen.getByRole("tab", { name: "Few-shot JSON" }));
  fireEvent.change(screen.getByLabelText("Few-shot extraction JSON"), { target: { value: JSON.stringify([
    { extractions: [{ extraction_class: "person", extraction_text: "Alice", attributes: { role: "signer" } }] },
    { extractions: [{ extraction_class: "action", extraction_text: "approved", attributes: { owner: "Bob" } }] },
  ]) } });
  fireEvent.click(screen.getByRole("button", { name: "Apply edits" }));
  const applied = JSON.parse(screen.getByLabelText("Applied recipe").textContent!);
  expect(applied.version).toBe(2);
  expect(applied.examples[0].text).toBe("Alice signed.");
  expect(applied.examples[0].source).toEqual(source);
  expect(applied.examples[1].text).toBe("Bob approved.");
  expect(applied.examples[1].extractions[0].attributes).toEqual({ owner: "Bob" });
});

it.each([
  ["Instructions", "Generic extraction instructions"],
  ["Few-shot JSON", "Few-shot extraction JSON"],
  ["Example text", null],
])("opens %s in the full-page recipe view", (tab, editorLabel) => {
  render(<Harness />);
  fireEvent.click(screen.getByRole("button", { name: `Expand ${tab}` }));
  const dialog = screen.getByRole("dialog", { name: "Active recipe" });
  expect(within(dialog).getByRole("tab", { name: tab })).toHaveAttribute("aria-selected", "true");
  if (editorLabel) expect(within(dialog).getByLabelText(editorLabel)).toBeVisible();
  else expect(within(dialog).getByText("Alice signed.")).toBeVisible();
  expect(within(dialog).getByRole("button", { name: "Close full-page view" })).toBeVisible();
});

it("preserves both unapplied editor drafts after closing and reopening", () => {
  render(<Harness />);
  const instructions = "Extract people and their responsibilities.";
  const examples = JSON.stringify([{ extractions: [
    { extraction_class: "person", extraction_text: "Alice", attributes: { role: "signer" } },
  ] }]);
  fireEvent.click(screen.getByRole("button", { name: "Expand Instructions" }));
  let dialog = screen.getByRole("dialog", { name: "Active recipe" });
  fireEvent.change(within(dialog).getByLabelText("Generic extraction instructions"), { target: { value: instructions } });
  fireEvent.click(within(dialog).getByRole("tab", { name: "Few-shot JSON" }));
  fireEvent.change(within(dialog).getByLabelText("Few-shot extraction JSON"), { target: { value: examples } });
  fireEvent.click(within(dialog).getByRole("button", { name: "Close full-page view" }));
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(screen.getByLabelText("Few-shot extraction JSON")).toHaveValue(examples);
  expect(screen.getByText("v1 · from sample.txt")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Expand Instructions" }));
  dialog = screen.getByRole("dialog", { name: "Active recipe" });
  expect(within(dialog).getByLabelText("Generic extraction instructions")).toHaveValue(instructions);
  expect(within(dialog).getByLabelText("Few-shot extraction JSON")).toHaveValue(examples);
});

it("closes on Escape-style cancellation, retaining the draft and restoring focus", () => {
  render(<Harness />);
  const trigger = screen.getByRole("button", { name: "Expand Instructions" });
  fireEvent.click(trigger);
  const dialog = screen.getByRole("dialog", { name: "Active recipe" });
  fireEvent.change(within(dialog).getByLabelText("Generic extraction instructions"), {
    target: { value: "Extract named people in order." },
  });
  fireEvent(dialog, new Event("cancel", { cancelable: true }));
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(screen.getByLabelText("Generic extraction instructions")).toHaveValue("Extract named people in order.");
  expect(screen.getByRole("button", { name: "Expand Instructions" })).toHaveFocus();
});

it("applies a validated full-page edit without closing the editor", () => {
  render(<Harness />);
  fireEvent.click(screen.getByRole("button", { name: "Expand Instructions" }));
  const dialog = screen.getByRole("dialog", { name: "Active recipe" });
  fireEvent.change(within(dialog).getByLabelText("Generic extraction instructions"), {
    target: { value: "Extract named people in order." },
  });
  fireEvent.click(within(dialog).getByRole("button", { name: "Apply edits" }));
  expect(screen.getByText("v2 · from sample.txt")).toBeInTheDocument();
  expect(screen.getByRole("dialog", { name: "Active recipe" })).toBeVisible();
  expect(within(dialog).getByRole("button", { name: "Apply edits" })).toBeDisabled();
});

it("shows invalid JSON inside the dialog and preserves the active recipe", () => {
  render(<Harness />);
  fireEvent.click(screen.getByRole("button", { name: "Expand Few-shot JSON" }));
  const dialog = screen.getByRole("dialog", { name: "Active recipe" });
  fireEvent.change(within(dialog).getByLabelText("Few-shot extraction JSON"), { target: { value: "not JSON" } });
  fireEvent.click(within(dialog).getByRole("button", { name: "Apply edits" }));
  expect(within(dialog).getByRole("alert")).toHaveTextContent(/JSON|Unexpected|token/i);
  expect(screen.getByText("v1 · from sample.txt")).toBeInTheDocument();
  expect(within(dialog).getByLabelText("Few-shot extraction JSON")).toHaveValue("not JSON");
});

it("disables expansion until a recipe exists", () => {
  render(<Harness initialRecipe={null} />);
  for (const tab of ["Instructions", "Few-shot JSON", "Example text"]) {
    expect(screen.getByRole("button", { name: `Expand ${tab}` })).toBeDisabled();
  }
});

const source = {
  name: "contract.txt", document_id: "doc-1", document_sha256: "a".repeat(64),
  start: 24, end: 37, text_sha256: "b".repeat(64), verified: true,
};

const sourcedRecipe = { ...sampleRecipe, examples: [{ ...sampleRecipe.examples[0], source }] };

it("shows real excerpt provenance and distinguishes origin from annotation accuracy", () => {
  render(<Harness initialRecipe={sourcedRecipe} />);
  fireEvent.click(screen.getByRole("tab", { name: "Example text" }));
  expect(screen.getByText("contract.txt")).toBeVisible();
  expect(screen.getByText(/24–37/)).toBeVisible();
  expect(screen.getByText(/Verified source text/)).toBeVisible();
  expect(screen.getByText(/not.*accuracy/i)).toBeVisible();
});

it("labels legacy examples as unverified, not synthetic or certified", () => {
  render(<Harness />);
  fireEvent.click(screen.getByRole("tab", { name: "Example text" }));
  expect(screen.getByText(/Legacy.*unverified/i)).toBeVisible();
  expect(screen.queryByText(/Synthetic examples/)).not.toBeInTheDocument();
});

it.each(["text", "source", "new"])("rejects %s source changes in the annotation-only editor", mode => {
  render(<Harness initialRecipe={sourcedRecipe} />);
  const example = { ...sourcedRecipe.examples[0], source: { ...source } };
  if (mode === "text") example.text = "Alice signed. Again.";
  if (mode === "source") example.source.start = 23;
  const next = mode === "new" ? [example, { ...example, text: "Alice signed. Again." }] : [example];
  fireEvent.click(screen.getByRole("tab", { name: "Few-shot JSON" }));
  fireEvent.change(screen.getByLabelText("Few-shot extraction JSON"), { target: { value: JSON.stringify(next) } });
  fireEvent.click(screen.getByRole("button", { name: "Apply edits" }));
  expect(screen.getByText("v1 · from sample.txt")).toBeInTheDocument();
  expect(JSON.parse(screen.getByLabelText("Applied recipe").textContent!).examples)
    .toEqual(sourcedRecipe.examples);
  expect(screen.getAllByRole("alert")[0]).toHaveTextContent(/unrecognized|exactly|example/i);
});

it("retains verified provenance when only instructions change", () => {
  render(<Harness initialRecipe={sourcedRecipe} />);
  fireEvent.change(screen.getByLabelText("Generic extraction instructions"), { target: { value: "Extract only named people." } });
  fireEvent.click(screen.getByRole("button", { name: "Apply edits" }));
  expect(JSON.parse(screen.getByLabelText("Applied recipe").textContent!).examples[0].source.verified).toBe(true);
});

it("downgrades all imported provenance and preserves its historical metadata", async () => {
  render(<Harness />);
  const file = new File([JSON.stringify(sourcedRecipe)], "recipe.json", { type: "application/json" });
  Object.defineProperty(file, "text", { value: async () => JSON.stringify(sourcedRecipe) });
  fireEvent.change(screen.getByLabelText("Recipe JSON file"), { target: { files: [file] } });
  await waitFor(() => expect(JSON.parse(screen.getByLabelText("Applied recipe").textContent!).examples[0].source)
    .toEqual({ ...source, verified: false }));
});

it("does not let a manual verification flag certify previously unverified provenance", () => {
  render(<Harness initialRecipe={{ ...sourcedRecipe,
    examples: [{ ...sourcedRecipe.examples[0], source: { ...source, verified: false } }] }} />);
  fireEvent.click(screen.getByRole("tab", { name: "Few-shot JSON" }));
  fireEvent.change(screen.getByLabelText("Few-shot extraction JSON"), { target: { value: JSON.stringify(sourcedRecipe.examples) } });
  fireEvent.click(screen.getByRole("button", { name: "Apply edits" }));
  expect(JSON.parse(screen.getByLabelText("Applied recipe").textContent!).examples[0].source.verified).toBe(false);
  expect(screen.getByText("v1 · from sample.txt")).toBeInTheDocument();
});

it.each([
  [],
  [{ extractions: [] }],
  [{ extractions: [{ extraction_class: "person", extraction_text: "Bob", attributes: {} }] }],
  [{ extractions: [{ extraction_class: "person", extraction_text: "Alice" }] },
    { extractions: [{ extraction_class: "person", extraction_text: "Alice" }] }],
].map(draft => ({ draft })))("rejects invalid annotation-only drafts without changing the active sources: %j", ({ draft }) => {
  render(<Harness />);
  fireEvent.click(screen.getByRole("tab", { name: "Few-shot JSON" }));
  fireEvent.change(screen.getByLabelText("Few-shot extraction JSON"), { target: { value: JSON.stringify(draft) } });
  fireEvent.click(screen.getByRole("button", { name: "Apply edits" }));
  expect(JSON.parse(screen.getByLabelText("Applied recipe").textContent!)).toEqual(sampleRecipe);
  expect(screen.getAllByRole("alert")).not.toHaveLength(0);
});

it("returns focus to the validation error after repeated invalid full-page Apply attempts", () => {
  render(<Harness />);
  fireEvent.click(screen.getByRole("button", { name: "Expand Few-shot JSON" }));
  const dialog = screen.getByRole("dialog", { name: "Active recipe" });
  fireEvent.change(within(dialog).getByLabelText("Few-shot extraction JSON"), { target: { value: "not JSON" } });
  const apply = within(dialog).getByRole("button", { name: "Apply edits" });
  fireEvent.click(apply);
  expect(within(dialog).getByRole("alert")).toHaveFocus();
  apply.focus();
  fireEvent.click(apply);
  expect(within(dialog).getByRole("alert")).toHaveFocus();
});
