import { beforeEach, expect, it } from "vitest";
import { parseRecipe, saveRecipe, loadRecipe, RECIPE_STORAGE_KEY } from "./recipe";

export const sampleRecipe = {
  id: "recipe-1", version: 1, goal: "Find people",
  instructions: "Extract people using exact source text.",
  source_name: "a.txt", created_at: "2026-10-12T00:00:00Z",
  source_characters_used: 13, source_characters_total: 13,
  examples: [{ text: "Alice signed.", extractions: [
    { extraction_class: "person", extraction_text: "Alice", attributes: { role: "signer" } },
  ] }],
};

beforeEach(() => localStorage.clear());

it("round trips legacy recipes without adding document text or credentials", () => {
  const recipe = parseRecipe(JSON.stringify(sampleRecipe));
  saveRecipe(recipe);
  expect(loadRecipe()).toEqual(sampleRecipe);
  expect(Object.keys(localStorage)).toEqual([RECIPE_STORAGE_KEY]);
});

const source = {
  name: "a.txt", document_id: "document-a", document_sha256: "a".repeat(64),
  start: 5, end: 18, text_sha256: "b".repeat(64), verified: true,
};

it("round trips nullable and source-backed provenance without certifying legacy examples", () => {
  const recipe = parseRecipe(JSON.stringify({
    ...sampleRecipe, examples: [{ ...sampleRecipe.examples[0], source }],
  }));
  saveRecipe(recipe);
  expect(loadRecipe()?.examples[0].source).toEqual(source);
  expect(parseRecipe(JSON.stringify(sampleRecipe)).examples[0].source).toBeUndefined();
  expect(parseRecipe(JSON.stringify({
    ...sampleRecipe, examples: [{ ...sampleRecipe.examples[0], source: null }],
  })).examples[0].source).toBeNull();
});

it.each([
  { start: -1 }, { end: 5 }, { end: 4 }, { end: 6.5 }, { verified: "true" },
  { document_sha256: "A".repeat(64) }, { text_sha256: "abc" }, { extra: "unknown" },
])("rejects malformed provenance %j rather than dropping it", invalid => {
  expect(() => parseRecipe(JSON.stringify({
    ...sampleRecipe, examples: [{ ...sampleRecipe.examples[0], source: { ...source, ...invalid } }],
  }))).toThrow();
});

it("accepts source excerpts at the Unicode character limit without altering CRLF or emoji", () => {
  const text = "😀".repeat(11_998) + "\r\n";
  const recipe = parseRecipe(JSON.stringify({
    ...sampleRecipe, examples: [{ text, source: { ...source, start: 7, end: 12_007 },
      extractions: [{ extraction_class: "symbol", extraction_text: "😀", attributes: {} }] }],
  }));
  expect(recipe.examples[0].text).toBe(text);
  expect(recipe.examples[0].source?.end).toBe(12_007);
});

it("accepts a 12,000-code-point extraction without treating surrogate pairs as extra characters", () => {
  const text = "😀".repeat(12_000);
  const recipe = parseRecipe(JSON.stringify({
    ...sampleRecipe, examples: [{ text, source: { ...source, start: 0, end: 12_000 },
      extractions: [{ extraction_class: "symbol", extraction_text: text, attributes: {} }] }],
  }));
  expect(recipe.examples[0].extractions[0].extraction_text).toBe(text);
});

it.each([["text", 500_000], ["extraction_text", 12_000]] as const)(
  "rejects %s beyond its separate Unicode limit", (field, limit) => {
  const text = "😀".repeat(limit + 1);
  expect(() => parseRecipe(JSON.stringify({
    ...sampleRecipe, examples: [{
      text, extractions: [{ extraction_class: "symbol", extraction_text: field === "text" ? "😀" : text, attributes: {} }],
    }],
  }))).toThrow();
});

it("persists complete long examples with selective annotations and unchanged source metadata", () => {
  const text = "Alice signed.\n" + "😀".repeat(20_000) + "\r\nUnlabelled document tail.";
  const whole = parseRecipe(JSON.stringify({
    ...sampleRecipe, examples: [{ ...sampleRecipe.examples[0], text,
      source: { ...source, start: 0, end: Array.from(text).length } }],
  }));
  saveRecipe(whole);
  expect(loadRecipe()?.examples[0]).toEqual(whole.examples[0]);
  const imported = parseRecipe(JSON.stringify(whole), "import");
  expect(imported.examples[0].text).toBe(text);
  expect(imported.examples[0].source?.verified).toBe(false);
  const edited = parseRecipe(JSON.stringify({ ...whole, instructions: "Extract all matching people." }), whole);
  expect(edited.examples[0].source?.verified).toBe(true);
});

it("accepts full-document examples at the 500,000-code-point limit", () => {
  const text = "😀".repeat(500_000);
  const whole = parseRecipe(JSON.stringify({
    ...sampleRecipe, examples: [{ text, source: { ...source, start: 0, end: 500_000 },
      extractions: [{ extraction_class: "symbol", extraction_text: "😀", attributes: {} }] }],
  }));
  expect(whole.examples[0].text).toBe(text);
});

it.each([
  { start: 0, end: 7, text: "😀Alice" },
  { start: 7, end: 13, text: "😀Alice\r\n" },
])("rejects verified metadata with an incorrect code-point interval %j", ({ start, end, text }) => {
  expect(() => parseRecipe(JSON.stringify({
    ...sampleRecipe, examples: [{ text, source: { ...source, start, end },
      extractions: [{ extraction_class: "person", extraction_text: "Alice", attributes: {} }] }],
  }))).toThrow(/interval|span|length/i);
});

it("accepts historical unverified intervals after text changes without silently correcting metadata", () => {
  const recipe = parseRecipe(JSON.stringify({
    ...sampleRecipe, examples: [{ ...sampleRecipe.examples[0], text: "Alice signed. Again.",
      source: { ...source, verified: false } }],
  }));
  expect(recipe.examples[0].source).toEqual({ ...source, verified: false });
});

it("rejects invented example text", () => {
  const invalid = structuredClone(sampleRecipe);
  invalid.examples[0].extractions[0].extraction_text = "Bob";
  expect(() => parseRecipe(JSON.stringify(invalid))).toThrow(/exact/i);
});

it("rejects empty examples and zero recipe versions", () => {
  expect(() => parseRecipe(JSON.stringify({ ...sampleRecipe, examples: [] }))).toThrow();
  expect(() => parseRecipe(JSON.stringify({ ...sampleRecipe, version: 0 }))).toThrow();
});

it("reports corrupted persisted recipes instead of silently dropping them", () => {
  localStorage.setItem(RECIPE_STORAGE_KEY, "not json");
  expect(() => loadRecipe()).toThrow();
});
