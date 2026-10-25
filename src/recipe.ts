import { z } from "zod";
import type { Recipe } from "./types";

const extractionTextSchema = z.string().min(1).refine(text => Array.from(text).length <= 12_000, {
  message: "Text must contain at most 12,000 Unicode code points.",
});

const extractionSchema = z.object({
  extraction_class: z.string().trim().min(1).max(100),
  extraction_text: extractionTextSchema,
  attributes: z.record(z.string(), z.json()).refine(value => Object.keys(value).length <= 30)
    .default({}),
}).strict();

const sourceSchema = z.object({
  name: z.string().min(1).max(1_024),
  document_id: z.string().min(1).max(100),
  document_sha256: z.string().regex(/^[a-f0-9]{64}$/),
  start: z.number().int().min(0),
  end: z.number().int().min(1),
  text_sha256: z.string().regex(/^[a-f0-9]{64}$/),
  verified: z.boolean(),
}).strict().refine(source => source.end > source.start, {
  path: ["end"], message: "Source end must be greater than start.",
});

const exampleSchema = z.object({
  text: z.string().min(1).refine(text => Array.from(text).length <= 500_000, {
    message: "Example text must contain at most 500,000 Unicode code points.",
  }),
  extractions: z.array(extractionSchema).min(1).max(100),
  source: sourceSchema.nullable().optional(),
}).strict().superRefine((example, context) => {
  example.extractions.forEach((extraction, index) => {
    if (!example.text.includes(extraction.extraction_text)) {
      context.addIssue({
        code: "custom", path: ["extractions", index, "extraction_text"],
        message: "Extraction text must be an exact substring of its example text.",
      });
    }
  });
});

const recipeStructureSchema = z.object({
  id: z.string().trim().min(1).max(100),
  version: z.number().int().min(1),
  instructions: z.string().trim().min(1).max(16_000),
  goal: z.string().trim().min(1).max(4_000),
  examples: z.array(exampleSchema).min(1).max(5),
  source_name: z.string().max(1_024),
  created_at: z.string().max(100),
  source_characters_used: z.number().int().min(0).default(0),
  source_characters_total: z.number().int().min(0).default(0),
}).strict();

export const recipeSchema = recipeStructureSchema.superRefine((recipe, context) => {
  recipe.examples.forEach((example, index) => {
    if (example.source?.verified && example.source.end - example.source.start !== Array.from(example.text).length) {
      context.addIssue({ code: "custom", path: ["examples", index, "source", "end"],
        message: "Verified source interval length must match the example's Unicode code-point length." });
    }
  });
});

export const RECIPE_STORAGE_KEY = "doc-cognition.recipe.v1";

function parseRecipeValue(value: unknown, schema = recipeSchema): Recipe {
  const result = schema.safeParse(value);
  if (!result.success) {
    throw new Error(result.error.issues.map(issue =>
      `${issue.path.join(".") || "Recipe"}: ${issue.message}`).join("\n"));
  }
  return result.data;
}

export function parseRecipe(json: string, provenancePolicy?: Recipe | "import"): Recipe {
  let parsed: unknown = JSON.parse(json);
  if (provenancePolicy) {
    // Downgrade edited/imported claims before validating verified interval lengths.
    parsed = downgradeProvenance(parseRecipeValue(parsed, recipeStructureSchema),
      provenancePolicy === "import" ? undefined : provenancePolicy);
  }
  return parseRecipeValue(parsed);
}

export function formatFewShotExamples(examples: Recipe["examples"]): string {
  return JSON.stringify(examples.map(({ extractions }) => ({ extractions })), null, 2);
}

export function parseFewShotExamples(json: string, originals: Recipe["examples"]): Recipe["examples"] {
  const result = z.array(z.object({
    extractions: z.array(extractionSchema).min(1).max(100),
  }).strict()).length(originals.length).safeParse(JSON.parse(json));
  if (!result.success) {
    throw new Error(result.error.issues.map(issue =>
      `Examples${issue.path.length ? `.${issue.path.join(".")}` : ""}: ${issue.message}`).join("\n"));
  }
  return result.data.map((item, index) => ({ ...originals[index], extractions: item.extractions }));
}

export function loadRecipe(): Recipe | null {
  const stored = localStorage.getItem(RECIPE_STORAGE_KEY);
  return stored === null ? null : parseRecipe(stored);
}

export function saveRecipe(recipe: Recipe) {
  localStorage.setItem(RECIPE_STORAGE_KEY, JSON.stringify(recipe));
}

export function downgradeProvenance(next: Recipe, original?: Recipe): Recipe {
  return { ...next, examples: next.examples.map(example => {
    if (!example.source) return example;
    const unchanged = original?.examples.some(previous => previous.text === example.text
      && JSON.stringify(previous.source) === JSON.stringify(example.source));
    return { ...example, source: { ...example.source, verified: example.source.verified && !!unchanged } };
  }) };
}

export function downloadJson(value: unknown, name: string) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
