import { parseRecipe } from "./recipe";
import type { DocumentPreview, ExtractionResult, ModelConfig, Recipe } from "./types";
import type { ExtractionOptions } from "./extractionOptions";
import { assertOptimizationBase, parseOptimizationProposal } from "./optimization";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api${path}`, init);
  } catch {
    throw new Error("Cannot reach the local API. Check that the Docker containers are running.");
  }
  if (!response.ok) {
    let message = `API request failed (${response.status}).`;
    const contentType = response.headers.get("content-type") ?? "";
    if (contentType.includes("application/json")) {
      const body: { detail?: unknown } = await response.json();
      if (typeof body.detail === "string") message = body.detail;
      else if (Array.isArray(body.detail)) message = "The submitted recipe or request is invalid. Check the examples and instructions.";
    } else if (response.status === 504) {
      message = "Processing timed out. Try a smaller document or a more focused extraction goal.";
    } else if (response.status === 413) {
      message = "Document exceeds the 20 MiB upload limit.";
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export const api = {
  config: () => request<ModelConfig>("/config"),
  upload: (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<DocumentPreview>("/documents", { method: "POST", body });
  },
  generate: async (document_ids: string[], goal: string, schema_id?: string): Promise<Recipe> => {
    const recipe = await request<unknown>("/recipes/generate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(schema_id ? { document_ids, goal, schema_id } : { document_ids, goal }),
    });
    return parseRecipe(JSON.stringify(recipe));
  },
  extract: (document_id: string, recipe: Recipe, options: ExtractionOptions, schema_id?: string) => request<ExtractionResult>("/extractions", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(schema_id ? { document_id, recipe, options, schema_id } : { document_id, recipe, options }),
  }),
  optimize: async (document_id: string, recipe: Recipe, result: ExtractionResult, schema_id?: string) => {
    const proposal = parseOptimizationProposal(await request<unknown>("/recipes/optimize", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(schema_id ? { document_id, recipe, result, schema_id } : { document_id, recipe, result }),
    }));
    assertOptimizationBase(proposal, recipe);
    return proposal;
  },
};
