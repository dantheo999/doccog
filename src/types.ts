import type { z } from "zod";
import type { recipeSchema } from "./recipe";
import type { ExtractionOptions } from "./extractionOptions";
import type { optimizationProposalSchema } from "./optimization";

export type Recipe = z.infer<typeof recipeSchema>;
export type OptimizationProposal = z.infer<typeof optimizationProposalSchema>;
export interface LocalDocument { file: File; path: string }
export interface DocumentPreview {
  id: string;
  name: string;
  text: string;
  characters: number;
  pages: number | null;
}
export interface Extraction {
  id: string;
  extraction_class: string;
  extraction_text: string;
  attributes: Record<string, unknown>;
  start: number | null;
  end: number | null;
  alignment_status: string | null;
  grounding: "exact" | "approximate" | "ungrounded";
}
export interface ExtractionResult {
  document_id: string;
  text: string;
  recipe_id: string;
  recipe_version: number;
  extractions: Extraction[];
  elapsed_seconds: number;
  options: ExtractionOptions;
}
export interface ModelConfig {
  configured: boolean;
  deployment: string;
  endpoint: string;
  error: string | null;
  max_upload_mib: number;
  max_characters: number;
}
