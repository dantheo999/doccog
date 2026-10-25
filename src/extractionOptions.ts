import { z } from "zod";

export const EXTRACTION_OPTION_FIELDS = [
  { key: "chunk_size", label: "Target chunk size (characters)", min: 500, max: 20000,
    help: "Smaller chunks reduce source text per call but increase request count, repeated recipe tokens and potential context loss." },
  { key: "max_workers", label: "Maximum concurrent LLM requests", min: 1, max: 16,
    help: "Limits parallel chunk requests, not total requests. More concurrency may help until Azure request or token quotas are reached." },
  { key: "extraction_passes", label: "Extraction passes", min: 1, max: 5,
    help: "Sequential passes can improve coverage but increase processing time and model costs. Accuracy is not guaranteed." },
  { key: "context_window_chars", label: "Previous-chunk context (characters)", min: 0, max: 6000,
    help: "0 turns context off. Includes trailing source text from the preceding chunk, not its model response. Parallel calls remain possible; shorter chunks supply less context." },
] as const;

const extractionOptionsSchema = z.object({
  chunk_size: z.number().int().min(500).max(20000),
  max_workers: z.number().int().min(1).max(16),
  extraction_passes: z.number().int().min(1).max(5),
  context_window_chars: z.number().int().min(0).max(6000),
}).strict();

export type ExtractionOptions = z.infer<typeof extractionOptionsSchema>;
export const DEFAULT_EXTRACTION_OPTIONS: ExtractionOptions = {
  chunk_size: 6000, max_workers: 2, extraction_passes: 1, context_window_chars: 0,
};
export const EXTRACTION_OPTIONS_STORAGE_KEY = "doc-cognition.extraction-options.v1";

export function parseExtractionOptions(value: unknown): ExtractionOptions {
  const parsed = extractionOptionsSchema.safeParse(value);
  if (!parsed.success) {
    throw new Error(parsed.error.issues.map(issue => {
      const field = EXTRACTION_OPTION_FIELDS.find(item => item.key === issue.path[0]);
      return `${field?.label ?? "Settings"}: ${issue.message}`;
    }).join("\n"));
  }
  return parsed.data;
}

export function loadExtractionOptions(): ExtractionOptions {
  const stored = localStorage.getItem(EXTRACTION_OPTIONS_STORAGE_KEY);
  return stored === null ? { ...DEFAULT_EXTRACTION_OPTIONS } : parseExtractionOptions(JSON.parse(stored));
}

export function saveExtractionOptions(options: ExtractionOptions) {
  localStorage.setItem(EXTRACTION_OPTIONS_STORAGE_KEY, JSON.stringify(options));
}
