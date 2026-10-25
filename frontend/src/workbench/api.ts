import type {
  Collection, Comparison, ExtractionSchema, Run, RunSummary, SchemaDraft, SchemaSummary, SolutionInfo,
  StoredDocument, Usage,
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api${path}`, init);
  } catch {
    throw new Error("Cannot reach the local API. Check that the backend is running.");
  }
  if (!response.ok) {
    let message = `API request failed (${response.status}).`;
    if ((response.headers.get("content-type") ?? "").includes("application/json")) {
      const body: { detail?: unknown } = await response.json();
      if (typeof body.detail === "string") message = body.detail;
      else if (Array.isArray(body.detail)) {
        message = body.detail.map((item: { loc?: unknown[]; msg?: string }) =>
          `${(item.loc ?? []).slice(1).join(".") || "request"}: ${item.msg ?? "invalid"}`).join("\n");
      }
    } else if (response.status === 413) message = "Document exceeds the 20 MiB upload limit.";
    else if (response.status === 504) message = "The request timed out at the proxy.";
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

const json = (method: string, body: unknown): RequestInit => ({
  method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
});

export const workbench = {
  solutions: () => request<SolutionInfo[]>("/solutions"),
  solution: (id: string) => request<SolutionInfo>(`/solutions/${id}`),
  preview: (id: string, schemaId: string, settings: Record<string, unknown>) =>
    request<Record<string, string>>(`/solutions/${id}/preview`, json("POST", { schema_id: schemaId, settings })),
  prepare: (id: string, documentId: string, settings: Record<string, unknown>) =>
    request<{ document_id: string; notes: string[] }>(`/solutions/${id}/prepare`,
      json("POST", { document_id: documentId, settings })),

  createCollection: (name: string) => request<Collection>("/collections", json("POST", { name })),
  collection: (id: string) => request<Collection>(`/collections/${id}`),
  collections: () => request<(Omit<Collection, "documents"> & { documents: number })[]>("/collections"),
  uploadToCollection: (file: File, collectionId: string, path: string) => {
    const body = new FormData();
    body.append("file", file);
    body.append("collection_id", collectionId);
    body.append("path", path);
    return request<StoredDocument>("/documents", { method: "POST", body });
  },
  document: (id: string) => request<StoredDocument>(`/documents/${id}`),

  schemas: () => request<SchemaSummary[]>("/schemas"),
  schema: (id: string, version?: number) =>
    request<ExtractionSchema>(`/schemas/${id}${version ? `?version=${version}` : ""}`),
  starterSchema: () => request<SchemaDraft>("/schemas/starter"),
  saveSchema: (draft: SchemaDraft, schemaId: string | null, exemplars?: string[]) =>
    request<ExtractionSchema>("/schemas", json("POST", { draft, schema_id: schemaId, exemplars })),
  suggestSchema: (documentIds: string[], guidance: string, reasoningEffort: string) =>
    request<{ draft: SchemaDraft; notes: string[]; usage: Usage; exemplars: string[] }>("/schemas/suggest",
      json("POST", { document_ids: documentIds, guidance, reasoning_effort: reasoningEffort })),

  runs: () => request<RunSummary[]>("/runs"),
  run: (id: string) => request<Run>(`/runs/${id}`),
  startRun: (body: { solution: string; label: string; schema_id: string; settings: Record<string, unknown>; document_ids: string[] }) =>
    request<RunSummary>("/runs", json("POST", body)),
  importRun: (body: { solution: string; label: string; model: string; schema_id: string;
    records: { document_id: string; data: Record<string, unknown> | null; error?: string | null; raw?: unknown }[] }) =>
    request<RunSummary>("/runs/import", json("POST", body)),
  importUnstract: (body: { schema_id: string; label: string; document_ids: string[]; files: { name: string; content: unknown }[] }) =>
    request<RunSummary>("/solutions/unstract/import", json("POST", body)),
  cancelRun: (id: string) => request<{ cancelled: boolean }>(`/runs/${id}/cancel`, { method: "POST" }),
  deleteRun: (id: string) => request<{ deleted: string }>(`/runs/${id}`, { method: "DELETE" }),
  compare: (runIds: string[], referenceRunId: string | null) =>
    request<Comparison>("/compare", json("POST", { run_ids: runIds, reference_run_id: referenceRunId })),
};

export function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
