import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { message, workbench } from "./api";
import type { Collection, ExtractionSchema } from "./types";

const STORAGE_KEY = "doc-cognition.workspace.v1";

interface Stored { collectionId: string | null; schemaId: string | null }

function load(): Stored {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "{}");
    return { collectionId: typeof value.collectionId === "string" ? value.collectionId : null,
      schemaId: typeof value.schemaId === "string" ? value.schemaId : null };
  } catch { return { collectionId: null, schemaId: null }; }
}

function persist(value: Stored) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(value)); } catch { /* storage unavailable */ }
}

export interface Workspace {
  collection: Collection | null;
  schema: ExtractionSchema | null;
  loading: boolean;
  error: string | null;
  setCollection: (collection: Collection | null) => void;
  setSchema: (schema: ExtractionSchema | null) => void;
  refresh: () => Promise<void>;
}

const WorkspaceContext = createContext<Workspace | null>(null);

export function WorkspaceProvider({ children }: { children: ReactNode }) {
  const [stored, setStored] = useState<Stored>(load);
  const [collection, setCollectionState] = useState<Collection | null>(null);
  const [schema, setSchemaState] = useState<ExtractionSchema | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [nextCollection, nextSchema] = await Promise.all([
        stored.collectionId ? workbench.collection(stored.collectionId).catch(() => null) : Promise.resolve(null),
        stored.schemaId ? workbench.schema(stored.schemaId).catch(() => null) : Promise.resolve(null),
      ]);
      setCollectionState(nextCollection);
      setSchemaState(nextSchema);
    } catch (failure) { setError(message(failure)); }
    finally { setLoading(false); }
  }, [stored.collectionId, stored.schemaId]);

  useEffect(() => { void refresh(); }, [refresh]);

  const value = useMemo<Workspace>(() => ({
    collection, schema, loading, error, refresh,
    setCollection: next => {
      setCollectionState(next);
      setStored(current => { const updated = { ...current, collectionId: next?.id ?? null }; persist(updated); return updated; });
    },
    setSchema: next => {
      setSchemaState(next);
      setStored(current => { const updated = { ...current, schemaId: next?.id ?? null }; persist(updated); return updated; });
    },
  }), [collection, schema, loading, error, refresh]);

  return <WorkspaceContext.Provider value={value}>{children}</WorkspaceContext.Provider>;
}

export function useWorkspace(): Workspace {
  const value = useContext(WorkspaceContext);
  if (!value) throw new Error("useWorkspace must be used inside WorkspaceProvider");
  return value;
}

/** Workspace reader that tolerates rendering outside the provider (standalone studio tests). */
export function useOptionalWorkspace(): Workspace | null {
  return useContext(WorkspaceContext);
}
