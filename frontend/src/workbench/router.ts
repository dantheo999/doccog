import { useEffect, useState } from "react";

/** Minimal hash router: #/apps/llm?run=abc -> { path: "/apps/llm", query: { run: "abc" } }. */
export interface Route { path: string; query: Record<string, string> }

export function parseHash(hash: string): Route {
  const raw = hash.replace(/^#/, "") || "/";
  const [path, search = ""] = raw.split("?");
  return { path: path || "/", query: Object.fromEntries(new URLSearchParams(search)) };
}

export function href(path: string, query: Record<string, string | undefined> = {}): string {
  const search = new URLSearchParams(Object.entries(query).filter((entry): entry is [string, string] => !!entry[1]));
  return `#${path}${search.size ? `?${search}` : ""}`;
}

export function navigate(path: string, query: Record<string, string | undefined> = {}) {
  window.location.hash = href(path, query);
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parseHash(window.location.hash));
  useEffect(() => {
    const update = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", update);
    return () => window.removeEventListener("hashchange", update);
  }, []);
  return route;
}
