import { useCallback, useEffect, useState } from "react";
import { FileStack, GitCompareArrows, House, ScanText, Shapes } from "lucide-react";
import { message, workbench } from "./workbench/api";
import { href, useRoute } from "./workbench/router";
import type { SolutionInfo } from "./workbench/types";
import { WorkspaceProvider, useWorkspace } from "./workbench/workspace";
import LandingPage from "./workbench/pages/LandingPage";
import WorkspacePage from "./workbench/pages/WorkspacePage";
import SolutionPage from "./workbench/pages/SolutionPage";
import ComparePage from "./workbench/pages/ComparePage";
import "./workbench/workbench.css";

export default function App() {
  return <WorkspaceProvider><Shell /></WorkspaceProvider>;
}

function Shell() {
  const route = useRoute();
  const workspace = useWorkspace();
  const [solutions, setSolutions] = useState<SolutionInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadSolutions = useCallback(() => {
    workbench.solutions().then(items => { setSolutions(items); setError(null); })
      .catch(failure => setError(message(failure)));
  }, []);
  useEffect(() => { loadSolutions(); }, [loadSolutions]);
  useEffect(() => { if (route.path === "/") loadSolutions(); }, [route.path, loadSolutions]);

  const appId = route.path.startsWith("/apps/") ? route.path.slice("/apps/".length) : null;
  const solution = appId ? solutions?.find(item => item.id === appId) : undefined;
  const nav = [
    { path: "/", label: "Home", icon: <House size={15} /> },
    { path: "/workspace", label: "Workspace", icon: <FileStack size={15} /> },
    { path: "/compare", label: "Compare", icon: <GitCompareArrows size={15} /> },
  ];

  let page;
  if (route.path === "/") page = <LandingPage solutions={solutions} error={error} onRefresh={loadSolutions} />;
  else if (route.path === "/workspace") page = <WorkspacePage />;
  else if (route.path === "/compare") page = <ComparePage />;
  else if (appId && solution) page = <SolutionPage key={solution.id} solution={solution} onRefresh={loadSolutions} />;
  else if (appId && !solutions && !error) page = <div className="wb-page"><p className="microcopy">Loading…</p></div>;
  else page = <div className="wb-page"><div className="wb-empty"><h2>Page not found</h2>
    <p>{appId ? `No solution called “${appId}” is registered.` : "Unknown page."}</p><a className="secondary" href="#/">Back to apps</a></div></div>;

  return <div className="wb-shell">
    <header className="wb-header">
      <a className="wb-brand" href="#/"><span className="brand-mark small"><ScanText size={19} /></span>
        <span><strong>Document Cognition</strong><small>Extraction workbench</small></span></a>
      <nav className="wb-nav" aria-label="Main">
        {nav.map(item => <a key={item.path} href={href(item.path)}
          className={route.path === item.path ? "active" : ""} aria-current={route.path === item.path ? "page" : undefined}>
          {item.icon}{item.label}</a>)}
        {solutions?.map(item => <a key={item.id} href={href(item.route)}
          className={route.path === item.route ? "active" : ""} aria-current={route.path === item.route ? "page" : undefined}>
          <Shapes size={14} style={{ color: item.accent }} />{item.name}</a>)}
      </nav>
      <a className="wb-context" href={href("/workspace")} title="Active document set and agreed schema">
        <span><b>{workspace.collection ? workspace.collection.documents.length : 0}</b> docs
          {workspace.collection && <i> · {workspace.collection.name}</i>}</span>
        <span>{workspace.schema ? <>schema <b>{workspace.schema.name}</b> v{workspace.schema.version}</> : "no agreed schema"}</span>
      </a>
    </header>
    <main className="wb-main">{page}</main>
  </div>;
}
