import { useEffect, useState } from "react";
import type { CSSProperties } from "react";
import { ArrowRight, CircleCheck, CircleDashed, Cloud, Cpu, ExternalLink, FileStack, GitCompareArrows, RefreshCw } from "lucide-react";
import { workbench } from "../api";
import { href } from "../router";
import type { RunSummary, SolutionInfo } from "../types";
import { useWorkspace } from "../workspace";

interface Props { solutions: SolutionInfo[] | null; error: string | null; onRefresh: () => void }

export default function LandingPage({ solutions, error, onRefresh }: Props) {
  const workspace = useWorkspace();
  const [runs, setRuns] = useState<RunSummary[]>([]);
  useEffect(() => { workbench.runs().then(setRuns).catch(() => setRuns([])); }, []);
  const steps = [
    { label: "Choose a document folder", done: !!workspace.collection?.documents.length, path: "/workspace" },
    { label: "Agree a schema from exemplars", done: !!workspace.schema, path: "/workspace" },
    { label: "Run one or more solutions", done: runs.length > 0, path: solutions?.[0]?.route ?? "/" },
    { label: "Compare the runs", done: false, path: "/compare" },
  ];

  return <div className="wb-page landing">
    <section className="landing-hero">
      <span className="eyebrow">PROTOTYPE · STRUCTURED EXTRACTION</span>
      <h1>Which technology extracts your documents best?</h1>
      <p>Agree one schema from exemplar documents, run each solution against the same document set, then
        compare extracted values, confidence and source grounding side by side.</p>
      <ol className="landing-steps">
        {steps.map((step, index) => <li key={step.label} className={step.done ? "done" : ""}>
          <a href={href(step.path)}>{step.done ? <CircleCheck size={15} /> : <CircleDashed size={15} />}
            <b>{index + 1}</b>{step.label}</a></li>)}
      </ol>
    </section>

    <section aria-labelledby="apps-heading">
      <div className="section-heading"><h2 id="apps-heading">Solutions</h2>
        <button className="icon-button" onClick={onRefresh} aria-label="Refresh solution status" title="Refresh status"><RefreshCw size={14} /></button></div>
      {error && <div className="notice error" role="alert"><span>{error}</span></div>}
      {!solutions && !error && <p className="microcopy">Loading solutions…</p>}
      <div className="app-grid">
        {solutions?.map(solution => <article key={solution.id} className="app-card panel"
          style={{ "--accent": solution.accent } as CSSProperties}>
          <header>
            <span className="app-runtime">{solution.runtime === "cloud" ? <Cloud size={13} /> : <Cpu size={13} />}
              {solution.runtime === "cloud" ? "Cloud model" : "Runs locally"}</span>
            <span className={`status-pill ${solution.status.ready ? "ready" : ""}`} title={solution.status.detail}>
              {solution.status.ready ? "Ready" : "Not ready"}</span>
          </header>
          <h3>{solution.name}</h3>
          <p className="app-tagline">{solution.tagline}</p>
          <p className="app-description">{solution.description}</p>
          <ul className="chip-list">{solution.capabilities.map(capability => <li key={capability}>{capability}</li>)}</ul>
          <dl className="app-facts">
            <div><dt>Technology</dt><dd>{solution.technology}</dd></div>
            <div><dt>Confidence</dt><dd>{solution.confidence_method}</dd></div>
            {!solution.status.ready && <div><dt>Status</dt><dd className="warn">{solution.status.detail}</dd></div>}
          </dl>
          <footer>
            {solution.external_url
              ? <><a className="primary" href={solution.external_url} target="_blank" rel="noreferrer">
                Open {solution.name}<ExternalLink size={14} /></a>
                <a className="secondary" href={href(solution.route)}>Prompts &amp; results<ArrowRight size={14} /></a></>
              : <a className="primary" href={href(solution.route)}>Open<ArrowRight size={14} /></a>}
          </footer>
        </article>)}
      </div>
    </section>

    <section className="tool-row" aria-label="Shared tools">
      <a className="tool-card panel" href={href("/workspace")}><FileStack size={20} /><div><h3>Workspace</h3>
        <p>{workspace.collection ? `${workspace.collection.documents.length} processed documents in “${workspace.collection.name}”`
          : "Choose a folder; documents are processed once and shared by every solution."}
          {workspace.schema ? ` · Schema “${workspace.schema.name}” v${workspace.schema.version}.` : " · Suggest a schema from exemplars."}</p></div>
        <ArrowRight size={16} /></a>
      <a className="tool-card panel" href={href("/compare")}><GitCompareArrows size={20} /><div><h3>Compare runs</h3>
        <p>{runs.length ? `${runs.length} run${runs.length === 1 ? "" : "s"} available. Matrix of values, agreement, confidence and grounding.`
          : "After running solutions, compare their values side by side."}</p></div><ArrowRight size={16} /></a>
    </section>

    {runs.length > 0 && <section aria-labelledby="recent-heading">
      <h2 id="recent-heading" className="section-heading">Recent runs</h2>
      <div className="wb-table-scroll"><table className="wb-table">
        <thead><tr><th>Run</th><th>Solution</th><th>Schema</th><th>Documents</th><th>Status</th><th>Created</th></tr></thead>
        <tbody>{runs.slice(0, 6).map(run => <tr key={run.id}>
          <td><a href={href(solutions?.find(item => item.id === run.solution)?.route ?? "/compare", { run: run.id })}>{run.label}</a></td>
          <td>{run.solution_name}</td><td>{run.schema_name} v{run.schema_version}</td>
          <td>{run.documents_done}/{run.documents_total}</td><td><span className={`run-status ${run.status}`}>{run.status}</span></td>
          <td>{new Date(run.created_at).toLocaleString()}</td></tr>)}</tbody>
      </table></div>
    </section>}
  </div>;
}
