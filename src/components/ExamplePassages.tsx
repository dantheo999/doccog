import type { Recipe } from "../types";

export default function ExamplePassages({ examples }: { examples: Recipe["examples"] }) {
  return <>
    <p className="excerpt-note">Source verification confirms text origin, not the accuracy of model annotations. Review every labelled extraction.</p>
    {examples.map((example, index) => <article key={index} className="example-passage">
      <span className="eyebrow">EXAMPLE {index + 1}</span>
      {example.source ? <div className="example-provenance">
        <strong>{example.source.name}</strong>
        <span>Source span {example.source.start.toLocaleString()}–{example.source.end.toLocaleString()} · Unicode characters (end exclusive)</span>
        <span className={example.source.verified ? "source-verified" : "source-unverified"}>
          {example.source.verified ? "Verified source text" : "Unverified source metadata · imported or edited; historical origin only"}
        </span>
        <details><summary>Inspect source metadata</summary><pre>{JSON.stringify(example.source, null, 2)}</pre></details>
      </div> : <p className="source-unverified">Legacy example · unverified source origin</p>}
      <p>{example.text}</p>
      <span className="microcopy">{example.extractions.length} labelled extractions</span>
    </article>)}
  </>;
}
