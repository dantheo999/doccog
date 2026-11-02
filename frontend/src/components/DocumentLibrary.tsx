import { useRef } from "react";
import { ChevronRight, FileText, FolderOpen, Search, ShieldCheck } from "lucide-react";
import type { LocalDocument } from "../types";

interface Props {
  documents: LocalDocument[];
  filteredDocuments: LocalDocument[];
  folder: string;
  ignored: number;
  query: string;
  selectedPath: string;
  exemplarPaths: string[];
  busy: boolean;
  onFolder: (files: FileList | null) => void;
  onQuery: (query: string) => void;
  onPreview: (document: LocalDocument) => void;
  onExemplar: (path: string) => void;
}

export default function DocumentLibrary({ documents, filteredDocuments, folder, ignored, query, selectedPath,
  exemplarPaths, busy, onFolder, onQuery, onPreview, onExemplar }: Props) {
  const folderInput = useRef<HTMLInputElement>(null);
  return <aside className="documents-panel panel">
    <div className="sidebar-heading"><span className="eyebrow">LOCAL WORKSPACE</span><span className="count">{documents.length}</span></div>
    <h2>Documents</h2>
    <button className="secondary folder-button" onClick={() => folderInput.current?.click()} disabled={busy}><FolderOpen size={17} />Choose folder</button>
    <input type="file" ref={node => {
      folderInput.current = node;
      node?.setAttribute("webkitdirectory", "");
      node?.setAttribute("directory", "");
    }} aria-label="Choose document folder" multiple hidden disabled={busy}
      onChange={event => {
        onFolder(event.target.files);
        event.target.value = "";
      }} />
    <div className="format-list"><span>PDF</span><span>DOCX</span><span>TXT</span><span>MD</span></div>
    {documents.length > 0 && <>
      <div className="folder-name" title={folder}><FolderOpen size={14} />{folder}</div>
      <div className="search-field"><Search size={15} /><input aria-label="Filter documents" placeholder="Filter documents…"
        value={query} onChange={event => onQuery(event.target.value)} /></div>
      <p id="exemplar-help" className="microcopy exemplar-help">Preview a document; check up to five exemplars separately.</p>
    </>}
    <div className="document-list">
      {documents.length === 0 ? <div className="empty sidebar-empty"><FolderOpen size={32} /><p>Your documents,<br />one folder away.</p>
        <span>Choose a local folder to begin.</span></div> : filteredDocuments.map(document => {
        const checked = exemplarPaths.includes(document.path);
        const extension = document.name.split(".").pop()?.toUpperCase();
        return <div key={document.path} className="document-row">
          <input type="checkbox" className="exemplar-checkbox" aria-label={`Use ${document.path} as exemplar`}
            aria-describedby="exemplar-help" title={`Use ${document.path} as exemplar`} checked={checked}
            disabled={busy || (!checked && exemplarPaths.length >= 5)} onChange={() => onExemplar(document.path)} />
          <button className={`document-item ${selectedPath === document.path ? "active" : ""}`}
            disabled={busy} onClick={() => onPreview(document)} title={document.path} aria-pressed={selectedPath === document.path}>
            <FileText size={19} /><span><strong>{document.name}</strong><small>{extension} · {document.storedId
                ? `${document.size.toLocaleString()} chars · processed` : `${(document.size / 1024).toFixed(1)} KB`}</small>
              {document.path !== document.name && <small className="relative-path">{document.path}</small>}</span>
            {selectedPath === document.path && <ChevronRight size={14} />}
          </button>
        </div>;
      })}
      {documents.length > 0 && !filteredDocuments.length && <p className="microcopy">No matching documents.</p>}
    </div>
    {ignored > 0 && <p className="microcopy ignored-files">{ignored} unsupported file{ignored === 1 ? "" : "s"} excluded.</p>}
    <div className="privacy-note"><ShieldCheck size={17} /><div><strong>Local originals stay untouched</strong>
      <p>Only previewed or checked files are uploaded to this local app. Processing sends text to your configured Azure model.
        Complete exemplar texts are saved in this browser and included in recipe JSON exports.</p></div></div>
  </aside>;
}
