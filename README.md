# Document Cognition Studio

A local React demo for teaching a reusable extraction task and applying it to
documents with [Google LangExtract](https://github.com/google/langextract).
The Python API uses Azure Foundry-hosted OpenAI models; it does not use Gemini,
Hugging Face, or a public demo service for your data.

## Start with Docker Desktop

Use Linux containers in Docker Desktop. In PowerShell:

```powershell
Set-Location 'F:\Git\source\repos\sandpit\Projects\DocCognition\LangExtract'
Copy-Item .env.example .env
notepad .env
docker compose up --build -d
```

Set these values in `.env`:

```dotenv
AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/openai/v1/
AZURE_OPENAI_DEPLOYMENT=YOUR-DEPLOYMENT-NAME
AZURE_OPENAI_API_KEY=YOUR-API-KEY
HOST_PORT=8080
```

The endpoint can be the resource root or its `/openai/v1/` URL. Supported Azure
hostnames end in `.openai.azure.com`, `.services.ai.azure.com` or
`.cognitiveservices.azure.com`. This prototype targets public Azure; sovereign
clouds and legacy `api-version` APIs are not included.

Use the **deployment name**, not necessarily the base model name. The deployment
must support Chat Completions and JSON mode; GPT-4.1 or GPT-4o family deployments
are suitable starting points. Responses-only models are not supported.

Open **http://localhost:8080** in Edge or Chrome. Change `HOST_PORT` if that port
is already in use. The application starts without credentials, but Generate
example, Extract and Optimize remain disabled until all three Azure settings are present.
The status badge indicates configuration presence, not a verified Azure connection.

After changing `.env`:

```powershell
docker compose up -d --force-recreate backend
```

To stop the demo, use `docker compose down`. Inspect failures with
`docker compose logs --tail 100 backend frontend`. Do not post logs or `.env`
contents publicly without checking for sensitive information.

## Demo workflow

1. **Choose folder** opens the browser's native local folder picker. Supported
   files are listed recursively with relative paths. Choose `samples` for two
   small example policies, or your own folder. File selection does not bulk-upload
   the folder: a document is uploaded only for preview or exemplar generation.
2. Select a document and describe the extraction in natural language, for example:
   **"Extract organisations, regulations and privacy obligations. For each
   obligation include the responsible party and deadline or retention period."**
3. Optionally check up to **five exemplar documents**, independently of the
   preview selection. With none checked, generation uses the previewed document.
   Click **Generate example**. The Azure model receives every selected document's
   **complete parsed text** and creates generic instructions and representative
   few-shot annotations. The backend inserts one whole-document example per
   source, verifies exact annotation substrings, and records source filenames,
   Unicode offsets and SHA-256 digests. Annotations need not label every matching
   occurrence: they illustrate requested classes and distinct patterns, while
   instructions require extraction of all matching target content. Irrelevant
   sources or invalid output fail explicitly rather than producing invented examples.
   Generation and Optimize send the expected response schema and eligible passage
   IDs alongside the source data, while retaining Chat Completions JSON-mode
   compatibility. A rejected response identifies the failed field/limit, missing
   source, invalid passage selection or nonliteral annotation in the UI and backend
   logs. During generation/Optimize, a nonliteral quote can be restored only if
   collapsing whitespace reveals exactly one contiguous source match. The backend
   copies the original source characters, rechecks annotation length and retains
   unchanged source text/digests. It logs only the example/annotation indices.
   Changed words, punctuation, case, ligatures or ambiguous matches still fail;
   annotations are never dropped. Stored annotations and manual/imported recipes
   must match source whitespace and line breaks exactly. Rejections do not
   establish that the documents or goal are unsuitable; a model can return invalid
   output for otherwise valid inputs. The app does not log full model responses or
   source text, and a failed attempt leaves the active recipe unchanged.
   Invalid generation/Optimize JSON or source annotations trigger at most one
   corrective model request against the same full sources and schema. For a
   structurally valid, source-covered proposal, the correction must retain the
   per-source annotation counts and classes; it cannot silently drop a bad quote.
   If correction fails, the operation fails explicitly with no substituted recipe.
4. Inspect the **Instructions**, **Few-shot JSON** and **Example text** tabs.
   Few-shot JSON shows annotations only: `[{ "extractions": [...] }, ...]`, with
   exactly one entry per exemplar in the original source order. Source text and
   provenance stay in **Example text** and full recipe exports, not this editor
   or Optimize's annotation comparison. Instructions and annotations are editable.
   **Apply edits** merges annotations into the unchanged source texts, validates and
   versions the active recipe. Invalid JSON or example spans are rejected;
   unapplied edits do not affect extraction. Each tab has an **expand** button
   for a full-browser-window review/editor view. Switch tabs or apply edits in
   that view, then use **Close full-page view** or **Escape** to return. Closing
   preserves unapplied drafts; it does not automatically apply them.
   **Example text** shows source provenance. Legacy recipes without provenance
   remain usable but are labelled unverified. Importing a recipe does not freshly
   verify its sources. Annotation editing cannot change source text/provenance;
   text/source changes in an externally edited, imported recipe remove its verified
   claim. Source-backed text does not certify model-generated labels.
5. Select any document, including the original source, and click **Extract**.
   LangExtract runs against its complete extracted text. Findings include class,
   attributes, character interval and grounding status. Click a card to focus
   its source highlight; click a highlight to select a finding. Repeated clicks
   on overlapping highlights cycle through the overlapping findings.
6. Filter by extraction class, inspect raw JSON, or export results. The active
   recipe stays unchanged on document switches, goal edits and failed generation.
   Only successful generation or a validated explicit edit/import/optimization
   application replaces it.
7. After a completed extraction, click **Optimize** to propose a revised recipe.
   Review the reasons and before/after Instructions, Few-shot JSON and Example
   text. **Apply revision** increments the recipe version and replaces unapplied
   drafts; existing findings stay visible but become stale. Click **Extract**
   yourself to evaluate the new recipe. **Discard**, **Close** or **Escape** leaves
   the active recipe, findings and drafts unchanged.

Recipe export/import uses the toolbar icons. The recipe is saved in this
browser's local storage and survives reloads; folder selection and documents do
not. Different browsers and localhost ports have separate storage. If an active
recipe changes after extraction, the old findings remain labelled as using an
earlier recipe until you run Extract again.

On desktop, the document list and recipe column fill the available browser
height. Source and findings share the middle column equally (50/50), with
independent scrolling for source text, findings, the document list and recipe
content. Narrower windows retain a stacked layout.

## Recipe optimization

Optimize uses the **applied recipe that produced the findings**, not editor drafts.
It is available even after an extraction with zero findings, but not when the
document or recipe revision no longer matches the result. Run Extract again after
changing the recipe. Conflicting document/recipe/model actions are disabled while
optimization runs; processing settings can still change for subsequent extractions.

The initial structured model request includes the **complete canonical document**,
**all findings** (not just the visible class filter), the goal, instructions,
every example/annotation and the options used for that extraction. Optimize
preserves the **complete original exemplar set**, including text, provenance,
source order and source-character accounting. It requests a revised annotation
set for every verified original exemplar; omissions, duplicates or replacement
sources receive one corrective request, then fail explicitly if still invalid.
The target document is analysis-only, not an automatically added exemplar.
Older verified excerpt-based recipes remain eligible as-is. Imported, legacy or
otherwise unverified examples are retained unchanged, including their annotations.
If no originals are verified, Optimize can revise instructions and reasons only.
Generate a new recipe to change its exemplar documents.

Each Before/After comparison header shows its example count. The browser also
rejects proposals that remove, add, reorder or replace original exemplar sources,
or revise unverified examples, both on receipt and before applying.

Optimization does not silently sample/truncate the document, omit findings, apply
changes or rerun extraction. A context-capacity failure leaves the current state
unchanged: choose a smaller document or an appropriately sized deployment.
Upload limits are not the model's context limits. Findings and proposed labels
are not ground truth; Optimize suggests improvements, not a measured accuracy gain.

## Extraction settings

Use the **settings cog** beside the Azure connection indicator to tune processing.

| Parameter | Default | Allowed range |
| --- | ---: | ---: |
| Target chunk size (characters) | 6,000 | 500-20,000 |
| Maximum concurrent LLM requests | 2 | 1-16 |
| Extraction passes | 1 | 1-5 |
| Previous-chunk context (characters) | 0 (off) | 0-6,000 |

**Save settings** validates and applies the values. Settings are stored separately
from the recipe in this browser and survive reloads. **Cancel**, **Close** or
**Escape** discard unsaved changes. **Reset to defaults** resets the dialog draft;
use Save to apply it. Settings do not alter instructions/examples or recipe versions,
and changes during processing affect only the next extraction.

Each chunk request repeats the active instructions and examples. Smaller chunks
can reduce per-call source text but add requests, repeated tokens and context loss.
Higher concurrency can improve throughput only within Azure request/token quotas.
The worker count and chunk batch size both use the concurrency setting; it is a
limit on simultaneous requests, not a limit on total requests.

Additional passes run sequentially and can improve coverage, at increased time and
cost; better accuracy or speed is not guaranteed. Previous-chunk context includes
up to the specified number of trailing source characters from the immediately
preceding chunk, not previous model responses. Short chunks provide fewer context
characters. This does not prevent parallel LLM calls; zero disables context.

The results header shows the settings actually used, and raw/exported result JSON
records them under `options`. Later settings edits do not relabel existing results.
The extraction API accepts an optional `options` object with `chunk_size`,
`max_workers`, `extraction_passes` and `context_window_chars`; omitted values retain
the defaults. Both the API and UI reject out-of-range values.

## Document handling and grounding

| Format | Text used for extraction |
| --- | --- |
| PDF | Selectable page text, with blank lines between pages |
| DOCX | Body paragraphs and table rows in document order |
| TXT / MD | UTF-8, or BOM-marked UTF-16; whitespace preserved |

Source highlighting is on the **extracted text**, not the original document
pages or Word layout. DOCX headers, footers, embedded images, text boxes and
nested tables are not included. PDF text order depends on its text layer.
Scanned/no-text and encrypted PDFs are rejected; OCR is not included.

Uploads are limited to **20 MiB**, PDF documents to **1,000 pages**, extracted
text to **500,000 Unicode characters**, and expanded DOCX contents to **100 MiB**.
Oversized or invalid files fail explicitly rather than being silently truncated.
These limits are guardrails, not promises of successful processing at the limit.

Recipe generation uses **all canonical text from every exemplar**, with no sampling
or excerpt selection. Example text supports the same **500,000-code-point** limit
as a parsed document; individual annotation text remains limited to **12,000 code
points**, with up to 100 annotations per example. Whole exemplar texts are sent
again with **every LangExtract chunk**, so large or numerous exemplars increase
token usage, latency and context pressure. Generation/Optimize capacity failures
do not silently truncate inputs. Browser storage has its own quota: if saving a
large recipe fails, the recipe remains active with a warning to export a copy.
The runtime prompt explicitly separates demonstration examples from LangExtract's
final target-chunk question: examples guide format/patterns, not target facts.
This does not certify model accuracy; all findings, including ungrounded ones,
remain visible for review rather than being silently filtered.
Every chunk response is checked with LangExtract's JSON resolver before alignment.
Malformed/incomplete output triggers one corrective request for that chunk only;
successfully parsed chunks are not rerun. Concurrency remains bounded by the worker
setting, including retries. A second failure aborts the whole extraction and identifies
invalid JSON, an invalid envelope/annotation shape, or truncated output in the UI.
No malformed chunk is skipped and no partial result is returned. Corrective requests
add tokens/time and can add at most one model call per chunk; they do not retry
authentication/quota failures as format errors or guarantee semantic accuracy.
Older short recipes remain usable; regenerate to get whole-document examples.
Extraction processes the **whole** document, by default in approximately
6,000-character chunks, with two concurrent calls and one extraction pass. The
settings cog changes these processing parameters without changing generation.
Only one model operation is allowed at a time. Narrow tasks and small documents
are recommended for demos. Long operations can time out at the frontend proxy's
15-minute limit; the backend may still be finishing the operation.

Intervals are zero-based Unicode code-point offsets with an exclusive end.
The viewer handles JavaScript's different Unicode string indexing. **Exact**
means the returned source substring equals the extraction text. **Approximate**
means LangExtract aligned a non-identical span (dashed underline); **ungrounded**
means no valid source interval was returned. Neither an exact match nor a
highlight guarantees that the interpretation is correct. Review all findings.

## Privacy and lifecycle

- Original local files are never modified or mounted into Docker.
- Uploaded raw file objects are closed after parsing. Extracted text is held
  only in bounded backend memory: document IDs expire after 30 minutes, and
  expired text is purged on the next cache operation. Earlier entries may be
  evicted by the 20-document cache limit. Reselect a document if it expires;
  stopping the backend clears all cached text.
- The UI holds documents, findings, exemplar selection and optimization candidates
  in memory. New saved/exported recipes **contain complete exemplar document texts**,
  their annotations, goal and source provenance, but no API key. These texts can contain
  sensitive information and survive reloads in browser storage: review before
  exporting/sharing and clear browser storage when appropriate.
- **Generate example sends complete selected exemplar texts and the task to your
  configured Azure service. Extract sends document chunks, instructions and
  few-shot examples. Optimize sends the full extraction document, all findings,
  the applied recipe and run options. Azure inference can incur costs.** Use only
  documents you are authorised to send to that deployment.
- Credentials are backend environment variables only; `.env` is ignored by Git
  and excluded from Docker build contexts. No keys are included in frontend
  bundles or returned by the API.
- Published only on `127.0.0.1`; the backend has no published host port. This is
  a trusted, single-user local prototype, not an authenticated production service.

## Development and checks

Docker is the supported runtime. For local development, Python 3.11+ and Node
24+ are recommended:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r backend\requirements-dev.txt
Set-Location backend
& ..\.venv\Scripts\python.exe -m pytest -q
# Set the three AZURE_OPENAI_* variables in this shell before running the API.
& ..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In another PowerShell terminal:

```powershell
Set-Location 'F:\Git\source\repos\sandpit\Projects\DocCognition\LangExtract\frontend'
npm ci
npm test
npm run build
npm run dev
```

The Vite dev server proxies `/api` to `127.0.0.1:8000`. API docs are available at
`/api/docs` on the backend; production uses the Nginx same-origin proxy.
Runtime Python dependencies and frontend dependencies are locked in
`backend\requirements.txt` and `frontend\package-lock.json`. Tests do not require
Azure credentials: they cover parsing, strict recipe validation, real LangExtract
alignment with deterministic inference, Unicode/overlap highlighting, persistent
recipes, stale-preview protection, multi-source grounded generation, full-input
optimization and explicit candidate review/application.

## References

The interface is an original React implementation inspired by the grounded
source/findings interaction in [RadExtract](https://google-radextract.hf.space/)
and the editable prompt/few-shot presentation in the
[LangExtract playground](https://google-langextract.hf.space/).

- [LangExtract repository and OpenAI provider configuration](https://github.com/google/langextract)
- [Introducing LangExtract](https://developers.googleblog.com/en/introducing-langextract-a-gemini-powered-information-extraction-library/)
- [Azure OpenAI endpoint and deployment configuration](https://learn.microsoft.com/en-us/azure/foundry-classic/openai/how-to/switching-endpoints)

LangExtract is distributed under Apache 2.0; dependency notices remain with
their packages. This is a demonstration, not a guarantee of extraction accuracy.
