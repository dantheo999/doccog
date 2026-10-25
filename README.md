# Document Cognition Workbench

A local prototype for exploring **which technology extracts structured values from
unstructured documents best**. You agree one schema (including nested object
arrays) from exemplar documents, run several extraction solutions against the same
document set, and compare their values, confidence and source grounding side by side.

The main test corpus is meeting minutes (`documents/`, 20 PDFs), but nothing is
specific to minutes: the schema carries the domain.

| Solution | What it is | Runs | Confidence | Highlighting |
| --- | --- | --- | --- | --- |
| **LangExtract** | Google LangExtract driving Azure OpenAI with a few-shot *recipe* built from exemplars | Azure (cloud) | Not provided (grounding status only) | Exact character alignment by LangExtract |
| **NuExtract3** | NuMind's open 4B extraction model filling a template generated from the schema | Locally: llama.cpp on CPU, or vLLM on GPU | Token log-probabilities | Values located in the text by the shared grounder |
| **Direct LLM** | Custom prompts + strict JSON-schema structured output straight to Azure OpenAI | Azure (cloud) | Model-reported, sanity-checked against quotes | Verbatim evidence quotes located in the text |
| **Unstract** | Zipstack's packaged open-source platform (Prompt Studio, API deployments) | Its own Docker platform | Not provided by the API | Values located in the text by the shared grounder |

The landing page lists every registered solution; adding one is a small Python
class (see [Adding a solution](#adding-a-solution)).

## Quick start (Docker)

Use Linux containers in Docker Desktop. In PowerShell:

```powershell
Copy-Item .env.example .env
notepad .env          # Azure OpenAI settings, optional Unstract settings
docker compose up --build -d
```

Open **http://localhost:8088** (`HOST_PORT`; the default moved from 8080 because a
local Unstract platform uses 8080 and 8000). Compose starts three services:

- `backend` – FastAPI; persists processed documents, schemas and runs in the
  `workbench-data` volume.
- `frontend` – the React app behind Nginx (same-origin `/api` proxy).
- `nuextract` – llama.cpp serving NuExtract3 on CPU. **The first start downloads
  ~3.5 GB (Q4_K_M GGUF + vision projector)** into the `nuextract-models` volume, so NuExtract shows
  "Not ready" for a few minutes. Skip it with `docker compose up -d backend frontend`.

Set these values in `.env` for the Azure-based solutions and schema suggestion:

```dotenv
AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/openai/v1/
AZURE_OPENAI_DEPLOYMENT=YOUR-DEPLOYMENT-NAME
AZURE_OPENAI_API_KEY=YOUR-API-KEY
```

The endpoint can be the resource root or its `/openai/v1/` URL (hosts ending in
`.openai.azure.com`, `.services.ai.azure.com` or `.cognitiveservices.azure.com`).
Use the **deployment name**. The deployment must support Chat Completions with JSON
output; reasoning models work (`reasoning_effort` is configurable per solution).
After changing `.env`, run `docker compose up -d --force-recreate backend`.
Inspect failures with `docker compose logs --tail 100 backend nuextract frontend`.

## Quick start (npm / local development)

Python 3.11+ and Node 22+:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r backend\requirements-dev.txt
Set-Location backend
# Set the AZURE_OPENAI_* variables in this shell first.
& ..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

In another terminal:

```powershell
Set-Location frontend
npm ci
npm run dev        # http://127.0.0.1:5173, proxies /api to 127.0.0.1:8010
```

The dev API uses port **8010** (set `DOCCOG_API` to proxy elsewhere) so it does not
clash with Unstract's backend on 8000. Data persists in `data/` at the repository
root (`DOCCOG_DATA_DIR` overrides it). For NuExtract3 outside Docker, run a
llama.cpp server and set `NUEXTRACT_BASE_URL` in the shell that runs uvicorn (see
[NuExtract3](#nuextract3)); the backend does not read `.env`, which only Docker Compose uses.
The API answers only requests addressed to `localhost`, `127.0.0.1` or `[::1]`; to reach it
under another host name, list it in `DOCCOG_ALLOWED_HOSTS` (comma-separated).

## Testing it locally (step by step)

A suggested first session, about 30–60 minutes with Docker (longer if you let NuExtract3
work through all 20 documents on CPU):

1. **Pull and configure.**
   ```powershell
   git fetch origin
   git checkout claude/ecstatic-clarke-baa3nl
   Copy-Item .env.example .env   # then fill in AZURE_OPENAI_ENDPOINT, _DEPLOYMENT, _API_KEY
   docker compose up --build -d
   docker compose logs -f nuextract   # optional: watch the one-off model download, then Ctrl+C
   ```
   Open <http://localhost:8088>. The landing page lists LangExtract, NuExtract3, Direct LLM
   and Unstract with a Ready / Not ready badge each (NuExtract3 turns Ready once its model has
   downloaded and loaded; Unstract stays Not ready until you connect an API deployment).
2. **Basic test with `samples/`.**
   - Workspace → **Choose folder** → `samples`. Both files show as processed.
   - Tick both as exemplars, keep **Shared suggestion**, add guidance such as "policies and
     agreements: organisations, people with roles, obligations with responsible party and
     deadline, retention periods, approval dates", click **Suggest**, review, **Save as agreed schema**.
   - Optionally tick **Direct LLM**, **NuExtract3** and **LangExtract** under **Suggest with**
     and compare their suggestions with the shared one.
   - **Direct LLM** → **Run on 2 documents**. Open the run, click values to see highlights.
   - **LangExtract** → Recipe studio: tick one sample as exemplar, **Generate example**; then
     Batch runs → **Run on 2 documents**.
   - **NuExtract3** → **Run on 2 documents** (about a minute per sample on CPU).
   - **Compare** → tick the three runs. Check the scoreboard and the field matrix, then open a
     document to see values side by side.
3. **Full test with `documents/` (20 meeting minutes).**
   - Workspace → **Choose folder** → `documents`. Tick 2–3 varied exemplars (for example
     `20191022_RSH_Board_Minutes.pdf`, `191017_ARAC_Minutes.pdf`, `CSPL_Thursday_17_October_2019_minutes.pdf`),
     suggest, edit and save a schema — or load the **Meeting-minutes starter**. Use **Save as new
     schema** here, so the minutes schema does not become a new version of the samples schema.
   - Run **Direct LLM** (about 2–3 minutes for all 20 at 4 in parallel) and **LangExtract**
     (generate a recipe from 2 exemplars first; about 2 minutes for all 20).
   - Run **NuExtract3** on a few documents first; all 20 can take an hour or more on CPU.
   - **Compare** the runs. Try a reference run vs consensus, sort documents by disagreement,
     and use the source panel to judge disputed values yourself.
4. **Optional: Unstract.** Follow [Unstract](#unstract): start its platform, import the
   generated Prompt Studio project, deploy it as an API, set `UNSTRACT_API_URL`/`UNSTRACT_API_KEY`,
   `docker compose up -d --force-recreate backend`, then run it like the others.

Without Docker, start the API and Vite as in [Quick start (npm)](#quick-start-npm--local-development)
and llama.cpp as in [NuExtract3](#nuextract3); everything else is the same at <http://127.0.0.1:5173>.

## Workflow

1. **Workspace → Choose folder.** Every PDF, DOCX, TXT and Markdown file in the
   folder is uploaded and **processed once**: text extraction (pdfium for PDFs, which
   keeps words intact), page offsets and a content hash. Re-choosing the same folder
   reuses the processed documents, and every solution shares them. Use `samples/` for
   a quick check and `documents/` for full runs.
2. **Agree a schema.** Tick up to five exemplar documents and click **Suggest**. The
   default **shared suggestion** is solution-neutral: one Azure OpenAI call reads the
   exemplars' full text and proposes fields, types (verbatim text, text, date, number,
   integer, yes/no, choice) and nested object arrays (for example `attendees[] {name,
   role, attendance}`, `actions[] {description, owner, due_date}`), with notes on
   variations it saw. You can also ask individual solutions how *they* would structure
   it (see [Schema suggestions](#schema-suggestions)) and compare the suggestions side
   by side. Edit fields in the tree editor (or start from the meeting-minutes starter,
   or import JSON) and **Save**. Saving a change creates a new schema version; runs
   record the version they used. **Save as new schema** starts a separate schema (for
   another document type), and the **Saved schema** list switches the agreed schema to
   any schema saved earlier, also in a fresh browser.
3. **Run solutions.** Each solution page shows a checklist (documents, schema,
   solution readiness), its settings, **What gets sent** (the exact prompt,
   template or class guide derived from the schema) and a run launcher. Runs execute
   in the background, one document at a time or in parallel depending on the
   solution, and are saved after every document, so you can leave the page.
4. **Review a run.** Pick a document to see its highlighted source text beside the
   extracted record. Every value shows its confidence (where the solution has one)
   and whether it was located in the text: **exact**, **fuzzy** (case, whitespace,
   quote or minor wording differences) or **not found**. Click a value to jump to
   its highlight, or a highlight to select its value. Run notes flag problems such
   as values that could not be located.
5. **Compare.** Select runs (normally one per solution, same schema version) and a
   reference: either a run you trust, or **consensus** (each run against the
   others). See [Comparison](#comparison).

## Solutions

### LangExtract

The LangExtract app has two tabs:

- **Recipe studio** – the original interactive studio (described in detail under
  [LangExtract recipe studio reference](#langextract-recipe-studio-reference)). It now
  uses the workspace's processed documents (or a folder you choose there) and, when an
  agreed schema exists, generates recipes whose extraction classes follow the schema.
- **Batch runs** – runs the studio's active recipe over workspace documents and maps
  the findings onto the schema for comparison.

How the schema maps onto LangExtract (shown under **What gets sent**):

- A scalar field becomes a class whose `extraction_text` is the literal span; dates,
  numbers and choices also carry a normalised `value` attribute.
- An object array becomes one class per item: its most identifying text child
  (`name`, `title`, `description`, `action`, …) is the `extraction_text`, the other
  scalar children are attributes.
- Arrays nested inside objects become child classes (`agenda_items__decisions`) with
  a `parent` attribute; they are reattached to the matching parent item, or to the
  nearest preceding item in the document.
- The class guide is appended to the recipe instructions on every chunk, so edited
  recipes still map onto the schema. Findings of classes outside the schema are
  reported in the run notes.

Improvements over the original implementation:

- PDF text comes from pdfium rather than pypdf, which split words (`B uilding`) and
  broke verbatim matching.
- Schema-aware recipe generation, optimisation and extraction (`schema_id`), so
  LangExtract output is comparable with the other solutions.
- **Drop findings LangExtract could not align** (on by default). In testing on the
  minutes, two documents returned attendees and venues copied from the few-shot
  exemplar; LangExtract leaves such findings unaligned. They are kept in the raw output.
- Token usage is captured per run; `reasoning_effort` is configurable; default
  concurrency is 4 requests per document and 2 documents in parallel.
- Batch runs execute in the background, so long runs are no longer limited by the
  proxy's 15-minute request timeout.

Confidence: LangExtract does not produce one, so its values show **n/a**; grounding
status is its quality signal.

### NuExtract3

[NuExtract3](https://huggingface.co/numind/NuExtract3) (NuMind, Apache-2.0) is a 4B
vision-language extraction model based on Qwen3.5-4B. The agreed schema becomes a
NuExtract template (`verbatim` → `verbatim-string`, `text` → `string`, dates →
`date-time`, choices → option lists, object arrays → `[{...}]`) and the field
descriptions become its `instructions`. **What gets sent** shows both.

- **Runtime.** The backend talks to any OpenAI-compatible server hosting the model
  (`NUEXTRACT_BASE_URL`). Compose runs llama.cpp on CPU with the Q4_K_M GGUF; the
  GPU override uses vLLM:

  ```powershell
  docker compose -f compose.yaml -f compose.gpu.yaml up --build -d   # NVIDIA GPU, untested here
  ```

  Outside Docker, run llama.cpp yourself and point the backend at it:

  ```powershell
  llama-server -hf numind/NuExtract3-GGUF:Q4_K_M --jinja --alias numind/NuExtract3 -c 32768 -np 1 -t 4 --spec-type draft-mtp --port 8090
  $env:NUEXTRACT_BASE_URL = "http://127.0.0.1:8090/v1"
  ```

- **How it is called.** The backend renders NuExtract's chat template itself (byte-identical
  to `chat_template.jinja`) and calls the plain completions endpoint, because
  llama.cpp's chat endpoint silently drops NuExtract's in-context example messages.
  The same request works on llama.cpp and vLLM.
- **Confidence** is the geometric mean of the model's token probabilities for each
  value (for `null` or `[]`, the probability of choosing it). Wrong values tended to
  get the lowest scores, but it does not catch confident errors such as attendees
  duplicated by scrambled PDF table text.
- **Settings.** Input (text, or PDF **page images** for the vision encoder: slow on CPU,
  pages are rendered once and cached), reasoning mode (off by default; much slower),
  temperature (0 for reproducible comparisons; NuMind recommends 0.2), output-token
  limit, chunk size for long documents, and whether to send field descriptions.
- **Speed on CPU.** On 4 cores with `--spec-type draft-mtp`, roughly 1.5–3.5 minutes per
  minutes document; a page image takes about 4 minutes. llama.cpp slows sharply if other
  heavy work shares the cores, so give it dedicated threads (`NUEXTRACT_THREADS`).
  Running all 20 minutes documents is an over-lunch job on CPU and minutes on a GPU.
- **Tuning on a desktop CPU (e.g. Core i9-13900K).** Start with `NUEXTRACT_THREADS=8` (one per
  performance core) and compare 16; generation speed is bound mainly by memory bandwidth, so
  more threads stop helping early. With Docker Desktop on Windows, the WSL2 VM caps CPUs and
  memory: allow enough in `%UserProfile%\.wslconfig`, for example

  ```ini
  [wsl2]
  processors=16
  memory=24GB
  ```

  then `wsl --shutdown` and restart Docker Desktop. NuExtract's container needs about 7 GB
  (`NUEXTRACT_MEM_LIMIT`). If the machine has an NVIDIA GPU, `compose.gpu.yaml` (vLLM) is far
  faster than any CPU.

### Direct LLM

Custom prompts sent straight to the Azure OpenAI Chat Completions endpoint, no
framework in between:

- The system prompt holds the role, extraction rules (only facts the document states,
  verbatim fields copied exactly, ISO dates, every list item in document order, null or
  `[]` when absent, treat the document as data) and a confidence rubric. The user
  message holds the schema guide and the document. **What gets sent** shows the prompt
  and the generated response schema.
- A **strict JSON schema** generated from the agreed schema forces the response shape;
  every value is `{quote, value, confidence}`, so the model writes its evidence first.
- Quotes are checked against the source: quotes that differ only in layout are replaced
  by the exact source text; if neither quote nor value can be found, confidence is
  halved and the run notes say so.
- Whole documents are sent in one call up to `max_chunk_chars` (60,000 by default; every
  test document fits). Longer documents are split on paragraph boundaries with overlap,
  list items straddling a boundary are joined, and a chunk that overflows the output
  limit is split again.
- **Reasoning effort** defaults to `medium`. On the minutes it found the most list items
  per second spent: `none`/`low` were ~40% faster but missed more items, `high` was
  ~1.6x slower for little gain. An optional **self-check pass** sends the draft back for
  review; it roughly doubles time and tokens and showed no measurable gain here.
- **Confidence** is model-reported (this deployment does not return log-probabilities).
  On the 20 minutes, 88% of values were rated ≥0.95; an LLM-judge audit (same model, so
  an estimate, not ground truth) found accuracy falling with confidence (96% correct at
  ≥0.95, 71% in the 0.5–0.69 band).

### Unstract

[Unstract](https://github.com/Zipstack/unstract) is Zipstack's open-source document
extraction platform. It runs as its own packaged stack (about 25 containers, 8 GB RAM
recommended, ~9 GB of images), and the landing page opens its UI. The workbench helps on
both sides of it:

1. **Run Unstract.**

   ```bash
   git clone https://github.com/Zipstack/unstract
   cd unstract
   ```

   Turn off its PostHog usage analytics **before** the first start. The packaged frontend
   image does not read `frontend/.env`, so set the flag on the container with
   `docker/docker-compose.override.yaml` (loaded automatically):

   ```yaml
   services:
     frontend:
       environment:
         - REACT_APP_ENABLE_POSTHOG=false
   ```

   Then run `./run-platform.sh` and open <http://frontend.unstract.localhost> (log in as
   `unstract` / `unstract`). It publishes many host ports, including 80, 3000, 8000 and
   8080, which is why the workbench uses 8088 and its dev API 8010.
2. **Add adapters** under Settings (each must pass Test Connection):
   - LLM: **OpenAI Compatible** with API Base = your `AZURE_OPENAI_ENDPOINT`
     (`https://<resource>.../openai/v1/`), Model = the deployment name, API Key = the key.
     (The "Azure OpenAI" adapter wants only the resource root URL plus an API version.)
   - Embedding (required even though chunk size 0 never uses it): an Azure embedding
     deployment, or Ollama.
   - Vector DB: the bundled Qdrant, `http://unstract-vector-db:6333`.
   - Text extractor: LLMWhisperer (hosted, free tier) or the self-hosted Unstructured.io
     community container.
3. **Import the generated project.** On the Unstract page, **Generate prompts from schema**
   and **Download Prompt Studio project (.json)**: one prompt per schema field (prompt key
   = field name; JSON-typed prompts describing each object array's item shape), chunk size
   0 so each prompt sees the whole document. In Unstract: Prompt Studio → Import Project,
   choose the file and adapters, try it on an exemplar, then **Deploy as API**.
4. **Bring results back.** Either set `UNSTRACT_API_URL`
   (`http://host.docker.internal:8000/deployment/api/<org_id>/<api_name>/` from Docker,
   `http://localhost:8000/...` from a local backend) and `UNSTRACT_API_KEY`, then use
   **New run** like any other solution, or import result JSON files downloaded from
   Unstract (matched to documents by file name). Unstract serves each API result only
   once, so keep downloaded files if you may re-import.

Unstract's open-source edition returns no confidence (that needs paid plugins), so its
values show grounding only. Cost note: with chunk size 0, every prompt sends the whole
document, so a 10-field schema makes ~10 LLM calls per document.

What was verified while building this: the platform (v0.197.0) runs, the generated
project imports, and the API-deployment upload/poll/result flow works end to end. Because
Unstract's containers had no internet access in the development sandbox, its LLM calls
went to a stand-in; real Azure extraction through Unstract is untested.

### Schema suggestions

Besides the shared suggestion, solutions that can propose a schema do so independently,
from the Workspace (**Suggest with**, several at once) or from each solution's page:

| Source | How it suggests |
| --- | --- |
| Shared suggestion | Solution-neutral Azure OpenAI call over the exemplars' full text (strict JSON output) |
| Direct LLM | The same approach, run with the Direct LLM's own settings |
| NuExtract3 | NuExtract's native template-generation mode (guidance plus each exemplar), converted to a schema |
| LangExtract | Derived: a recipe is generated without a schema and its classes/attributes become fields (LangExtract has no schema feature; classes nest one level) |
| Unstract | Not available: the open-source edition has no prompt or schema generation |

Every suggestion is saved. Select two or more to compare them field by field (fields
only one source proposed and fields whose types differ are marked), then load any of
them into the editor as the draft for the agreed schema.

## Results, confidence and grounding

Every solution returns the same thing per document: a schema-shaped record plus a
flat list of values, each with its path (`attendees[3].role`), value, optional
evidence quote, optional confidence (0–1) and a source span. Solutions that return
values rather than spans (NuExtract3, Direct LLM, Unstract) are grounded by a shared
locator: exact match, then case/whitespace/quote-insensitive match, then fuzzy
alignment; dates also try written forms (`2019-10-22` → `22 October 2019`). Children
of an object item are searched near the item first, so repeated initials or names
resolve to the right occurrence.

- **Exact** – the quote or value appears verbatim. **Fuzzy** – it matched after
  normalisation or approximate alignment (dashed underline). **Not found** – often a
  paraphrase, a normalised value or a hallucination; review these first.
- A highlight or high confidence does not make a value correct. Confidence methods
  differ by solution and are not directly comparable; each run states its method.

Offsets are zero-based Unicode code-point offsets with an exclusive end, matching
the viewer's indexing. Highlighting is on the **extracted text**, not the PDF page
image; dashed separators mark page boundaries.

## Comparison

There is no ground truth, so the comparison page measures **agreement**: with a
reference run you choose, or with the other selected runs (consensus).

- Values are compared by type: dates and numbers must be equal, choices and yes/no
  match case-insensitively, text uses a fuzzy similarity. Arrays (including object
  arrays) are aligned item by item, so order does not matter, and scored like F1
  over the aligned items. A similarity of at least 0.8 counts as agreeing.
- Cells where every compared run is empty are not scored, so sparse fields do not
  inflate agreement.
- The scoreboard shows, per run, agreement, agree rate, fill rate, share of values
  located in the text, mean confidence, time per document, tokens and failures. The
  field matrix shows the same per field; the document matrix per document. Drilling
  into a document shows each run's values side by side, aligned object-array items,
  and the source highlight for any value.

Agreement is a proxy: consensus can reward shared mistakes. To approximate accuracy,
pick the run you have reviewed most carefully as the reference.

## Data, privacy and lifecycle

- Original local files are never modified or mounted. Uploaded copies, extracted
  text, schemas and runs are stored in the data directory (`workbench-data` volume in
  Docker, `data/` locally) so runs can be reviewed and compared later. Delete runs in
  the UI; remove all data with `docker compose down -v` or by deleting `data/`.
- Recipes and settings live in browser storage. Recipes **contain complete exemplar
  document texts**: review before exporting or sharing.
- **Schema suggestion sends complete exemplar texts to Azure. LangExtract sends
  document chunks with instructions and few-shot examples; Direct LLM sends whole
  documents (or large chunks) with the schema. Azure inference can incur costs.**
  NuExtract3 stays on your machine. Unstract sends documents to whatever adapters you
  configure in Unstract.
- Credentials are backend environment variables only; `.env` is ignored by Git and
  excluded from Docker build contexts. No key is returned by the API.
- The workbench is published only on `127.0.0.1`; the backend and NuExtract have no
  published host ports. Nginx and the API also reject requests for any other host
  name, so a web page that rebinds its own DNS name to `127.0.0.1` cannot read your
  documents or start runs. This is a trusted, single-user local prototype, not an
  authenticated production service.

Uploads are limited to 20 MiB, PDFs to 1,000 pages, extracted text to 500,000
characters and expanded DOCX contents to 100 MiB. Scanned PDFs without a text layer
are rejected (NuExtract3's page-image input is the route to try for those later).

## Adding a solution

1. Subclass `Solution` in `backend/app/solutions/<name>.py`: set the descriptive
   attributes (name, tagline, capabilities, confidence method, `route`), declare
   settings as `SettingSpec`s, implement `status()` and
   `extract(document, schema, settings, context) -> Outcome`, and optionally
   `prepare()` (cached pre-processing) and `preview()`.
2. Return a schema-shaped record whose leaves are `Leaf(value, quote, confidence,
   start, end)`; the job runner grounds and flattens it.
3. Register the class in `backend/app/solutions/__init__.py`. The landing page,
   navigation, generic settings form, runs and comparison pick it up automatically;
   set `external_url` for packaged tools with their own UI. Results produced outside
   the workbench can be collated with `POST /api/runs/import`.

## Development and checks

```powershell
Set-Location backend
& ..\.venv\Scripts\python.exe -m pytest -q
Set-Location ..\frontend
npm test
npm run build
```

Tests need no credentials or model servers: they cover parsing, the schema model,
grounding, record building and merging, the LangExtract schema mapping, the
comparison engine, the job runner and API, each solution's request/response
handling with mocked transports, and the React pages. API docs are served by the
backend at `/api/docs`; open them on the dev API (<http://127.0.0.1:8010/api/docs>). In
Docker, Nginx's Content-Security-Policy blocks the Swagger UI's CDN assets, so that page
stays blank there; the schema itself is at `/api/openapi.json`.

| Path | Purpose |
| --- | --- |
| `backend/app/schema.py` | Shared schema model, type coercion |
| `backend/app/grounding.py`, `results.py` | Locating values in text; records and runs |
| `backend/app/solutions/` | One module per solution plus the registry |
| `backend/app/jobs.py`, `storage.py` | Background runs; JSON stores |
| `backend/app/compare.py` | Cross-run alignment and agreement |
| `backend/app/schema_suggest.py` | Schema suggestion from exemplars |
| `frontend/src/workbench/` | Shell pages: landing, workspace, solutions, compare |
| `frontend/src/LangExtractStudio.tsx` | The LangExtract recipe studio |

## LangExtract recipe studio reference

This is the original LangExtract studio, now the **Recipe studio** tab of the LangExtract app.
Inside the workbench it lists the Workspace's processed documents (choosing a folder in the
studio still works and takes precedence) and, when an agreed schema exists, recipe
generation, extraction and optimisation follow its classes. Batch runs use the recipe that is
active here.

### Studio workflow

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

### Recipe optimization

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

### Extraction settings

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


### Recipe limits and LangExtract processing

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

## References

The interface is an original React implementation. The grounded source/findings interaction
was inspired by [RadExtract](https://google-radextract.hf.space/) and the editable
prompt/few-shot presentation by the [LangExtract playground](https://google-langextract.hf.space/).

- [LangExtract](https://github.com/google/langextract) ·
  [Introducing LangExtract](https://developers.googleblog.com/en/introducing-langextract-a-gemini-powered-information-extraction-library/)
- [NuExtract3 model card](https://huggingface.co/numind/NuExtract3) ·
  [GGUF builds](https://huggingface.co/numind/NuExtract3-GGUF) · [llama.cpp](https://github.com/ggml-org/llama.cpp) ·
  [vLLM](https://github.com/vllm-project/vllm)
- [Unstract](https://github.com/Zipstack/unstract)
- [Azure OpenAI endpoint and deployment configuration](https://learn.microsoft.com/en-us/azure/foundry-classic/openai/how-to/switching-endpoints)

Licences: LangExtract and NuExtract3 are Apache-2.0; Unstract is AGPL-3.0 and runs as a
separate application (not bundled). Dependency notices remain with their packages. This is
a prototype for comparing approaches, not a guarantee of extraction accuracy.
