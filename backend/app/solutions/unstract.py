"""Unstract adapter: the packaged Unstract platform (Prompt Studio + API deployment) run by the user.

The landing page links to Unstract's own UI (external_url). This module generates the Prompt Studio project
from the agreed schema (preview) and collects results through an Unstract API deployment: upload the
original file, poll the status endpoint, map {prompt_key: value} onto the schema.

API deployment contract (Unstract v0.197.0 source, backend/api_v2):
  POST {api_url}  Authorization: Bearer <key>; multipart "files"; form timeout (-1 async, else <=300 s
       synchronous wait), include_metadata, include_metrics, optional llm_profile_id
       -> {"message": {"execution_status", "status_api": "/deployment/api/<org>/<api>/?execution_id=..",
                       "error", "result": [per-file items] | null}}
  GET  {status_api} -> 200 {"status": "COMPLETED", "message": [per-file items]} (results are served once;
       later GETs answer 406), 422 {"status": "PENDING"|"EXECUTING"|"ERROR", "message": ...}
"""
import json
import mimetypes
import os
import threading
import time
from urllib.parse import parse_qsl, urljoin, urlsplit, urlunsplit

import httpx

from app.azure import ServiceError
from app.documents import ParsedDocument
from app.results import Leaf, Usage, annotate_plain
from app.schema import ExtractionSchema
from app.solutions.base import Outcome, RunContext, SettingSpec, Solution
from app.unstract_export import (item_file_name, item_output, file_items, output_to_record, prompt_confidences,
                                 prompt_list_text, schema_to_project_export)


DEFAULT_UI_URL = "http://frontend.unstract.localhost"
RUNNING = {"PENDING", "EXECUTING", "QUEUED", "READY", "INITIATED", "STARTED", "RETRY", "RECEIVED"}
STATUS_CACHE_SECONDS = 30


def ui_url() -> str:
    return os.environ.get("UNSTRACT_UI_URL", "").strip() or DEFAULT_UI_URL


def env_api_url() -> str:
    return os.environ.get("UNSTRACT_API_URL", "").strip()


def api_key() -> str:
    return os.environ.get("UNSTRACT_API_KEY", "").strip()


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}".casefold() if parts.scheme and parts.netloc else ""


def allowed_origins() -> set[str]:
    """The API key is only ever sent to these origins (the configured Unstract hosts)."""
    origins = {_origin(env_api_url()), _origin(ui_url())}
    for extra in os.environ.get("UNSTRACT_ALLOWED_ORIGINS", "").split(","):
        origins.add(_origin(extra.strip()))
    return {origin for origin in origins if origin}


def api_name(url: str) -> str:
    parts = [part for part in urlsplit(url).path.split("/") if part]
    return parts[-1] if parts else ""


def _trim_metadata(value):
    """Keep raw results small: Unstract repeats the retrieved context (often the whole document) per prompt."""
    if isinstance(value, dict):
        trimmed = {}
        for key, item in value.items():
            if key in ("context", "extracted_text") and item:
                if isinstance(item, dict):
                    trimmed[key] = {name: f"<{sum(len(str(part)) for part in (text if isinstance(text, list) else [text]))} characters omitted>"
                                    for name, text in item.items()}
                else:
                    trimmed[key] = f"<{len(str(item))} characters omitted>"
            else:
                trimmed[key] = _trim_metadata(item)
        return trimmed
    if isinstance(value, list):
        return [_trim_metadata(item) for item in value]
    return value


def attach_confidence(value, confidence) -> None:
    """Copy a per-prompt confidence (number, or structure mirroring the value) onto Leaf objects."""
    def numbers(item) -> list[float]:
        if isinstance(item, bool):
            return []
        if isinstance(item, (int, float)):
            return [float(item)] if 0 <= item <= 1 else []
        if isinstance(item, list):
            return [number for part in item for number in numbers(part)]
        if isinstance(item, dict):
            return [number for part in item.values() for number in numbers(part)]
        return []

    def apply_all(item, score):
        if isinstance(item, Leaf):
            if item.value is not None:
                item.confidence = score
        elif isinstance(item, list):
            for part in item:
                apply_all(part, score)
        elif isinstance(item, dict):
            for part in item.values():
                apply_all(part, score)

    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        if 0 <= confidence <= 1:
            apply_all(value, float(confidence))
        return
    if isinstance(value, list) and isinstance(confidence, list) and len(value) == len(confidence):
        for item, score in zip(value, confidence):
            attach_confidence(item, score)
        return
    if isinstance(value, dict) and isinstance(confidence, dict) and set(confidence) & set(value):
        for key, item in value.items():
            if key in confidence:
                attach_confidence(item, confidence[key])
        return
    found = numbers(confidence)
    if found:
        apply_all(value, sum(found) / len(found))


class UnstractSolution(Solution):
    id = "unstract"
    name = "Unstract"
    tagline = "Open-source Prompt Studio platform, run as its own packaged app"
    description = ("Zipstack's Unstract (AGPL-3.0) is a complete document-extraction platform: Prompt Studio "
                   "prompts per field, a text extractor, LLM and vector-DB adapters, and API deployments. You "
                   "run it yourself and work in its UI; this workbench generates the Prompt Studio project from "
                   "the agreed schema and collects results through your API deployment for comparison.")
    technology = "Unstract (Zipstack) · Prompt Studio + API deployment"
    runtime = "local"
    capabilities = ["Packaged platform with its own UI", "Prompt Studio project generated from the schema",
                    "Results collected through an API deployment", "Imports results downloaded from Unstract"]
    confidence_method = ("Not provided by Unstract community edition (its confidence and highlight data need "
                         "LLMWhisperer plus enterprise plugins). If a response carries confidence data it is used "
                         "per prompt; otherwise grounding status is the quality signal.")
    route = "/apps/unstract"
    accent = "#ffb36b"
    document_concurrency = 2
    external_url = ui_url()

    def __init__(self):
        self._status_cache: tuple[float, tuple, dict] | None = None
        self._status_lock = threading.Lock()

    def info(self) -> dict:
        return {**super().info(), "external_url": ui_url()}

    def settings_spec(self) -> list[SettingSpec]:
        return [
            SettingSpec("api_url", "API deployment URL", "text", env_api_url(),
                        help="Unstract > API Deployments > your API's endpoint, e.g. "
                             "http://host.docker.internal:8000/deployment/api/<org_id>/<api_name>/. Must be on the "
                             "Unstract host configured in UNSTRACT_API_URL; the key comes from UNSTRACT_API_KEY."),
            SettingSpec("server_wait", "Synchronous wait (seconds)", "integer", 0, min=0, max=300,
                        help="0 queues the file and polls (recommended). Up to 300 asks Unstract to hold the "
                             "upload request open until the result is ready."),
            SettingSpec("poll_interval", "Poll interval (seconds)", "number", 3, min=1, max=60,
                        help="How often to check the execution status."),
            SettingSpec("max_wait", "Give up after (seconds)", "integer", 1800, min=30, max=7200,
                        help="Per document; Unstract keeps running and the result stays in its logs."),
            SettingSpec("include_metadata", "Include Unstract metadata", "boolean", False,
                        help="Adds per-prompt metadata (and confidence where your edition provides it) to the "
                             "raw result. Retrieved context is trimmed from what is stored."),
            SettingSpec("include_metrics", "Include Unstract metrics", "boolean", True,
                        help="Per-prompt timings and token usage."),
            SettingSpec("llm_profile_id", "LLM profile ID (optional)", "text", "",
                        help="Prompt Studio profile to use instead of the deployment's default."),
            SettingSpec("prompt_guidance", "Extra guidance for generated prompts", "text", "",
                        help="Only changes the generated Prompt Studio project (re-import it in Unstract). Runs "
                             "use whatever prompts the deployed project has."),
        ]

    # --- status ---------------------------------------------------------------------------

    def status(self) -> dict:
        try:
            return self._status()
        except Exception as error:  # noqa: BLE001 - status must never raise
            return {"ready": False, "detail": f"Unstract status check failed: {error}", "model": ""}

    def _status(self) -> dict:
        url, key = env_api_url(), api_key()
        if not url or not key:
            missing = " and ".join(name for name, value in (("UNSTRACT_API_URL", url), ("UNSTRACT_API_KEY", key))
                                   if not value)
            return {"ready": False, "model": api_name(url),
                    "detail": f"Set {missing} to the API deployment endpoint and key from Unstract "
                              f"(API Deployments). Open Unstract at {ui_url()} to build and deploy the project."}
        cache_key = (url, key)
        with self._status_lock:
            cached = self._status_cache
            if cached and cached[1] == cache_key and time.monotonic() - cached[0] < STATUS_CACHE_SECONDS:
                return cached[2]
        result = self._probe(url, key)
        with self._status_lock:
            self._status_cache = (time.monotonic(), cache_key, result)
        return result

    def _probe(self, url: str, key: str) -> dict:
        """A GET without execution_id is rejected by validation (400) only after the key check (401/403/404)."""
        name = api_name(url)
        try:
            response = httpx.get(url, headers={"Authorization": f"Bearer {key}"},
                                 timeout=httpx.Timeout(1.2, connect=1.0))
        except httpx.HTTPError as error:
            return {"ready": False, "model": name,
                    "detail": f"Unstract API deployment not reachable at {url} ({type(error).__name__})."}
        if response.status_code in (401, 403):
            return {"ready": False, "model": name, "detail": "Unstract rejected UNSTRACT_API_KEY for this deployment."}
        if response.status_code == 404:
            return {"ready": False, "model": name,
                    "detail": f"Unstract has no active API deployment at {url}."}
        if response.status_code >= 500:
            return {"ready": False, "model": name, "detail": f"Unstract answered HTTP {response.status_code}."}
        return {"ready": True, "model": name, "detail": f"Unstract API deployment '{name}' reachable."}

    def model_label(self, settings: dict) -> str:
        url = settings.get("api_url") or env_api_url()
        return f"Unstract · {api_name(url) or 'API deployment'}"

    # --- preview ----------------------------------------------------------------------------

    def preview(self, schema: ExtractionSchema, settings: dict) -> dict[str, str]:
        project = schema_to_project_export(schema, guidance=settings.get("prompt_guidance", ""))
        return {
            "Prompt Studio project (save as .json, then Prompt Studio > Import Project)": json.dumps(project, indent=2),
            "Prompts (to add by hand: prompt key, enforce type, prompt text)": prompt_list_text(project["prompts"]),
            "Profile": ("Chunk size 0 (whole document as context, no retrieval). Unstract still needs an LLM, an "
                        "embedding, a vector DB and a text extractor adapter on the profile. Enable highlight or "
                        "single-pass only if your edition has those plugins."),
        }

    # --- extraction -------------------------------------------------------------------------

    def validate_settings(self, settings: dict) -> dict:
        validated = super().validate_settings(settings)
        url = (validated.get("api_url") or "").strip() or env_api_url()
        if not url:
            raise ServiceError("Set the Unstract API deployment URL (or UNSTRACT_API_URL).", 422)
        if urlsplit(url).scheme not in ("http", "https") or not urlsplit(url).netloc:
            raise ServiceError("The Unstract API deployment URL must be an http(s) URL.", 422)
        if _origin(url) not in allowed_origins():
            raise ServiceError("The Unstract API deployment URL must be on the Unstract host configured in "
                               "UNSTRACT_API_URL (or listed in UNSTRACT_ALLOWED_ORIGINS).", 422)
        validated["api_url"] = url
        return validated

    def _client(self, settings: dict) -> httpx.Client:
        return httpx.Client(timeout=httpx.Timeout(settings.get("server_wait", 0) + 60, connect=10))

    def extract(self, document: ParsedDocument, schema: ExtractionSchema, settings: dict,
                context: RunContext) -> Outcome:
        key = api_key()
        if not key:
            raise ServiceError("Set UNSTRACT_API_KEY to the API deployment's key.", 503)
        url = (settings.get("api_url") or "").strip() or env_api_url()
        if not url or _origin(url) not in allowed_origins():
            raise ServiceError("The Unstract API deployment URL is missing or not on the configured host.", 422)
        content = context.store.source(document.id)
        notes: list[str] = []
        file_name = document.name
        if content is None:
            if document.name.lower().endswith(".pdf"):
                raise ServiceError("The original PDF is not stored. Upload the document again.", 410)
            content = document.text.encode()
            notes.append("Sent the stored text because the original file is not available.")
        mime = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
        server_wait = int(settings.get("server_wait", 0))
        form = {"timeout": str(server_wait if server_wait > 0 else -1),
                "include_metadata": "true" if settings.get("include_metadata") else "false",
                "include_metrics": "true" if settings.get("include_metrics", True) else "false"}
        if (settings.get("llm_profile_id") or "").strip():
            form["llm_profile_id"] = settings["llm_profile_id"].strip()
        headers = {"Authorization": f"Bearer {key}"}
        context.check()
        with self._client(settings) as client:
            try:
                response = client.post(url, headers=headers, data=form, files={"files": (file_name, content, mime)})
            except httpx.HTTPError as error:
                raise ServiceError(f"Could not reach Unstract at {url}: {type(error).__name__}.") from error
            body = self._json(response)
            if response.status_code != 200:
                raise ServiceError(self._error_text(response.status_code, body), self._status_for(response.status_code))
            message = body.get("message") if isinstance(body, dict) else None
            if not isinstance(message, dict):
                raise ServiceError("Unstract returned an unexpected response to the upload.")
            execution_id = message.get("execution_id")
            status = str(message.get("execution_status") or "").upper()
            results = message.get("result")
            final = body
            if status == "ERROR" or (message.get("error") and not results):
                raise ServiceError(f"Unstract execution failed: {message.get('error') or 'unknown error'}")
            if not (status == "COMPLETED" and isinstance(results, list)):
                status_url = self._status_url(url, message.get("status_api"), execution_id)
                final, results = self._poll(client, status_url, headers, settings, context)
        item = self._pick(results or [], file_name)
        output, metadata, error = item_output(item)
        if error:
            raise ServiceError(f"Unstract could not process this file: {error}")
        plain, mapping_notes = output_to_record(output, schema)
        record = annotate_plain(plain, schema.fields)
        confidences = prompt_confidences(metadata, [field.name for field in schema.fields])
        for name, confidence in confidences.items():
            attach_confidence(record.get(name), confidence)
        usage = self._usage(item)
        raw = {"execution_id": execution_id, "api_deployment": api_name(url),
               "request": {**form, "file": file_name}, "item": _trim_metadata(item),
               "status": final.get("status") if isinstance(final, dict) else None}
        if settings.get("include_metadata") and not confidences:
            notes.append("Unstract returned no confidence data (community edition has none).")
        return Outcome(record=record, usage=usage, notes=list(dict.fromkeys(notes + mapping_notes)), raw=raw)

    @staticmethod
    def _json(response: httpx.Response):
        try:
            return response.json()
        except ValueError:
            return {"detail": response.text[:500]}

    @staticmethod
    def _status_for(code: int) -> int:
        return 429 if code == 429 else 502

    @staticmethod
    def _error_text(code: int, body) -> str:
        detail = ""
        if isinstance(body, dict):
            message = body.get("message")
            if isinstance(message, dict):
                detail = message.get("error") or message.get("execution_status") or ""
            elif isinstance(message, str):
                detail = message
            detail = detail or body.get("detail") or body.get("error_message") or ""
            errors = body.get("errors")
            if not detail and isinstance(errors, list) and errors:
                first = errors[0]
                detail = first.get("detail") if isinstance(first, dict) else str(first)
        if code in (401, 403):
            return f"Unstract rejected the API key{': ' + str(detail) if detail else ''}."
        if code == 404:
            return f"Unstract API deployment not found or inactive{': ' + str(detail) if detail else ''}."
        if code == 429:
            return f"Unstract rate limit reached{': ' + str(detail) if detail else ''}. Lower the concurrency or retry."
        return f"Unstract returned HTTP {code}{': ' + str(detail)[:500] if detail else ''}."

    @staticmethod
    def _status_url(api_url: str, status_api: str | None, execution_id: str | None) -> str:
        if status_api:
            joined = urljoin(api_url, status_api)
            if _origin(joined) == _origin(api_url):
                return joined
        if not execution_id:
            raise ServiceError("Unstract did not return an execution id to poll.")
        return f"{api_url.split('?')[0]}?execution_id={execution_id}"

    def _poll(self, client: httpx.Client, status_url: str, headers: dict, settings: dict,
              context: RunContext) -> tuple[dict, list]:
        # httpx replaces a URL's query string when params= is given, so carry execution_id over explicitly.
        parts = urlsplit(status_url)
        base = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
        params = {**dict(parse_qsl(parts.query)),
                  "include_metadata": "true" if settings.get("include_metadata") else "false",
                  "include_metrics": "true" if settings.get("include_metrics", True) else "false"}
        deadline = time.monotonic() + float(settings.get("max_wait", 1800))
        interval = float(settings.get("poll_interval", 3))
        failures = 0
        execution = params.get("execution_id", "")
        while True:
            self._sleep(interval, context, deadline)
            if time.monotonic() >= deadline:
                raise ServiceError(f"Unstract did not finish within {settings.get('max_wait', 1800)} s "
                                   f"(execution {execution}). It may still complete in Unstract.", 504)
            try:
                response = client.get(base, headers=headers, params=params)
            except httpx.HTTPError as error:
                failures += 1
                if failures >= 5:
                    raise ServiceError(f"Lost contact with Unstract while polling: {type(error).__name__}.") from error
                continue
            body = self._json(response)
            status = str(body.get("status") or "").upper() if isinstance(body, dict) else ""
            if response.status_code == 200 and status == "COMPLETED":
                message = body.get("message")
                if not isinstance(message, list):
                    raise ServiceError("Unstract finished but returned no per-file results.")
                return body, message
            if response.status_code == 406:
                raise ServiceError(f"Unstract already handed out the result of execution {execution} "
                                   "(results are served once). Re-run the document or import the result file.")
            if status == "ERROR":
                message = body.get("message")
                raise ServiceError(f"Unstract execution failed: {self._message_error(message)}")
            if status == "STOPPED":
                raise ServiceError("The Unstract execution was stopped.")
            if status in RUNNING or (response.status_code == 422 and not status):
                failures = 0
                continue
            if response.status_code >= 500 or response.status_code == 429:
                failures += 1
                if failures >= 5:
                    raise ServiceError(self._error_text(response.status_code, body))
                continue
            raise ServiceError(self._error_text(response.status_code, body), self._status_for(response.status_code))

    @staticmethod
    def _message_error(message) -> str:
        if isinstance(message, list):
            errors = [item.get("error") for item in message if isinstance(item, dict) and item.get("error")]
            return "; ".join(map(str, errors)) or "unknown error"
        if isinstance(message, dict):
            return str(message.get("error") or message)
        return str(message or "unknown error")

    @staticmethod
    def _sleep(seconds: float, context: RunContext, deadline: float):
        end = min(time.monotonic() + seconds, deadline)
        while True:
            context.check()
            remaining = end - time.monotonic()
            if remaining <= 0:
                return
            context.cancel.wait(min(0.5, remaining))

    @staticmethod
    def _pick(results: list, file_name: str) -> dict:
        items = file_items(results)
        if not items:
            raise ServiceError("Unstract returned no result for this file.")
        for item in items:
            if (item_file_name(item) or "") == file_name:
                return item
        for item in items:
            if (item_file_name(item) or "").casefold() == file_name.casefold():
                return item
        return items[0]

    @staticmethod
    def _usage(item: dict) -> Usage:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        usage = metadata.get("usage") if isinstance(metadata.get("usage"), dict) else {}

        def number(key):
            value = usage.get(key)
            return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0
        return Usage(calls=1, prompt_tokens=number("prompt_tokens"), completion_tokens=number("completion_tokens"))
