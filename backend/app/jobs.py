"""Background execution of extraction runs, persisted after every document."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import logging
import threading
import time

from app.azure import ServiceError
from app.cognition import CognitionError
from app.documents import DocumentError, DocumentStore
from app.results import Run, finalise
from app.solutions.base import RunContext, Solution
from app.storage import RunStore, now


logger = logging.getLogger(__name__)
HEARTBEAT_SECONDS = 15
STALE_SECONDS = 90  # a run whose heartbeat is older than this has lost its runner


def heartbeat_age(run: Run) -> float:
    stamp = run.heartbeat_at or run.started_at or run.created_at
    try:
        then = datetime.fromisoformat(stamp)
    except (TypeError, ValueError):
        return float("inf")
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - then).total_seconds()


class JobRunner:
    def __init__(self, runs: RunStore, store: DocumentStore):
        self.runs = runs
        self.store = store
        self._contexts: dict[str, RunContext] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def recover(self):
        """Close runs whose runner is gone (backend restart or crash): queued/running, not owned by
        this process, and with a stale heartbeat. Runs another live process owns keep running."""
        for summary in self.runs.list():
            if summary["status"] in ("queued", "running"):
                self.reconcile(self.runs.get(summary["id"]))

    def reconcile(self, run: Run) -> Run:
        if (run.status not in ("queued", "running") or self.active(run.id)
                or heartbeat_age(run) < STALE_SECONDS):
            return run
        for document in run.documents:
            if document.status in ("queued", "running"):
                document.status = "cancelled"
                document.error = "Interrupted: the backend stopped while this run was in progress."
        run.status = "error"
        run.error = "Interrupted: the backend stopped while this run was in progress. Start a new run."
        run.finished_at = now()
        self.runs.save(run)
        return run

    def active(self, run_id: str) -> bool:
        with self._guard:
            return run_id in self._contexts

    def start(self, run: Run, solution: Solution, wait: bool = False) -> threading.Thread:
        context = RunContext(store=self.store)
        run.heartbeat_at = now()
        with self._guard:
            self._contexts[run.id] = context
            self._locks[run.id] = threading.Lock()
        self.runs.save(run)
        thread = threading.Thread(target=self._execute, args=(run, solution, context),
                                  name=f"run-{run.id}", daemon=True)
        thread.start()
        if wait:
            thread.join()
        return thread

    def cancel(self, run_id: str) -> bool:
        with self._guard:
            context = self._contexts.get(run_id)
        if context is None:
            return False
        context.cancel.set()
        return True

    def _save(self, run: Run):
        with self._locks[run.id]:
            self.runs.save(run)

    def _beat(self, run: Run, done: threading.Event):
        while not done.wait(HEARTBEAT_SECONDS):
            run.heartbeat_at = now()
            self._save(run)

    def _execute(self, run: Run, solution: Solution, context: RunContext):
        run.status = "running"
        run.started_at = run.heartbeat_at = now()
        self._save(run)
        finished = threading.Event()
        threading.Thread(target=self._beat, args=(run, finished), name=f"beat-{run.id}", daemon=True).start()
        try:
            with ThreadPoolExecutor(max_workers=max(1, solution.document_concurrency)) as pool:
                list(pool.map(lambda index: self._document(run, index, solution, context),
                              range(len(run.documents))))
            if context.cancel.is_set():
                run.status = "cancelled"
            elif all(document.status == "error" for document in run.documents):
                run.status = "error"
                run.error = run.documents[0].error if run.documents else "No documents."
            else:
                run.status = "done"
        except Exception:  # noqa: BLE001 - a crashed runner must still close the run
            logger.exception("Run %s crashed", run.id)
            run.status = "error"
            run.error = "The run stopped unexpectedly. Check the backend logs."
        finally:
            finished.set()
            run.finished_at = now()
            self._save(run)
            with self._guard:
                self._contexts.pop(run.id, None)

    def _document(self, run: Run, index: int, solution: Solution, context: RunContext):
        result = run.documents[index]
        if context.cancel.is_set():
            result.status = "cancelled"
            self._save(run)
            return
        result.status = "running"
        result.started_at = now()
        self._save(run)
        started = time.monotonic()
        try:
            document = self.store.get(result.document_id)
            if document.sha256 != result.document_sha256:
                raise ServiceError("The stored document text changed since this run was created.", 409)
            notes = solution.prepare(document, run.settings, context)
            context.check()
            outcome = solution.extract(document, run.extraction_schema, run.settings, context)
            data, values = finalise(document.text, run.extraction_schema, outcome.record)
            ungrounded = sum(value.grounding == "none" for value in values)
            result.data = data
            result.values = values
            result.usage = outcome.usage
            result.raw = outcome.raw
            result.notes = notes + outcome.notes + (
                [f"{ungrounded} of {len(values)} values could not be located in the text."] if ungrounded else [])
            result.status = "done"
        except (ServiceError, CognitionError, DocumentError) as error:
            cancelled = context.cancel.is_set()
            result.status = "cancelled" if cancelled else "error"
            result.error = "Run cancelled." if cancelled else str(error)
        except Exception:  # noqa: BLE001 - keep other documents running
            logger.exception("Run %s failed on document %s", run.id, result.document_name)
            result.status = "error"
            result.error = "Unexpected error while processing this document. Check the backend logs."
        finally:
            result.elapsed_seconds = round(time.monotonic() - started, 2)
            result.finished_at = now()
            self._save(run)
