from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path, PurePath
import re
import threading
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table
from docx.text.paragraph import Paragraph
from lxml.etree import XMLSyntaxError
import pypdfium2 as pdfium
from pypdf import PdfReader
from pypdf.errors import PdfReadError, PyPdfError


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_CHARACTERS = 500_000
PARSER_VERSION = "2"
# pdfium marks generated line-end hyphens as \x02; Symbol-font bullets use a private-use code point.
PDF_TRANSLATION = str.maketrans({"\x02": "-", "\uf0b7": "\u2022", "\uf0a7": "\u25aa", "\x00": ""})


class DocumentError(ValueError):
    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


@dataclass
class ParsedDocument:
    name: str
    text: str
    pages: int | None = None
    id: str = ""
    page_starts: list[int] = field(default_factory=list)
    sha256: str = ""
    file_sha256: str = ""
    parser: str = ""
    created_at: str = ""

    def __post_init__(self):
        if not self.sha256:
            self.sha256 = hashlib.sha256(self.text.encode()).hexdigest()

    def response(self) -> dict:
        return {**asdict(self), "characters": len(self.text)}

    def summary(self) -> dict:
        return {key: value for key, value in self.response().items() if key != "text"}


def clean_pdf_page(text: str) -> str:
    text = text.translate(PDF_TRANSLATION).replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[ \t]+\n", "\n", text).strip("\n")


def pdf_page_texts(content: bytes) -> list[str]:
    """Selectable text per page; pdfium keeps words intact where pypdf splits them."""
    document = pdfium.PdfDocument(content)
    try:
        pages = []
        for page in document:
            textpage = page.get_textpage()
            try:
                pages.append(clean_pdf_page(textpage.get_text_bounded()))
            finally:
                textpage.close()
                page.close()
        return pages
    finally:
        document.close()


def render_pdf_pages(content: bytes, dpi: int = 150) -> list[bytes]:
    document = pdfium.PdfDocument(content)
    try:
        images = []
        for page in document:
            bitmap = page.render(scale=dpi / 72)
            buffer = BytesIO()
            bitmap.to_pil().convert("RGB").save(buffer, format="PNG", optimize=True)
            images.append(buffer.getvalue())
            page.close()
        return images
    finally:
        document.close()


def parse_document(name: str, content: bytes) -> ParsedDocument:
    if len(content) > MAX_UPLOAD_BYTES:
        raise DocumentError("Document exceeds the 20 MiB upload limit.", 413)
    if not content:
        raise DocumentError("This document is empty.")
    name = name.replace("\\", "/").rsplit("/", 1)[-1][:1_024]
    suffix = PurePath(name).suffix.lower()
    pages = None
    page_starts: list[int] = []
    try:
        if suffix in (".txt", ".md"):
            encoding = "utf-16" if content.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
            text = content.decode(encoding)
            if "\x00" in text:
                raise DocumentError("Text files must contain UTF-8 or BOM-marked UTF-16 text, not binary data.")
        elif suffix == ".pdf":
            reader = PdfReader(BytesIO(content))
            if reader.is_encrypted:
                raise DocumentError("Encrypted PDFs are not supported. Choose an unprotected copy.")
            if len(reader.pages) > 1_000:
                raise DocumentError("PDF exceeds the 1,000-page limit.")
            try:
                parts = pdf_page_texts(content)
            except pdfium.PdfiumError:
                parts = [page.extract_text() or "" for page in reader.pages]
            if sum(len(part) + 2 for part in parts) > MAX_CHARACTERS:
                raise DocumentError("Document exceeds the 500,000-character limit.")
            text = "\n\n".join(parts)
            pages = len(parts)
            page_starts, offset = [], 0
            for part in parts:
                page_starts.append(offset)
                offset += len(part) + 2
            if not text.strip():
                raise DocumentError("This PDF contains no selectable text. Scanned PDFs need OCR, which this demo does not include.")
        elif suffix == ".docx":
            with ZipFile(BytesIO(content)) as archive:
                if sum(item.file_size for item in archive.infolist()) > 100 * 1024 * 1024:
                    raise DocumentError("DOCX expanded contents exceed the 100 MiB safety limit.")
            document = Document(BytesIO(content))
            parts = []
            for block in document.iter_inner_content():
                if isinstance(block, Paragraph):
                    parts.append(block.text)
                elif isinstance(block, Table):
                    parts.extend("\t".join(cell.text for cell in row.cells) for row in block.rows)
            text = "\n".join(parts)
        else:
            raise DocumentError("Unsupported document. Choose PDF, DOCX, TXT or Markdown (.md).")
    except (UnicodeError, BadZipFile, PackageNotFoundError, XMLSyntaxError, PyPdfError,
            PdfReadError, KeyError, EOFError, ValueError) as error:
        if isinstance(error, DocumentError):
            raise
        raise DocumentError(f"Could not read this {suffix or 'document'} file. Check that it is valid and not encrypted.") from error
    if not text.strip():
        raise DocumentError("This document contains no readable text.")
    if len(text) > MAX_CHARACTERS:
        raise DocumentError("Document exceeds the 500,000-character limit.")
    return ParsedDocument(name=name, text=text, pages=pages, page_starts=page_starts,
                          file_sha256=hashlib.sha256(content).hexdigest(),
                          parser=f"{suffix.lstrip('.') or 'text'}-v{PARSER_VERSION}",
                          created_at=datetime.now(timezone.utc).isoformat())


def atomic_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


class DocumentStore:
    """Processed documents, content-addressed so re-selecting a folder reuses earlier processing.

    With a root directory, text and uploaded bytes persist across restarts (needed to review
    and compare runs later). Without one (tests), documents live only in memory.
    """

    def __init__(self, root: Path | None = None, max_cached: int = 100):
        self.root = root
        self.max_cached = max_cached
        self._cache: OrderedDict[str, ParsedDocument] = OrderedDict()
        self._sources: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def _path(self, document_id: str, suffix: str) -> Path:
        assert self.root is not None
        return self.root / "documents" / f"{document_id}.{suffix}"

    def _remember(self, document: ParsedDocument):
        self._cache[document.id] = document
        self._cache.move_to_end(document.id)
        while len(self._cache) > self.max_cached:
            evicted, _ = self._cache.popitem(last=False)
            if self.root is not None:
                self._sources.pop(evicted, None)

    def add(self, document: ParsedDocument, content: bytes | None = None) -> ParsedDocument:
        seed = (document.file_sha256 or hashlib.sha256(document.text.encode()).hexdigest()) + PARSER_VERSION
        document.id = hashlib.sha256(f"{seed}:{document.name}".encode()).hexdigest()[:24]
        with self._lock:
            if self.root is not None:
                payload = json.dumps(asdict(document), ensure_ascii=False).encode()
                atomic_write(self._path(document.id, "json"), payload)
                if content is not None:
                    atomic_write(self._path(document.id, "source"), content)
            elif content is not None:
                self._sources[document.id] = content
            self._remember(document)
            return document

    def get(self, document_id: str) -> ParsedDocument:
        if not re.fullmatch(r"[0-9a-f]{24}", document_id or ""):
            raise DocumentError("Document expired or was removed. Select it again to upload a fresh copy.", 410)
        with self._lock:
            document = self._cache.get(document_id)
            if document is None and self.root is not None:
                try:
                    data = json.loads(self._path(document_id, "json").read_bytes())
                    document = ParsedDocument(**data)
                except FileNotFoundError:
                    document = None
                except (ValueError, TypeError) as error:
                    raise DocumentError("Stored document data is unreadable. Select the document again.", 410) from error
            if document is None:
                raise DocumentError("Document expired or was removed. Select it again to upload a fresh copy.", 410)
            self._remember(document)
            return document

    def source(self, document_id: str) -> bytes | None:
        self.get(document_id)
        if self.root is None:
            return self._sources.get(document_id)
        try:
            return self._path(document_id, "source").read_bytes()
        except FileNotFoundError:
            return None

    def page_images(self, document_id: str, dpi: int = 150) -> list[bytes]:
        """Rendered PDF pages (cached), for vision-capable solutions."""
        document = self.get(document_id)
        if not document.name.lower().endswith(".pdf"):
            return []
        if self.root is not None:
            folder = self.root / "documents" / f"{document_id}-pages-{dpi}"
            if folder.is_dir():
                files = sorted(folder.glob("*.png"))
                if len(files) == document.pages:
                    return [file.read_bytes() for file in files]
        content = self.source(document_id)
        if content is None:
            raise DocumentError("The original PDF is not stored. Select the document again to render its pages.", 410)
        images = render_pdf_pages(content, dpi)
        if self.root is not None:
            for index, image in enumerate(images):
                atomic_write(folder / f"{index:04d}.png", image)
        return images

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._sources.clear()
            if self.root is not None:
                import shutil
                shutil.rmtree(self.root / "documents", ignore_errors=True)
