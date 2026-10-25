from collections import OrderedDict
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import PurePath
import threading
import time
from typing import Callable
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table
from docx.text.paragraph import Paragraph
from lxml.etree import XMLSyntaxError
from pypdf import PdfReader
from pypdf.errors import PdfReadError, PyPdfError


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_CHARACTERS = 500_000


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

    def response(self) -> dict:
        return {**asdict(self), "characters": len(self.text)}


def parse_document(name: str, content: bytes) -> ParsedDocument:
    if len(content) > MAX_UPLOAD_BYTES:
        raise DocumentError("Document exceeds the 20 MiB upload limit.", 413)
    if not content:
        raise DocumentError("This document is empty.")
    name = name.replace("\\", "/").rsplit("/", 1)[-1][:1_024]
    suffix = PurePath(name).suffix.lower()
    pages = None
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
            parts = []
            characters = 0
            for page in reader.pages:
                part = page.extract_text() or ""
                characters += len(part) + 2
                if characters > MAX_CHARACTERS:
                    raise DocumentError("Document exceeds the 500,000-character limit.")
                parts.append(part)
            text = "\n\n".join(parts)
            pages = len(parts)
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
    return ParsedDocument(name=name, text=text, pages=pages)


class DocumentStore:
    def __init__(self, ttl: int = 1_800, max_documents: int = 20,
                 clock: Callable[[], float] = time.monotonic):
        self.ttl = ttl
        self.max_documents = max_documents
        self.clock = clock
        self._documents: OrderedDict[str, tuple[float, ParsedDocument]] = OrderedDict()
        self._lock = threading.Lock()

    def _expire(self):
        now = self.clock()
        for key, (created, _) in list(self._documents.items()):
            if now - created >= self.ttl:
                del self._documents[key]

    def add(self, document: ParsedDocument) -> ParsedDocument:
        with self._lock:
            self._expire()
            while len(self._documents) >= self.max_documents:
                self._documents.popitem(last=False)
            document.id = str(uuid4())
            self._documents[document.id] = (self.clock(), document)
            return document

    def get(self, document_id: str) -> ParsedDocument:
        with self._lock:
            self._expire()
            item = self._documents.get(document_id)
            if item is None:
                raise DocumentError("Document expired or was removed. Select it again to upload a fresh copy.", 410)
            return item[1]
