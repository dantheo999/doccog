"""Where stored-text characters sit on the PDF page, so grounded values can be boxed on page images.

Every solution grounds values as character offsets in the stored text (pdfium text, cleaned). Each page's
stored text is aligned with pdfium's own character stream (the source of that text), which gives every
offset its character box. Boxes are fractions of the rendered (crop box, rotated) page, top-left origin."""
from collections import OrderedDict
from dataclasses import dataclass
from difflib import SequenceMatcher
import threading

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw

from app.documents import PDF_TRANSLATION, PDFIUM_LOCK, ParsedDocument

Box = tuple[float, float, float, float]  # left, top, right, bottom as fractions of the page


@dataclass
class PageLayout:
    width: float  # rendered size in points (after rotation)
    height: float
    boxes: dict[int, Box]  # document text offset -> character box


def _rotate(box: Box, rotation: int) -> Box:
    """A box on the unrotated page -> the same box on the page as rendered (rotated clockwise)."""
    left, top, right, bottom = box
    corners = [(left, top), (right, bottom)]
    if rotation == 90:
        corners = [(1 - v, u) for u, v in corners]
    elif rotation == 180:
        corners = [(1 - u, 1 - v) for u, v in corners]
    elif rotation == 270:
        corners = [(v, 1 - u) for u, v in corners]
    xs, ys = [u for u, _ in corners], [v for _, v in corners]
    return min(xs), min(ys), max(xs), max(ys)


def _page_layout(page: pdfium.PdfPage, text: str, offset: int) -> PageLayout:
    crop_left, crop_bottom, crop_right, crop_top = page.get_cropbox()
    crop_width, crop_height = (crop_right - crop_left) or 1.0, (crop_top - crop_bottom) or 1.0
    rotation = page.get_rotation() % 360
    textpage = page.get_textpage()
    try:
        count = textpage.count_chars()
        # pdfium's character stream, normalised the way the stored text was (same length, one entry per char).
        stream = "".join((chr(pdfium_raw.FPDFText_GetUnicode(textpage, index) or 0).translate(PDF_TRANSLATION) or " ")
                         .replace("\r", "\n") for index in range(count))
        boxes: dict[int, Box] = {}
        matcher = SequenceMatcher(None, text, stream, autojunk=False)
        for text_start, stream_start, size in matcher.get_matching_blocks():
            for step in range(size):
                if text[text_start + step].isspace():
                    continue
                left, bottom, right, top = textpage.get_charbox(stream_start + step)
                if right <= left or top <= bottom:
                    continue
                box = ((left - crop_left) / crop_width, (crop_top - top) / crop_height,
                       (right - crop_left) / crop_width, (crop_top - bottom) / crop_height)
                boxes[offset + text_start + step] = _rotate(box, rotation)
    finally:
        textpage.close()
    width, height = (crop_height, crop_width) if rotation in (90, 270) else (crop_width, crop_height)
    return PageLayout(width=width, height=height, boxes=boxes)


def document_layout(document: ParsedDocument, content: bytes) -> list[PageLayout]:
    """One layout per page, aligned with the document's stored text (page_starts)."""
    starts = document.page_starts or [0]
    with PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(content)
        try:
            layouts = []
            for index, page in enumerate(pdf):
                try:
                    if index >= len(starts):
                        break
                    end = starts[index + 1] - 2 if index + 1 < len(starts) else len(document.text)
                    layouts.append(_page_layout(page, document.text[starts[index]:end], starts[index]))
                finally:
                    page.close()
            return layouts
        finally:
            pdf.close()


def span_rects(layouts: list[PageLayout], start: int, end: int) -> list[dict]:
    """Rectangles covering text offsets [start, end): one per line run of characters on a page."""
    rects: list[dict] = []
    current: dict | None = None
    for page_index, layout in enumerate(layouts):
        for position in range(start, end):
            box = layout.boxes.get(position)
            if box is None:
                continue
            left, top, right, bottom = box
            middle = (top + bottom) / 2
            same_line = (current is not None and current["page"] == page_index
                         and current["y"] <= middle <= current["y"] + current["h"]
                         and left >= current["x"] + current["w"] - 0.02)
            if same_line:
                x2 = max(current["x"] + current["w"], right)
                y2 = max(current["y"] + current["h"], bottom)
                current["x"], current["y"] = min(current["x"], left), min(current["y"], top)
                current["w"], current["h"] = x2 - current["x"], y2 - current["y"]
            else:
                current = {"page": page_index, "x": left, "y": top, "w": right - left, "h": bottom - top}
                rects.append(current)
    return [{key: round(value, 5) if isinstance(value, float) else value for key, value in rect.items()}
            for rect in rects]


class LayoutCache:
    """Recent documents' layouts (aligning a document takes about half a second)."""

    def __init__(self, size: int = 8):
        self.size = size
        self._items: OrderedDict[str, list[PageLayout]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, document: ParsedDocument, content: bytes) -> list[PageLayout]:
        key = f"{document.id}:{document.sha256}"
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        layouts = document_layout(document, content)
        with self._lock:
            self._items[key] = layouts
            while len(self._items) > self.size:
                self._items.popitem(last=False)
        return layouts
