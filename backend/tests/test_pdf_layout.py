"""Boxes for stored-text spans on the PDF page: a box must enclose exactly the text it was made for."""
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from fastapi.testclient import TestClient

from app.documents import parse_document
from app.main import app, store
from app.pdf_layout import _rotate, document_layout, span_rects


DOCUMENTS = Path(__file__).resolve().parents[2] / "documents"
MINUTES = DOCUMENTS / "20191022_RSH_Board_Minutes.pdf"
pytestmark = pytest.mark.skipif(not MINUTES.exists(), reason="sample minutes not present")


def text_in_rect(content: bytes, rect: dict) -> str:
    """What pdfium itself reads inside a rect given as page fractions."""
    pdf = pdfium.PdfDocument(content)
    try:
        page = pdf[rect["page"]]
        left, bottom, right, top = page.get_cropbox()
        width, height = right - left, top - bottom
        textpage = page.get_textpage()
        try:
            return textpage.get_text_bounded(
                left + rect["x"] * width, top - (rect["y"] + rect["h"]) * height,
                left + (rect["x"] + rect["w"]) * width, top - rect["y"] * height)
        finally:
            textpage.close()
            page.close()
    finally:
        pdf.close()


def test_a_word_span_boxes_exactly_that_word_on_its_page():
    content = MINUTES.read_bytes()
    document = parse_document(MINUTES.name, content)
    layouts = document_layout(document, content)
    assert len(layouts) == document.pages
    last_page_start = document.page_starts[-1]
    for word in ("Minutes", "Board"):
        start = document.text.index(word, last_page_start) if word in document.text[last_page_start:] \
            else document.text.index(word)
        rects = span_rects(layouts, start, start + len(word))
        assert len(rects) == 1
        assert 0 <= rects[0]["x"] < 1 and 0 <= rects[0]["y"] < 1 and rects[0]["w"] > 0 and rects[0]["h"] > 0
        expected_page = max(index for index, offset in enumerate(document.page_starts) if offset <= start)
        assert rects[0]["page"] == expected_page
        assert text_in_rect(content, rects[0]).strip() == word


def test_a_span_across_lines_gets_one_rect_per_line_and_unknown_offsets_none():
    content = MINUTES.read_bytes()
    document = parse_document(MINUTES.name, content)
    layouts = document_layout(document, content)
    newline = document.text.index("\n", 200)
    rects = span_rects(layouts, newline - 5, newline + 6)
    assert len(rects) == 2 and rects[1]["y"] > rects[0]["y"]
    assert span_rects(layouts, len(document.text) + 10, len(document.text) + 20) == []


def test_rotation_maps_page_fractions_clockwise():
    box = (0.1, 0.2, 0.3, 0.4)
    assert _rotate(box, 0) == box
    assert _rotate(box, 180) == pytest.approx((0.7, 0.6, 0.9, 0.8))
    assert _rotate(box, 90) == pytest.approx((0.6, 0.1, 0.8, 0.3))
    assert _rotate(box, 270) == pytest.approx((0.2, 0.7, 0.4, 0.9))


def test_page_and_box_endpoints():
    client = TestClient(app)
    content = MINUTES.read_bytes()
    uploaded = client.post("/api/documents", files={"file": (MINUTES.name, content, "application/pdf")}).json()
    start = uploaded["text"].index("Board")
    response = client.post(f"/api/documents/{uploaded['id']}/boxes",
                           json={"spans": [{"id": "a", "start": start, "end": start + 5}]})
    assert response.status_code == 200
    body = response.json()
    assert len(body["pages"]) == uploaded["pages"] and body["pages"][0]["width"] > 0
    assert len(body["boxes"]["a"]) == 1
    image = client.get(f"/api/documents/{uploaded['id']}/pages/0.png")
    assert image.status_code == 200 and image.headers["content-type"] == "image/png"
    assert image.content.startswith(b"\x89PNG")
    assert client.get(f"/api/documents/{uploaded['id']}/pages/{uploaded['pages']}.png").status_code == 404
    text = client.post("/api/documents", files={"file": ("a.txt", b"Alice signed.", "text/plain")}).json()
    assert client.post(f"/api/documents/{text['id']}/boxes", json={"spans": []}).status_code == 422
    assert store.source(uploaded["id"]) == content
