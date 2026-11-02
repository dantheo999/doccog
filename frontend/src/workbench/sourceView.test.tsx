import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { beforeEach, expect, it, vi } from "vitest";
import SourceView from "./components/SourceView";
import type { FieldValue, StoredDocument } from "./types";

const pdf: StoredDocument = { id: "a".repeat(24), name: "minutes.pdf", text: "Board meeting\n\nAlice chaired.",
  characters: 29, pages: 2, page_starts: [0, 15], sha256: "x" };
const value = (field: string, start: number | null, end: number | null, path: (string | number)[] = [field]): FieldValue =>
  ({ path, field, value: "v", quote: null, confidence: null, grounding: start === null ? "none" : "exact", start, end });
const values = [value("title", 0, 5), value("chair", 15, 20), value("venue", null, null)];
let requests: { url: string; body: unknown }[] = [];

beforeEach(() => {
  localStorage.clear();
  requests = [];
  vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
    const body = JSON.parse(String(init?.body));
    requests.push({ url, body });
    return Response.json({ pages: [{ width: 612, height: 792 }, { width: 612, height: 792 }],
      boxes: { title: [{ page: 0, x: 0.1, y: 0.1, w: 0.2, h: 0.02 }], chair: [] } });
  }));
});

function Harness({ document = pdf }: { document?: StoredDocument }) {
  const [selected, setSelected] = useState<string | null>(null);
  return <><SourceView document={document} values={values} selected={selected} onSelect={setSelected} />
    <output data-testid="selected">{selected ?? ""}</output></>;
}

it("shows PDF pages with a box per located value, and selects a value from its box", async () => {
  render(<Harness />);
  const box = await screen.findByRole("button", { name: "title: v" });
  expect(box).toHaveStyle({ left: "10.000%", top: "10.000%", width: "20.000%" });
  expect(screen.getAllByRole("img").map(image => image.getAttribute("src")))
    .toEqual([`/api/documents/${pdf.id}/pages/0.png`, `/api/documents/${pdf.id}/pages/1.png`]);
  // Only located values are sent; one could not be placed on the page and is reported.
  expect(requests[0]).toEqual({ url: `/api/documents/${pdf.id}/boxes`, body: { spans: [
    { id: "title", start: 0, end: 5 }, { id: "chair", start: 15, end: 20 }] } });
  expect(screen.getByText(/1 located value could not be placed/)).toBeInTheDocument();
  fireEvent.click(box);
  expect(screen.getByTestId("selected")).toHaveTextContent("title");
  await waitFor(() => expect(box).toHaveAttribute("aria-pressed", "true"));
});

it("switches to the text view and remembers it; non-PDF documents only have text", async () => {
  const first = render(<Harness />);
  fireEvent.click(await screen.findByRole("button", { name: "Text" }));
  expect(screen.getByLabelText("Document source text")).toHaveTextContent("Alice chaired.");
  first.unmount();
  const second = render(<Harness />);
  expect(screen.getByLabelText("Document source text")).toBeInTheDocument();
  second.unmount();
  localStorage.clear();
  render(<Harness document={{ ...pdf, name: "notes.txt", pages: null, page_starts: [] }} />);
  expect(screen.queryByRole("button", { name: "PDF pages" })).toBeNull();
  expect(screen.getByLabelText("Document source text")).toBeInTheDocument();
});
