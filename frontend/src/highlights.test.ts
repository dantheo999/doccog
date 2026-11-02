import { describe, expect, it } from "vitest";
import { buildSegments } from "./highlights";

describe("source highlighting", () => {
  it("uses Unicode code-point offsets rather than UTF-16 positions", () => {
    expect(buildSegments("A😀B", [{ id: "x", start: 1, end: 2 }])).toEqual([
      { text: "A", ids: [] }, { text: "😀", ids: ["x"] }, { text: "B", ids: [] },
    ]);
  });

  it("splits overlapping spans without repeating or losing text", () => {
    expect(buildSegments("Alice signs", [
      { id: "person", start: 0, end: 5 }, { id: "event", start: 0, end: 11 },
    ])).toEqual([
      { text: "Alice", ids: ["person", "event"] }, { text: " signs", ids: ["event"] },
    ]);
  });

  it("starts a segment at every extra cut, such as a page start", () => {
    expect(buildSegments("Page oneAlice", [{ id: "a", start: 6, end: 13 }], [0, 8, 13, 40])).toEqual([
      { text: "Page o", ids: [] }, { text: "ne", ids: ["a"] }, { text: "Alice", ids: ["a"] },
    ]);
    expect(buildSegments("ab", [], [1]).map(segment => segment.text)).toEqual(["a", "b"]);
  });

  it("ignores invalid and ungrounded intervals", () => {
    expect(buildSegments("Alice", [
      { id: "a", start: null, end: null }, { id: "b", start: 0, end: 9 },
      { id: "c", start: 3, end: 1 },
    ])).toEqual([{ text: "Alice", ids: [] }]);
  });
});
