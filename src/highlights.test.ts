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

  it("ignores invalid and ungrounded intervals", () => {
    expect(buildSegments("Alice", [
      { id: "a", start: null, end: null }, { id: "b", start: 0, end: 9 },
      { id: "c", start: 3, end: 1 },
    ])).toEqual([{ text: "Alice", ids: [] }]);
  });
});
