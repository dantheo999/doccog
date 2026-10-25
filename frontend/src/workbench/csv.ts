/** CSV export helpers for the comparison downloads. */

export function csvCell(value: unknown): string {
  let text = value === null || value === undefined ? "" : String(value);
  // Extracted text could start a spreadsheet formula (=, +, -, @, tab, CR); keep it as text. Numbers stay numbers.
  if (typeof value === "string" && /^[=+\-@\t\r]/.test(text) && !/^[-+]?\d+(\.\d+)?$/.test(text)) text = `'${text}`;
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export const toCsv = (rows: unknown[][]) => rows.map(row => row.map(csvCell).join(",")).join("\r\n");
