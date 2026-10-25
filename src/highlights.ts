export interface Span {
  id: string;
  start: number | null;
  end: number | null;
}
export interface Segment {
  text: string;
  ids: string[];
}

export function buildSegments(text: string, spans: Span[]): Segment[] {
  const points = Array.from(text);
  const valid = spans.filter((span): span is Span & { start: number; end: number } =>
    span.start !== null && span.end !== null && Number.isInteger(span.start) && Number.isInteger(span.end)
    && span.start >= 0 && span.end > span.start && span.end <= points.length);
  const boundaries = [...new Set([0, points.length, ...valid.flatMap(span => [span.start, span.end])])]
    .sort((a, b) => a - b);
  return boundaries.slice(0, -1).map((start, index) => {
    const end = boundaries[index + 1];
    return {
      text: points.slice(start, end).join(""),
      ids: valid.filter(span => span.start <= start && span.end >= end).map(span => span.id),
    };
  });
}

export function classColor(name: string): string {
  let hash = 0;
  for (const character of name) hash = (hash * 31 + character.codePointAt(0)!) >>> 0;
  return ["#aa98ff", "#72d6bb", "#ffc47b", "#7dbfff", "#ef9fc6", "#c4d87a"][hash % 6];
}
