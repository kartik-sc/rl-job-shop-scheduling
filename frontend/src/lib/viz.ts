import * as d3 from "d3";

export const riskColor = d3
  .scaleLinear<string>()
  .domain([0, 0.3, 0.7, 1])
  .range(["#34d399", "#a3e635", "#fbbf24", "#fb7185"])
  .clamp(true)
  .interpolate(d3.interpolateRgb);

export function shade(hex: string, k: number): string {
  const c = d3.color(hex);
  if (!c) return hex;
  return (k >= 0 ? c.brighter(k) : c.darker(-k)).formatHex();
}

export const JOB_HUES = [
  "#38bdf8", "#c084fc", "#2dd4bf", "#fbbf24", "#818cf8", "#f472b6",
  "#fb923c", "#4ade80", "#22d3ee", "#e879f9",
];

export function jobColor(id: number): string {
  return JOB_HUES[id % JOB_HUES.length];
}

export function mulberry32(a: number) {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function clamp(x: number, a: number, b: number) {
  return Math.max(a, Math.min(b, x));
}

export function easeInOut(p: number) {
  return p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;
}
