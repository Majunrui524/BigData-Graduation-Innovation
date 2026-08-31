export function formatNumber(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

export function formatMetric(value: number, digits = 3): string {
  return value.toFixed(digits);
}

export function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

export function scoreToColor(score: number): string {
  const value = clamp01(score);
  const red = Math.round(168 + (255 - 168) * value);
  const green = Math.round(64 + (213 - 64) * (1 - value));
  const blue = Math.round(45 + (71 - 45) * (1 - value));
  return `rgb(${red}, ${green}, ${blue})`;
}

export function excerpt(value: string, maxLength = 180): string {
  if (!value) return "";
  if (value.length <= maxLength) return value;
  return `${value.slice(0, maxLength - 1).trimEnd()}…`;
}
