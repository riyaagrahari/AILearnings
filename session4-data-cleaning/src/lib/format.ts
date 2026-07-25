export function comma(n: number): string {
  return n.toLocaleString("en-US");
}

/** 62_141_238 -> "62.1M", 43_415_548 -> "43.4M", 9_800 -> "9.8K" */
export function compact(n: number): string {
  if (Math.abs(n) >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (Math.abs(n) >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return String(n);
}

export function pct(n: number, digits = 1): string {
  return n.toFixed(digits) + "%";
}

/** Turn a snake_case reason/key into "Title Case" for display. */
export function humanize(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .replace(/\bPii\b/, "PII")
    .replace(/\bIpv4\b/, "IPv4")
    .replace(/\bHi\b/, "Hindi")
    .replace(/\bMr\b/, "Marathi")
    .replace(/\bTe\b/, "Telugu");
}
