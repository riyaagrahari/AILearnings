export const clamp = (
  value: number,
  min: number,
  max: number,
): number => Math.min(max, Math.max(min, value));

export const lerp = (
  start: number,
  end: number,
  t: number,
): number => start + (end - start) * t;

export const inverseLerp = (
  start: number,
  end: number,
  value: number,
): number => (value - start) / (end - start);

export const remap = (
  value: number,
  inMin: number,
  inMax: number,
  outMin: number,
  outMax: number,
): number => {
  const t = inverseLerp(inMin, inMax, value);

  return lerp(outMin, outMax, t);
};

export const distance = (
  x1: number,
  y1: number,
  x2: number,
  y2: number,
): number => {
  const dx = x2 - x1;
  const dy = y2 - y1;

  return Math.sqrt(dx * dx + dy * dy);
};

export const sigmoid = (x: number): number =>
  1 / (1 + Math.exp(-x));

export const degreesToRadians = (
  degrees: number,
): number => (degrees * Math.PI) / 180;

export const radiansToDegrees = (
  radians: number,
): number => (radians * 180) / Math.PI;