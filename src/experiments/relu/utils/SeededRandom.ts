/**
 * Deterministic pseudo-random generator.
 * Uses Mulberry32 for reproducible datasets.
 */

export class SeededRandom {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  next(): number {
    let t = (this.state += 0x6d2b79f5);

    t = Math.imul(t ^ (t >>> 15), t | 1);

    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);

    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  nextBetween(min: number, max: number): number {
    return min + (max - min) * this.next();
  }

  nextInt(min: number, max: number): number {
    return Math.floor(this.nextBetween(min, max + 1));
  }

  gaussian(mean = 0, stdDev = 1): number {
    let u = 0;
    let v = 0;

    while (u === 0) u = this.next();
    while (v === 0) v = this.next();

    const mag = Math.sqrt(-2 * Math.log(u));

    const z0 = mag * Math.cos(2 * Math.PI * v);

    return mean + stdDev * z0;
  }

  shuffle<T>(array: readonly T[]): T[] {
    const result = [...array];

    for (let i = result.length - 1; i > 0; i--) {
      const j = this.nextInt(0, i);

      [result[i], result[j]] = [result[j], result[i]];
    }

    return result;
  }
}