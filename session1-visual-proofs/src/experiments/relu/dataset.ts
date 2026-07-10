import { SeededRandom } from "./utils/SeededRandom";

export type BinaryLabel = 0 | 1;

export interface Point2D {
  x: number;
  y: number;
}

export interface DataPoint extends Point2D {
  label: BinaryLabel;
}

export interface Dataset {
  points: DataPoint[];
}

export interface DatasetGeneratorOptions {
  samples: number;
  noise: number;
  radius: number;
  seed: number;
}

export function generateConcentricCircles(
  opts: DatasetGeneratorOptions,
): Dataset {
  const { samples, noise, radius, seed } = opts;

  const rng = new SeededRandom(seed);

  const points: DataPoint[] = [];

  // Half of samples inner circle (label 0), half outer circle (label 1)
  const half = Math.max(2, Math.floor(samples / 2));

  for (let i = 0; i < half; i++) {
    // inner circle around radius * 0.5
    const r = (radius * 0.5) + rng.gaussian(0, noise);
    const angle = rng.nextBetween(0, Math.PI * 2);

    points.push({
      x: r * Math.cos(angle),
      y: r * Math.sin(angle),
      label: 0,
    });
  }

  for (let i = 0; i < samples - half; i++) {
    // outer circle around radius
    const r = radius + rng.gaussian(0, noise);
    const angle = rng.nextBetween(0, Math.PI * 2);

    points.push({
      x: r * Math.cos(angle),
      y: r * Math.sin(angle),
      label: 1,
    });
  }

  // Shuffle with seeded RNG
  const shuffled = rng.shuffle(points);

  return { points: shuffled };
}

export function datasetToTensors(dataset: Dataset) {
  const xs = dataset.points.map((p) => [p.x, p.y]);
  const ys = dataset.points.map((p) => p.label);

  return {
    xs: xs,
    ys: ys,
  } as const;
}
