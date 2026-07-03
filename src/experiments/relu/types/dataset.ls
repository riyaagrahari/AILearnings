export interface Point2D {
  x: number;
  y: number;
}

export type BinaryLabel = 0 | 1;

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

export interface DatasetStatistics {
  totalSamples: number;
  classZero: number;
  classOne: number;
}

export interface TrainingSample {
  xs: [number, number];
  ys: BinaryLabel;
}