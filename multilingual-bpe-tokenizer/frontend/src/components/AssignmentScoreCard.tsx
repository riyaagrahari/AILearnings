import type { StatisticsResponse } from "../types";
import { Card } from "./Card";

function formatScore(score: number | string): string {
  if (typeof score === "string") return score;
  return score.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

export function AssignmentScoreCard({ stats }: { stats: StatisticsResponse }) {
  const isInfinite = stats.assignment_score === "Infinity";

  return (
    <Card
      title="Assignment score"
      subtitle="score = 1000 / (max Xi − min Xi), where Xi = tokens / words (fertility)"
    >
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Metric label="Largest ratio" value={stats.largest_ratio.toFixed(4)} />
        <Metric label="Smallest ratio" value={stats.smallest_ratio.toFixed(4)} />
        <Metric label="Difference" value={stats.difference.toFixed(4)} />
        <Metric
          label="Assignment score"
          value={formatScore(stats.assignment_score)}
          highlight
          infinite={isInfinite}
        />
      </div>
      {isInfinite && (
        <p className="mt-4 text-xs text-slate-500 dark:text-slate-400">
          The difference between the largest and smallest ratio is exactly zero, so the
          score is mathematically unbounded (displayed as <strong>Infinity</strong> rather
          than dividing by zero).
        </p>
      )}
    </Card>
  );
}

function Metric({
  label,
  value,
  highlight = false,
  infinite = false,
}: {
  label: string;
  value: string;
  highlight?: boolean;
  infinite?: boolean;
}) {
  return (
    <div
      className={`rounded-xl p-4 ${
        highlight
          ? "bg-gradient-to-br from-indigo-500 to-violet-600 text-white shadow-md shadow-indigo-500/30"
          : "bg-slate-50 dark:bg-slate-800/60"
      }`}
    >
      <p
        className={`text-xs font-medium uppercase tracking-wide ${
          highlight ? "text-indigo-100" : "text-slate-500 dark:text-slate-400"
        }`}
      >
        {label}
      </p>
      <p
        className={`mt-1 font-mono-token text-xl font-bold sm:text-2xl ${
          highlight
            ? "text-white"
            : infinite
              ? "text-indigo-600 dark:text-indigo-400"
              : "text-slate-900 dark:text-slate-50"
        }`}
      >
        {value}
      </p>
    </div>
  );
}
