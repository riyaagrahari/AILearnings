import type { StatisticsResponse } from "../types";
import { Card } from "./Card";

function formatRatio(value: number): string {
  return value.toFixed(4);
}

export function LanguageStatsTable({ stats }: { stats: StatisticsResponse }) {
  const ratios = stats.languages.map((entry) => entry.ratio);
  const min = Math.min(...ratios);
  const max = Math.max(...ratios);

  return (
    <Card title="Language statistics" subtitle="Xi = total tokens ÷ total words (fertility), per language">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[480px] text-left text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500 dark:border-slate-700 dark:text-slate-400">
              <th className="py-2 pr-4 font-medium">Language</th>
              <th className="py-2 pr-4 font-medium">Total tokens</th>
              <th className="py-2 pr-4 font-medium">Total words</th>
              <th className="py-2 pr-4 font-medium">Xi (ratio)</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
            {stats.languages.map((entry) => (
              <tr key={entry.language}>
                <td className="py-2.5 pr-4 font-medium text-slate-900 dark:text-slate-100">
                  {entry.language}
                </td>
                <td className="py-2.5 pr-4 font-mono-token text-slate-700 dark:text-slate-300">
                  {entry.total_tokens.toLocaleString()}
                </td>
                <td className="py-2.5 pr-4 font-mono-token text-slate-700 dark:text-slate-300">
                  {entry.total_words.toLocaleString()}
                </td>
                <td className="py-2.5 pr-4">
                  <span
                    className={`inline-flex items-center rounded-md px-2 py-0.5 font-mono-token text-xs font-semibold ${
                      entry.ratio === max && entry.ratio !== min
                        ? "bg-amber-100 text-amber-800 dark:bg-amber-500/20 dark:text-amber-300"
                        : entry.ratio === min && entry.ratio !== max
                          ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-500/20 dark:text-emerald-300"
                          : "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300"
                    }`}
                  >
                    {formatRatio(entry.ratio)}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
