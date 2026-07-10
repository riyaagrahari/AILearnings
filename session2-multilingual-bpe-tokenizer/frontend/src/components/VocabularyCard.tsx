import type { StatisticsResponse } from "../types";
import { Card } from "./Card";

export function VocabularyCard({ stats }: { stats: StatisticsResponse }) {
  return (
    <Card title="Vocabulary" subtitle="Loaded from the trained tokenizer's vocab.json">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div className="rounded-xl bg-slate-50 p-4 dark:bg-slate-800/60">
          <p className="text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">
            Vocabulary size
          </p>
          <p className="mt-1 font-mono-token text-2xl font-bold text-slate-900 dark:text-slate-50">
            {stats.vocab_size.toLocaleString()}
          </p>
        </div>
        <div className="rounded-xl bg-slate-50 p-4 dark:bg-slate-800/60">
          <p className="text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">
            Training languages
          </p>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {stats.languages.map((entry) => (
              <span
                key={entry.language}
                className="rounded-full bg-indigo-100 px-2.5 py-1 text-xs font-semibold text-indigo-700 dark:bg-indigo-500/20 dark:text-indigo-300"
              >
                {entry.language}
              </span>
            ))}
          </div>
        </div>
      </div>
    </Card>
  );
}
