import { useCallback, useEffect, useState } from "react";
import { ApiError, getHealth, getStatistics } from "./api/client";
import { AssignmentScoreCard } from "./components/AssignmentScoreCard";
import { Card } from "./components/Card";
import { DownloadSection } from "./components/DownloadSection";
import { Header } from "./components/Header";
import { LanguageStatsTable } from "./components/LanguageStatsTable";
import { Playground } from "./components/Playground";
import { ErrorState, LoadingSkeleton } from "./components/StatusStates";
import { StatsCharts } from "./components/StatsCharts";
import { VocabularyCard } from "./components/VocabularyCard";
import type { StatisticsResponse } from "./types";

export default function App() {
  const [connected, setConnected] = useState<boolean | null>(null);
  const [stats, setStats] = useState<StatisticsResponse | null>(null);
  const [statsLoading, setStatsLoading] = useState(true);
  const [statsError, setStatsError] = useState<string | null>(null);

  const loadStatistics = useCallback(() => {
    setStatsLoading(true);
    setStatsError(null);
    getStatistics()
      .then((data) => {
        setStats(data);
        setConnected(true);
      })
      .catch((err) => {
        setStatsError(err instanceof ApiError ? err.message : "Failed to load statistics.");
        if (err instanceof ApiError && err.status !== 0) setConnected(true);
        else setConnected(false);
      })
      .finally(() => setStatsLoading(false));
  }, []);

  useEffect(() => {
    loadStatistics();
    getHealth()
      .then(() => setConnected(true))
      .catch(() => setConnected(false));
  }, [loadStatistics]);

  return (
    <div className="min-h-screen">
      <Header connected={connected} />

      <main className="mx-auto max-w-6xl space-y-6 px-4 py-8 sm:px-6">
        <section>
          <h2 className="sr-only">Overview</h2>
          {statsLoading && (
            <Card>
              <LoadingSkeleton lines={4} />
            </Card>
          )}
          {!statsLoading && statsError && (
            <ErrorState message={statsError} onRetry={loadStatistics} />
          )}
          {!statsLoading && !statsError && stats && (
            <div className="space-y-6">
              <VocabularyCard stats={stats} />
              <LanguageStatsTable stats={stats} />
              <AssignmentScoreCard stats={stats} />
              <StatsCharts stats={stats} />
            </div>
          )}
        </section>

        <Playground />

        <DownloadSection />

        <footer className="pb-6 pt-2 text-center text-xs text-slate-400 dark:text-slate-600">
          Every statistic above comes from running the actual trained tokenizer — nothing is
          hardcoded. See <code className="font-mono-token">docs/BPE_ALGORITHM.md</code> in the
          repository for the design rationale.
        </footer>
      </main>
    </div>
  );
}
