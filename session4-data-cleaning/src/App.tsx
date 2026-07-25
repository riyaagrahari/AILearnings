import { Background } from "./components/Background";
import { Hero } from "./components/sections/Hero";
import { Strategies } from "./components/sections/Strategies";
import { Dataset } from "./components/sections/Dataset";
import { Pipeline } from "./components/sections/Pipeline";
import { Concerns } from "./components/sections/Concerns";
import { FinalStats } from "./components/sections/FinalStats";
import { stats } from "./data/stats";

const NAV = [
  { id: "strategies", label: "Strategies" },
  { id: "dataset", label: "Dataset" },
  { id: "pipeline", label: "Pipeline" },
  { id: "concerns", label: "Concerns" },
  { id: "stats", label: "Statistics" },
];

export default function App() {
  return (
    <div className="relative min-h-screen">
      <Background />

      <nav className="sticky top-0 z-40 border-b border-white/5 bg-ink/70 backdrop-blur-lg">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-3">
          <a href="#top" className="flex items-center gap-2 font-semibold text-white">
            <span className="grid h-7 w-7 place-items-center rounded-lg bg-gradient-to-br from-accent-cyan to-accent-indigo text-xs font-bold text-ink">
              S4
            </span>
            <span className="hidden sm:inline">Corpus Cleaning</span>
          </a>
          <div className="flex items-center gap-1 text-sm">
            {NAV.map((n) => (
              <a
                key={n.id}
                href={`#${n.id}`}
                className="rounded-full px-3 py-1.5 text-slate-400 transition hover:bg-white/5 hover:text-white"
              >
                {n.label}
              </a>
            ))}
          </div>
        </div>
      </nav>

      <main id="top">
        <Hero />
        <Strategies />
        <Dataset />
        <Pipeline />
        <Concerns />
        <FinalStats />
      </main>

      <footer className="border-t border-white/5 py-10">
        <div className="mx-auto max-w-6xl px-6 text-sm text-slate-500">
          <p>
            Session 4 · every statistic is measured by{" "}
            <code className="text-slate-400">pipeline/run_pipeline.py</code> over real{" "}
            <a
              href={stats.meta.dataset_url}
              target="_blank"
              rel="noreferrer"
              className="text-accent-cyan hover:underline"
            >
              wikimedia/wikipedia
            </a>{" "}
            data ({stats.meta.license}). Generated {stats.meta.generated_at}.
          </p>
          <p className="mt-1">
            Cleaning strategies carried over from Session 3&apos;s India-first 40B foundation-model
            design.
          </p>
        </div>
      </footer>
    </div>
  );
}
