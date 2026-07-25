import type { ReactNode } from "react";
import { motion } from "framer-motion";
import { ArrowDownRight, Database, Layers, Sparkles } from "lucide-react";
import { stats } from "../../data/stats";
import { AnimatedNumber } from "../ui/AnimatedNumber";
import { compact, pct } from "../../lib/format";

export function Hero() {
  const { totals, strategy_count, meta } = stats;
  return (
    <header className="relative mx-auto flex min-h-[92vh] max-w-6xl flex-col justify-center px-6 py-24">
      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
      >
        <div className="flex items-center gap-3">
          <span className="font-mono text-xs text-accent-cyan/70">SESSION 04</span>
          <span className="h-px w-10 bg-gradient-to-r from-accent-cyan/60 to-transparent" />
          <span className="eyebrow">Data Cleaning, Measured</span>
        </div>

        <h1 className="mt-6 max-w-4xl text-4xl font-extrabold leading-[1.05] tracking-tight text-white sm:text-6xl md:text-7xl">
          Cleaning an <span className="text-gradient">India-first</span> corpus with
          Session&nbsp;3&apos;s eight strategies.
        </h1>

        <p className="mt-7 max-w-2xl text-lg leading-relaxed text-slate-400">
          I took the <strong className="text-slate-200">eight-stage cleaning pipeline</strong> proposed
          in Session&nbsp;3, implemented every stage from scratch, and ran it over a real{" "}
          <strong className="text-slate-200">{compact(totals.raw_characters)}-character</strong>{" "}
          Wikipedia corpus in Hindi, Telugu and Marathi. Every number on this page is measured,
          not fabricated.
        </p>

        <div className="mt-10 grid gap-4 sm:grid-cols-3">
          <HeroStat
            icon={<Layers className="h-5 w-5" />}
            label="Cleaning strategies"
            value={<>{strategy_count.total_stages} <span className="text-lg text-slate-500">stages</span></>}
            sub={`${strategy_count.active_cleanups} active cleanups + ingest & ship`}
          />
          <HeroStat
            icon={<Database className="h-5 w-5" />}
            label="Raw corpus ingested"
            value={<AnimatedNumber value={totals.raw_characters / 1e6} decimals={1} suffix="M chars" />}
            sub={`${compact(totals.raw_documents)} real articles · ${meta.languages.length} languages`}
          />
          <HeroStat
            icon={<Sparkles className="h-5 w-5" />}
            label="Kept for training"
            value={<AnimatedNumber value={totals.character_retention_pct} decimals={1} suffix="%" />}
            sub={`${compact(totals.final_characters)} chars survived · ${pct(totals.document_retention_pct)} of docs`}
          />
        </div>

        <a
          href="#strategies"
          className="mt-12 inline-flex items-center gap-2 text-sm font-medium text-slate-300 transition hover:text-white"
        >
          Walk the pipeline <ArrowDownRight className="h-4 w-4" />
        </a>
      </motion.div>
    </header>
  );
}

function HeroStat({
  icon,
  label,
  value,
  sub,
}: {
  icon: ReactNode;
  label: string;
  value: ReactNode;
  sub: string;
}) {
  return (
    <div className="glass rounded-2xl p-5">
      <div className="flex items-center gap-2 text-accent-cyan/80">
        {icon}
        <span className="text-xs font-semibold uppercase tracking-widest">{label}</span>
      </div>
      <div className="mt-3 text-3xl font-bold text-white">{value}</div>
      <div className="mt-1 text-xs text-slate-500">{sub}</div>
    </div>
  );
}
