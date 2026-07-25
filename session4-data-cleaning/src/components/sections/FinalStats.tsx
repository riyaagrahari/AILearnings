import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ReactNode } from "react";
import { Boxes, Hash, ScrollText, Timer } from "lucide-react";
import { Section } from "../layout/Section";
import { SectionHeading } from "../ui/SectionHeading";
import { Reveal } from "../ui/Reveal";
import { AnimatedNumber } from "../ui/AnimatedNumber";
import { stats } from "../../data/stats";
import { STRATEGY_INFO } from "../../data/strategyInfo";
import { comma, compact } from "../../lib/format";

const AXIS = "#64748b";
const GRID = "rgba(255,255,255,0.06)";

export function FinalStats() {
  const { totals, per_language, meta } = stats;
  const corpus = stats.stages.find((s) => s.id === "corpus")!;
  const quality = stats.stages.find((s) => s.id === "quality")!;

  const removalsByStage = stats.stages
    .filter((s) => s.removed > 0)
    .map((s) => ({ name: s.short, removed: s.removed, color: STRATEGY_INFO[s.id].color }));

  const langData = meta.languages.map((l) => ({
    name: l.name,
    before: per_language.before[l.code] ?? 0,
    after: per_language.after[l.code] ?? 0,
  }));

  const deciles = (quality.score_deciles ?? []).map((v, i) => ({
    name: `${i / 10}`,
    docs: v,
  }));

  return (
    <Section id="stats">
      <SectionHeading
        index="05"
        eyebrow="Final statistics"
        title="The corpus that comes out the other end."
        description="A frozen, versioned, rebalanced training snapshot — with every headline number computed live from the run."
      />

      {/* headline number cards */}
      <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard icon={<Boxes className="h-5 w-5" />} label="Final documents">
          <AnimatedNumber value={totals.final_documents} />
        </StatCard>
        <StatCard icon={<ScrollText className="h-5 w-5" />} label="Final words">
          <AnimatedNumber value={(corpus.final_words ?? 0) / 1e6} decimals={2} suffix="M" />
        </StatCard>
        <StatCard icon={<Hash className="h-5 w-5" />} label="Est. tokens (chars/4)">
          <AnimatedNumber value={(corpus.estimated_tokens ?? 0) / 1e6} decimals={2} suffix="M" />
        </StatCard>
        <StatCard icon={<Timer className="h-5 w-5" />} label="Pipeline runtime">
          <AnimatedNumber value={meta.runtime_seconds} decimals={0} suffix="s" />
        </StatCard>
      </div>

      {/* retention summary bar */}
      <Reveal className="mt-6">
        <div className="glass rounded-2xl p-6">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <div className="eyebrow">End-to-end retention</div>
              <div className="mt-2 text-3xl font-bold text-white">
                {comma(totals.raw_documents)}{" "}
                <span className="text-lg font-normal text-slate-500">raw docs</span> →{" "}
                <span className="text-gradient">{comma(totals.final_documents)}</span>{" "}
                <span className="text-lg font-normal text-slate-500">shipped</span>
              </div>
            </div>
            <div className="flex gap-6 text-right">
              <div>
                <div className="text-2xl font-bold text-accent-cyan">
                  {totals.character_retention_pct}%
                </div>
                <div className="text-xs text-slate-500">characters kept</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-accent-indigo">
                  {totals.document_retention_pct}%
                </div>
                <div className="text-xs text-slate-500">documents kept</div>
              </div>
            </div>
          </div>
          <div className="mt-5 flex h-4 w-full overflow-hidden rounded-full bg-white/5">
            <div
              className="h-full bg-gradient-to-r from-accent-cyan to-accent-indigo"
              style={{ width: `${totals.character_retention_pct}%` }}
            />
          </div>
        </div>
      </Reveal>

      {/* charts */}
      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Reveal>
          <ChartCard title="Documents removed by stage">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={removalsByStage} margin={{ top: 8, right: 8, bottom: 0, left: -10 }}>
                <CartesianGrid stroke={GRID} vertical={false} />
                <XAxis dataKey="name" stroke={AXIS} tick={{ fontSize: 11 }} tickLine={false} />
                <YAxis stroke={AXIS} tick={{ fontSize: 11 }} tickLine={false} tickFormatter={compact} />
                <Tooltip
                  cursor={{ fill: "rgba(255,255,255,0.04)" }}
                  contentStyle={tooltipStyle}
                  formatter={(v: number) => [comma(v), "removed"]}
                />
                <Bar dataKey="removed" radius={[6, 6, 0, 0]}>
                  {removalsByStage.map((d, i) => (
                    <Cell key={i} fill={d.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        </Reveal>

        <Reveal delay={0.05}>
          <ChartCard title="Per-language mix: before → after rebalancing">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={langData} margin={{ top: 8, right: 8, bottom: 0, left: -10 }}>
                <CartesianGrid stroke={GRID} vertical={false} />
                <XAxis dataKey="name" stroke={AXIS} tick={{ fontSize: 11 }} tickLine={false} />
                <YAxis stroke={AXIS} tick={{ fontSize: 11 }} tickLine={false} tickFormatter={compact} />
                <Tooltip
                  cursor={{ fill: "rgba(255,255,255,0.04)" }}
                  contentStyle={tooltipStyle}
                  formatter={(v: number) => comma(v)}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="before" name="raw" fill="#334155" radius={[6, 6, 0, 0]} />
                <Bar dataKey="after" name="shipped" fill="#6366f1" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        </Reveal>

        <Reveal delay={0.1} className="lg:col-span-2">
          <ChartCard title="Quality-score distribution of surviving documents (0 = worst, 0.9 = best)">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={deciles} margin={{ top: 8, right: 8, bottom: 0, left: -10 }}>
                <CartesianGrid stroke={GRID} vertical={false} />
                <XAxis dataKey="name" stroke={AXIS} tick={{ fontSize: 11 }} tickLine={false} />
                <YAxis stroke={AXIS} tick={{ fontSize: 11 }} tickLine={false} tickFormatter={compact} />
                <Tooltip
                  cursor={{ fill: "rgba(255,255,255,0.04)" }}
                  contentStyle={tooltipStyle}
                  formatter={(v: number) => [comma(v), "docs"]}
                  labelFormatter={(l) => `score ≥ ${l}`}
                />
                <Bar dataKey="docs" fill="#22d3ee" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        </Reveal>
      </div>

      {/* frozen snapshot */}
      <Reveal delay={0.1} className="mt-6">
        <div className="glass rounded-2xl p-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="eyebrow">Frozen training snapshot</div>
              <p className="mt-2 max-w-xl text-sm text-slate-400">
                Rebalanced to the target mix{" "}
                <span className="text-slate-200">
                  {Object.entries(corpus.target_mix ?? {})
                    .map(([k, v]) => `${k} ${Math.round(v * 100)}%`)
                    .join(" · ")}
                </span>
                , curriculum-ordered by quality, and content-hashed for reproducibility.
              </p>
            </div>
            <div className="rounded-xl bg-white/[0.03] px-4 py-3 font-mono text-sm">
              <div className="text-[10px] uppercase tracking-widest text-slate-500">
                snapshot sha256
              </div>
              <div className="text-accent-cyan">{corpus.snapshot_sha256_16}…</div>
            </div>
          </div>
        </div>
      </Reveal>
    </Section>
  );
}

const tooltipStyle = {
  background: "rgba(10,15,36,0.95)",
  border: "1px solid rgba(255,255,255,0.1)",
  borderRadius: 12,
  fontSize: 12,
  color: "#e2e8f0",
};

function StatCard({
  icon,
  label,
  children,
}: {
  icon: ReactNode;
  label: string;
  children: ReactNode;
}) {
  return (
    <div className="glass rounded-2xl p-5">
      <div className="flex items-center gap-2 text-accent-cyan/80">
        {icon}
        <span className="text-xs font-semibold uppercase tracking-widest">{label}</span>
      </div>
      <div className="mt-3 text-3xl font-bold text-white">{children}</div>
    </div>
  );
}

function ChartCard({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="glass rounded-2xl p-5">
      <h3 className="mb-4 text-sm font-semibold text-slate-300">{title}</h3>
      {children}
    </div>
  );
}
