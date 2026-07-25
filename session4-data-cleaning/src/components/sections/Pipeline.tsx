import { motion } from "framer-motion";
import { ArrowRight, Quote } from "lucide-react";
import { Section } from "../layout/Section";
import { SectionHeading } from "../ui/SectionHeading";
import { Reveal } from "../ui/Reveal";
import { stats } from "../../data/stats";
import { STRATEGY_INFO } from "../../data/strategyInfo";
import { comma, humanize } from "../../lib/format";

export function Pipeline() {
  const raw = stats.totals.raw_documents;
  const active = stats.stages.filter((s) => s.id !== "raw");

  return (
    <Section id="pipeline">
      <SectionHeading
        index="03"
        eyebrow="What was cleaned, and why"
        title="30,208 → 7,580 documents, stage by stage."
        description="Every document streams through the eight stages in order. The funnel below shows exactly how many survived each one, the reasons things were dropped, and real examples pulled straight from the run."
      />

      <div className="mt-10 space-y-3">
        {active.map((s, i) => {
          const info = STRATEGY_INFO[s.id];
          const Icon = info.icon;
          const outW = Math.max(2, (s.docs_out / raw) * 100);
          const removedW = (s.removed / raw) * 100;
          return (
            <Reveal key={s.id} delay={i * 0.05}>
              <div className="glass rounded-2xl p-5">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-3">
                    <span
                      className="grid h-9 w-9 place-items-center rounded-lg"
                      style={{ background: `${info.color}1f`, color: info.color }}
                    >
                      <Icon className="h-4 w-4" />
                    </span>
                    <div>
                      <div className="font-semibold text-white">{s.title}</div>
                      <div className="text-xs text-slate-500">
                        {comma(s.docs_in)} in{" "}
                        <ArrowRight className="mx-0.5 inline h-3 w-3" /> {comma(s.docs_out)} out
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-semibold text-rose-300">
                      {s.removed > 0 ? `−${comma(s.removed)}` : "0 dropped"}
                    </div>
                    <div className="text-[11px] text-slate-500">
                      {((s.docs_out / raw) * 100).toFixed(1)}% of raw remains
                    </div>
                  </div>
                </div>

                {/* funnel bar */}
                <div className="mt-4 flex h-3 w-full overflow-hidden rounded-full bg-white/5">
                  <motion.div
                    className="h-full rounded-l-full"
                    style={{ background: info.color }}
                    initial={{ width: 0 }}
                    whileInView={{ width: `${outW}%` }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
                  />
                  {removedW > 0.15 && (
                    <div
                      className="h-full bg-rose-500/40"
                      style={{ width: `${Math.min(removedW, 100 - outW)}%` }}
                    />
                  )}
                </div>

                {/* reason chips */}
                {s.removed_reasons && Object.keys(s.removed_reasons).length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {Object.entries(s.removed_reasons)
                      .filter(([, v]) => v > 0)
                      .map(([k, v]) => (
                        <span
                          key={k}
                          className="rounded-full bg-rose-500/10 px-2.5 py-1 text-[11px] text-rose-200/90"
                        >
                          {humanize(k)} · {comma(v)}
                        </span>
                      ))}
                  </div>
                )}

                <Examples stageId={s.id} />
              </div>
            </Reveal>
          );
        })}
      </div>
    </Section>
  );
}

function Examples({ stageId }: { stageId: string }) {
  const s = stats.stages.find((x) => x.id === stageId)!;

  if (stageId === "normalize" && s.example) {
    return (
      <div className="mt-4 grid gap-2 rounded-xl bg-white/[0.03] p-3 text-sm sm:grid-cols-2">
        <div>
          <div className="eyebrow mb-1 flex items-center gap-1">
            <Quote className="h-3 w-3" /> before
          </div>
          <p dir="auto" className="font-mono text-xs leading-relaxed text-slate-400">
            …{s.example.before}…
          </p>
        </div>
        <div>
          <div className="eyebrow mb-1 text-emerald-300/80">after</div>
          <p dir="auto" className="font-mono text-xs leading-relaxed text-slate-200">
            …{s.example.after}…
          </p>
        </div>
      </div>
    );
  }

  const ex = s.examples_removed;
  if (!ex || ex.length === 0) return null;

  return (
    <div className="mt-4 rounded-xl bg-white/[0.03] p-3">
      <div className="eyebrow mb-2">Real examples from this run</div>
      <ul className="space-y-1.5 text-xs text-slate-400">
        {ex.slice(0, 4).map((e, idx) => (
          <li key={idx} dir="auto" className="flex flex-wrap items-baseline gap-x-2">
            {stageId === "langid" && (
              <>
                <span className="text-slate-200">{e.title}</span>
                <span className="text-slate-500">
                  labelled {String(e.src_lang)}, detected {String(e.detected)} (conf {String(e.confidence)})
                </span>
              </>
            )}
            {stageId === "dedup" && (
              <>
                <span className="text-rose-200/80 line-through decoration-rose-400/40">{e.dropped}</span>
                <ArrowRight className="h-3 w-3 text-slate-600" />
                <span className="text-slate-200">{e.kept_as}</span>
                <span className="text-slate-500">≈{String(e.jaccard_est)} Jaccard</span>
              </>
            )}
            {stageId === "quality" && (
              <>
                <span className="text-slate-200">{e.title}</span>
                <span className="text-slate-500">
                  {humanize(String(e.reason))} · {String(e.words)} words
                </span>
              </>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
