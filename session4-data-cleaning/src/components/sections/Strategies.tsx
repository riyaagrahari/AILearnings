import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown } from "lucide-react";
import { Section } from "../layout/Section";
import { SectionHeading } from "../ui/SectionHeading";
import { Reveal } from "../ui/Reveal";
import { stats } from "../../data/stats";
import { STRATEGY_INFO } from "../../data/strategyInfo";
import { comma } from "../../lib/format";

/** One-line, human summary of what each stage measurably did to the corpus. */
function effect(id: string): string {
  const s = stats.stages.find((x) => x.id === id)!;
  switch (id) {
    case "raw":
      return `${comma(s.docs_out)} documents ingested, hashed & license-tagged`;
    case "langid":
      return `${comma(s.removed)} wrong-script docs dropped · ${comma(
        (s.review_pool?.code_switch ?? 0) + (s.review_pool?.low_confidence ?? 0),
      )} routed to review`;
    case "normalize":
      return `${comma(s.counters?.quotes ?? 0)} quotes fixed · ${comma(
        s.counters?.zwj_zwnj_preserved ?? 0,
      )} ZWJ/ZWNJ preserved`;
    case "dedup":
      return `${comma(s.removed_reasons?.exact_duplicate ?? 0)} exact + ${comma(
        s.removed_reasons?.near_duplicate ?? 0,
      )} near-duplicates removed`;
    case "quality":
      return `${comma(s.removed)} low-quality / stub documents dropped`;
    case "safety":
      return `${comma(s.docs_with_pii_redacted ?? 0)} docs PII-redacted · ${comma(
        s.removed,
      )} toxic docs dropped`;
    case "code":
      return `SPDX license allow-listed · ${comma(
        s.docs_with_secret_redacted ?? 0,
      )} secrets found`;
    case "corpus":
      return `Rebalanced to target mix · ${comma(s.estimated_tokens ?? 0)} est. tokens shipped`;
    default:
      return "";
  }
}

export function Strategies() {
  const [open, setOpen] = useState<string | null>("dedup");
  const { strategy_count } = stats;

  return (
    <Section id="strategies">
      <SectionHeading
        index="01"
        eyebrow="How many strategies, and what are they"
        title="Eight stages. Six of them are real cleanups."
        description="Session 3 defines its data-cleaning pipeline as eight stages. Two are bookends — raw ingest and final corpus assembly — and the six in between are the active cleanups. Each card pairs Session 3's design rationale with the effect I actually measured running it."
      />

      <Reveal className="mt-8 flex flex-wrap gap-3">
        <Pill label="Total stages" value={String(strategy_count.total_stages)} />
        <Pill label="Active cleanups" value={String(strategy_count.active_cleanups)} />
        <Pill label="Bookends (ingest + ship)" value="2" />
      </Reveal>

      <div className="mt-10 grid gap-4 md:grid-cols-2">
        {stats.stages.map((s, i) => {
          const info = STRATEGY_INFO[s.id];
          const Icon = info.icon;
          const isOpen = open === s.id;
          return (
            <Reveal key={s.id} delay={i * 0.04}>
              <div
                className={`glass overflow-hidden rounded-2xl transition-colors ${
                  isOpen ? "border-accent-indigo/40" : ""
                }`}
              >
                <button
                  onClick={() => setOpen(isOpen ? null : s.id)}
                  className="flex w-full items-start gap-4 p-5 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-indigo/60"
                >
                  <span
                    className="grid h-11 w-11 shrink-0 place-items-center rounded-xl"
                    style={{ background: `${info.color}1f`, color: info.color }}
                  >
                    <Icon className="h-5 w-5" />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="flex items-center gap-2">
                      <span className="font-mono text-[11px] text-slate-500">
                        {String(i + 1).padStart(2, "0")}
                      </span>
                      <span className="text-base font-semibold text-white">{s.title}</span>
                      {!s.is_active_cleanup && (
                        <span className="rounded-full bg-white/5 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-slate-400">
                          bookend
                        </span>
                      )}
                    </span>
                    <span className="mt-1 block text-sm text-slate-400">{effect(s.id)}</span>
                  </span>
                  <ChevronDown
                    className={`mt-1 h-4 w-4 shrink-0 text-slate-500 transition-transform ${
                      isOpen ? "rotate-180" : ""
                    }`}
                  />
                </button>

                <AnimatePresence initial={false}>
                  {isOpen && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: "auto", opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
                    >
                      <div className="space-y-4 border-t border-white/5 px-5 pb-5 pt-4 text-sm">
                        <p className="text-slate-300">{info.why}</p>
                        <div>
                          <div className="eyebrow mb-2">Methods</div>
                          <ul className="space-y-1.5">
                            {info.methods.map((m) => (
                              <li key={m} className="flex gap-2 text-slate-400">
                                <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-accent-cyan/70" />
                                {m}
                              </li>
                            ))}
                          </ul>
                        </div>
                        <div className="rounded-xl bg-white/[0.03] p-3 text-slate-400">
                          <span className="font-semibold text-slate-300">Trade-off · </span>
                          {info.tradeoff}
                        </div>
                        <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-slate-500">
                          <span>
                            In <span className="text-slate-300">{comma(s.docs_in)}</span> docs
                          </span>
                          <span>
                            Out <span className="text-slate-300">{comma(s.docs_out)}</span> docs
                          </span>
                          <span>
                            Removed <span className="text-rose-300">{comma(s.removed)}</span>
                          </span>
                          <span>
                            Ran in <span className="text-slate-300">{s.seconds}s</span>
                          </span>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </Reveal>
          );
        })}
      </div>
    </Section>
  );
}

function Pill({ label, value }: { label: string; value: string }) {
  return (
    <div className="glass rounded-full px-4 py-2 text-sm">
      <span className="font-bold text-white">{value}</span>{" "}
      <span className="text-slate-400">{label}</span>
    </div>
  );
}
