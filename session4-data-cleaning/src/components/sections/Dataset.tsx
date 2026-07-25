import type { ReactNode } from "react";
import { motion } from "framer-motion";
import { BookOpenText, ExternalLink, Scale } from "lucide-react";
import { Section } from "../layout/Section";
import { SectionHeading } from "../ui/SectionHeading";
import { Reveal } from "../ui/Reveal";
import { stats } from "../../data/stats";
import { comma, compact } from "../../lib/format";

const LANG_META: Record<string, { script: string; note: string }> = {
  hi: { script: "Devanagari", note: "Largest Indic language; shares its script with Marathi" },
  te: { script: "Telugu", note: "Distinct Dravidian script — a clean language-ID signal" },
  mr: { script: "Devanagari", note: "Same script as Hindi → language-ID can't rely on script alone" },
};

export function Dataset() {
  const { meta, totals } = stats;
  const raw = stats.stages.find((s) => s.id === "raw")!;
  const perLang = raw.per_language_docs ?? {};

  return (
    <Section id="dataset">
      <SectionHeading
        index="02"
        eyebrow="What dataset was picked"
        title="Real Wikipedia, in three Indian languages."
        description="Session 3's data mix leans on Wikipedia and high-quality web in Indic languages. So I pulled the genuine wikimedia/wikipedia dumps — open, CC BY-SA, no synthetic text — for Hindi, Telugu and Marathi, matching Session 2's Indic focus."
      />

      <div className="mt-10 grid gap-6 lg:grid-cols-3">
        <Reveal className="lg:col-span-2">
          <div className="glass h-full rounded-2xl p-6">
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="flex items-center gap-2 text-accent-cyan/80">
                  <BookOpenText className="h-5 w-5" />
                  <span className="text-xs font-semibold uppercase tracking-widest">Source dataset</span>
                </div>
                <h3 className="mt-2 text-2xl font-bold text-white">{meta.dataset}</h3>
              </div>
              <a
                href={meta.dataset_url}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 rounded-full bg-white/5 px-3 py-1.5 text-xs text-slate-300 transition hover:bg-white/10 hover:text-white"
              >
                Hugging Face <ExternalLink className="h-3 w-3" />
              </a>
            </div>

            <div className="mt-5 flex flex-wrap gap-2 text-xs">
              <Tag icon={<Scale className="h-3 w-3" />}>{meta.license}</Tag>
              <Tag>{comma(totals.raw_documents)} articles</Tag>
              <Tag>{compact(totals.raw_characters)} characters ingested</Tag>
              <Tag>{compact(meta.char_budget_per_language)} char budget / language</Tag>
            </div>

            <div className="mt-6 grid gap-3 sm:grid-cols-3">
              {meta.languages.map((l) => (
                <div key={l.code} className="rounded-xl bg-white/[0.03] p-4">
                  <div className="flex items-baseline justify-between">
                    <span className="text-sm font-semibold text-white">{l.name}</span>
                    <span className="font-mono text-[11px] uppercase text-slate-500">{l.code}</span>
                  </div>
                  <div className="mt-1 text-2xl font-bold text-accent-cyan">
                    {compact(perLang[l.code] ?? 0)}
                  </div>
                  <div className="text-[11px] text-slate-500">articles · {LANG_META[l.code].script}</div>
                  <p className="mt-2 text-xs leading-snug text-slate-400">{LANG_META[l.code].note}</p>
                </div>
              ))}
            </div>
          </div>
        </Reveal>

        <Reveal delay={0.1}>
          <div className="glass flex h-full flex-col justify-between rounded-2xl p-6">
            <div>
              <div className="eyebrow">The 10–100M target</div>
              <p className="mt-3 text-sm leading-relaxed text-slate-400">
                The assignment asks for a dataset in the{" "}
                <span className="text-slate-200">10–100M</span> range. Streaming a bounded budget out
                of the parquet shards lands the working set at:
              </p>
            </div>
            <div className="mt-6">
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5 }}
                className="text-5xl font-extrabold text-gradient"
              >
                {(totals.raw_characters / 1e6).toFixed(1)}M
              </motion.div>
              <div className="mt-1 text-sm text-slate-400">raw characters</div>
              <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-white/5">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-accent-cyan to-accent-indigo"
                  style={{ width: `${(totals.raw_characters / 100e6) * 100}%` }}
                />
              </div>
              <div className="mt-1 flex justify-between text-[10px] text-slate-600">
                <span>10M</span>
                <span>100M</span>
              </div>
            </div>
          </div>
        </Reveal>
      </div>

      <Reveal delay={0.15} className="mt-6">
        <div className="glass rounded-2xl p-5 text-sm text-slate-400">
          <span className="font-semibold text-slate-200">Why this dataset · </span>
          Hindi and Marathi share the Devanagari script while Telugu uses its own — so language
          identification genuinely has to work (script counting <em>and</em> marker words), not just
          read a label. It&apos;s open and reproducible (no auth, CC BY-SA), and it mirrors the
          &ldquo;Wikipedia&rdquo; and &ldquo;Indic Literature&rdquo; slices of Session 3&apos;s data mix.
        </div>
      </Reveal>
    </Section>
  );
}

function Tag({ children, icon }: { children: ReactNode; icon?: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-white/5 px-3 py-1 text-slate-300">
      {icon}
      {children}
    </span>
  );
}
