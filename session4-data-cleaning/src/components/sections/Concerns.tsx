import type { ReactNode } from "react";
import { EyeOff, Fingerprint, KeyRound, Languages, ShieldAlert } from "lucide-react";
import { Section } from "../layout/Section";
import { SectionHeading } from "../ui/SectionHeading";
import { Reveal } from "../ui/Reveal";
import { stats } from "../../data/stats";
import { comma, humanize } from "../../lib/format";

export function Concerns() {
  const langid = stats.stages.find((s) => s.id === "langid")!;
  const safety = stats.stages.find((s) => s.id === "safety")!;
  const code = stats.stages.find((s) => s.id === "code")!;
  const normalize = stats.stages.find((s) => s.id === "normalize")!;

  const review = langid.review_pool ?? {};
  const reviewTotal = Object.values(review).reduce((a, b) => a + b, 0);
  const pii = safety.pii_redactions ?? {};
  const piiTotal = Object.values(pii).reduce((a, b) => a + b, 0);

  return (
    <Section id="concerns">
      <SectionHeading
        index="04"
        eyebrow="Other strategies & concerns handled"
        title="Not everything is a delete."
        description="Several Session-3 concerns aren't about dropping documents — they're about routing, redacting, preserving and gate-keeping. Here's what those quieter strategies did to the corpus."
      />

      <div className="mt-10 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <Reveal>
          <ConcernCard
            icon={<Languages className="h-5 w-5" />}
            color="#3b82f6"
            title="Language review pool"
            headline={`${comma(reviewTotal)} docs`}
            sub="kept, not discarded — flagged for human audit"
          >
            <Rows
              rows={[
                ["Code-switch (heavy Latin)", review.code_switch ?? 0],
                ["Low-confidence language-ID", review.low_confidence ?? 0],
                ["Hindi/Marathi mismatch flagged", review.hi_mr_mismatch_flagged ?? 0],
              ]}
            />
          </ConcernCard>
        </Reveal>

        <Reveal delay={0.05}>
          <ConcernCard
            icon={<Fingerprint className="h-5 w-5" />}
            color="#f472b6"
            title="PII redaction"
            headline={`${comma(piiTotal)} spans`}
            sub={`redacted across ${comma(safety.docs_with_pii_redacted ?? 0)} documents`}
          >
            <Rows rows={Object.entries(pii).map(([k, v]) => [humanize(k), v])} />
          </ConcernCard>
        </Reveal>

        <Reveal delay={0.1}>
          <ConcernCard
            icon={<ShieldAlert className="h-5 w-5" />}
            color="#fb7185"
            title="Toxicity filtering"
            headline={`${comma(safety.removed)} docs`}
            sub="dropped above the toxicity-density threshold"
          >
            <p className="text-xs leading-relaxed text-slate-400">
              A conservative multilingual slur list drives a density gate. On curated Wikipedia this
              is deliberately near-zero — the guard matters far more on raw web crawl.
            </p>
          </ConcernCard>
        </Reveal>

        <Reveal delay={0.05}>
          <ConcernCard
            icon={<KeyRound className="h-5 w-5" />}
            color="#2dd4bf"
            title="Code gate (transferable)"
            headline={code.license_allow_listed ? "License OK" : "Blocked"}
            sub={`${code.source_license} is on the SPDX allow-list`}
          >
            <p className="text-xs leading-relaxed text-slate-400">
              {code.note} Secrets found: {comma(code.docs_with_secret_redacted ?? 0)}.
            </p>
          </ConcernCard>
        </Reveal>

        <Reveal delay={0.1}>
          <ConcernCard
            icon={<EyeOff className="h-5 w-5" />}
            color="#22d3ee"
            title="Script-integrity preservation"
            headline={`${comma(normalize.counters?.zwj_zwnj_preserved ?? 0)}`}
            sub="ZWJ / ZWNJ joiners kept intact (never stripped)"
          >
            <Rows
              rows={[
                ["Smart quotes normalized", normalize.counters?.quotes ?? 0],
                ["Whitespace chars collapsed", normalize.counters?.whitespace_chars ?? 0],
                ["NFC-changed documents", normalize.counters?.nfc_changed ?? 0],
              ]}
            />
          </ConcernCard>
        </Reveal>

        <Reveal delay={0.15}>
          <ConcernCard
            icon={<Languages className="h-5 w-5" />}
            color="#6366f1"
            title="Deduplication method"
            headline="MinHash + LSH"
            sub="built from scratch, no libraries"
          >
            <p className="text-xs leading-relaxed text-slate-400">
              {stats.stages.find((s) => s.id === "dedup")!.method}. Indian district stubs share
              near-identical template text — exactly the near-duplicates this catches.
            </p>
          </ConcernCard>
        </Reveal>
      </div>
    </Section>
  );
}

function ConcernCard({
  icon,
  color,
  title,
  headline,
  sub,
  children,
}: {
  icon: ReactNode;
  color: string;
  title: string;
  headline: string;
  sub: string;
  children: ReactNode;
}) {
  return (
    <div className="glass glass-hover flex h-full flex-col rounded-2xl p-5">
      <div className="flex items-center gap-2" style={{ color }}>
        {icon}
        <span className="text-xs font-semibold uppercase tracking-widest">{title}</span>
      </div>
      <div className="mt-3 text-2xl font-bold text-white">{headline}</div>
      <div className="text-xs text-slate-500">{sub}</div>
      <div className="mt-4 border-t border-white/5 pt-3">{children}</div>
    </div>
  );
}

function Rows({ rows }: { rows: [string, number][] }) {
  return (
    <ul className="space-y-1.5 text-xs">
      {rows.map(([label, value]) => (
        <li key={label} className="flex items-center justify-between gap-2">
          <span className="text-slate-400">{label}</span>
          <span className="font-mono text-slate-200">{comma(value)}</span>
        </li>
      ))}
    </ul>
  );
}
