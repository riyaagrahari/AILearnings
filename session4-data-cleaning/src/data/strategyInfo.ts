import {
  FileStack,
  Languages,
  Regex,
  CopyCheck,
  Gauge,
  ShieldAlert,
  GitBranch,
  Database,
  type LucideIcon,
} from "lucide-react";

/**
 * The "why / methods / trade-off" prose is carried over verbatim from Session 3
 * (`session3-report-training/src/data/cleaning.ts`) so this Session-4 widget
 * shows the *design rationale* alongside the *measured effect* of running it.
 */
export interface StrategyInfo {
  id: string;
  icon: LucideIcon;
  color: string;
  why: string;
  methods: string[];
  tradeoff: string;
}

export const STRATEGY_INFO: Record<string, StrategyInfo> = {
  raw: {
    id: "raw",
    icon: FileStack,
    color: "#64748b",
    why: "Heterogeneous crawl, code, books and Indic corpora arrive with wildly different encodings, structure and quality.",
    methods: [
      "Format-aware extraction (HTML/PDF/LaTeX/notebooks)",
      "Provenance & license tagging",
      "Content-hash assignment",
    ],
    tradeoff:
      "Aggressive extraction risks dropping structure (tables, code blocks); we keep structure-preserving parsers even when slower.",
  },
  langid: {
    id: "langid",
    icon: Languages,
    color: "#3b82f6",
    why: "Per-language routing is essential to protect Indic share and apply script-specific rules; mislabelled language poisons downstream mixing.",
    methods: [
      "Script + language classifier (Unicode-range based)",
      "Code-switch (Hinglish) detection",
      "Confidence thresholds with a human-audited review pool",
    ],
    tradeoff:
      "Short and code-mixed texts are hard to classify; we keep low-confidence docs in a review pool rather than silently discarding Indic data.",
  },
  normalize: {
    id: "normalize",
    icon: Regex,
    color: "#22d3ee",
    why: "Indic scripts encode the same grapheme many ways; without normalization the tokenizer wastes vocabulary and fertility inflates.",
    methods: [
      "NFC canonicalization",
      "Preserve ZWJ/ZWNJ (semantic in Indic scripts)",
      "Whitespace, quote and digit normalization",
    ],
    tradeoff:
      "Over-normalizing (e.g. stripping ZWJ) corrupts meaning; we normalize form but never collapse semantically distinct sequences.",
  },
  dedup: {
    id: "dedup",
    icon: CopyCheck,
    color: "#6366f1",
    why: "Duplicates waste compute, memorize verbatim text and skew the distribution toward whatever is most copied online.",
    methods: [
      "MinHash + LSH near-dedup",
      "Exact content-hash dedup",
      "Cross-split dedup to prevent eval leakage",
    ],
    tradeoff:
      "Too-aggressive dedup removes legitimately repeated facts/boilerplate structure; thresholds are tuned per-domain (code vs prose).",
  },
  quality: {
    id: "quality",
    icon: Gauge,
    color: "#818cf8",
    why: "A learned quality signal lets us upweight information-dense text and downweight spam without brittle hand-rules.",
    methods: [
      "Heuristic features (length, alpha-ratio, symbol-ratio, repetition)",
      "Per-document 0–1 quality score",
      "Per-language calibrated thresholds",
    ],
    tradeoff:
      "Quality models inherit annotator bias and can penalise dialectal Indic text; we calibrate per language and keep a diversity floor.",
  },
  safety: {
    id: "safety",
    icon: ShieldAlert,
    color: "#f472b6",
    why: "Remove extreme toxicity and high-risk PII before the model ever sees it — cheaper and safer than fixing it post-hoc.",
    methods: [
      "Multi-lingual toxicity classifiers",
      "PII detection & redaction (email, phone, Aadhaar, IP, card)",
      "Density-thresholded toxicity removal",
    ],
    tradeoff:
      "Over-filtering erases legitimate discussion of sensitive topics; we tune for precision on illegal content and defer nuance to alignment.",
  },
  code: {
    id: "code",
    icon: GitBranch,
    color: "#2dd4bf",
    why: "Code needs domain-specific gates: licenses, secrets and machine-generated files that generic text filters miss.",
    methods: [
      "SPDX license allow-listing",
      "Secret/key detection",
      "Static-analysis & test-signal quality gates",
    ],
    tradeoff:
      "Strict license filtering shrinks volume; we accept less code to eliminate legal and secret-leak risk.",
  },
  corpus: {
    id: "corpus",
    icon: Database,
    color: "#34d399",
    why: "The curated, mixed and scheduled corpus that feeds pre-training — versioned and reproducible.",
    methods: [
      "Domain rebalancing to target mix",
      "Curriculum ordering & annealing schedule",
      "Frozen, versioned snapshots",
    ],
    tradeoff:
      "Any fixed mix is a bet; we snapshot and ablate mixes on small proxies before committing to the full run.",
  },
};
