export interface LangName {
  code: string;
  name: string;
}

export interface Meta {
  session: number;
  title: string;
  dataset: string;
  dataset_url: string;
  license: string;
  languages: LangName[];
  char_budget_per_language: number;
  generated_at: string;
  runtime_seconds: number;
}

export interface Stage {
  id: string;
  title: string;
  short: string;
  is_active_cleanup: boolean;
  seconds: number;
  docs_in: number;
  docs_out: number;
  removed: number;
  removed_reasons?: Record<string, number>;
  // stage-specific extras (all optional):
  chars_out?: number;
  chars_in?: number;
  chars_normalized_away?: number;
  counters?: Record<string, number>;
  per_language_docs?: Record<string, number>;
  provenance?: string;
  license?: string;
  review_pool?: Record<string, number>;
  method?: string;
  score_deciles?: number[];
  thresholds?: Record<string, number>;
  pii_redactions?: Record<string, number>;
  docs_with_pii_redacted?: number;
  secret_redactions?: Record<string, number>;
  docs_with_secret_redacted?: number;
  source_license?: string;
  license_allow_listed?: boolean;
  note?: string;
  target_mix?: Record<string, number>;
  per_language_final_docs?: Record<string, number>;
  final_words?: number;
  final_chars?: number;
  estimated_tokens?: number;
  snapshot_sha256_16?: string;
  examples_removed?: Array<Record<string, string | number>>;
  example?: { before: string; after: string } | null;
}

export interface Stats {
  meta: Meta;
  strategy_count: { total_stages: number; active_cleanups: number };
  totals: {
    raw_documents: number;
    raw_characters: number;
    final_documents: number;
    final_characters: number;
    documents_removed: number;
    document_retention_pct: number;
    character_retention_pct: number;
  };
  per_language: {
    before: Record<string, number>;
    after: Record<string, number>;
    names: Record<string, string>;
  };
  stages: Stage[];
}
