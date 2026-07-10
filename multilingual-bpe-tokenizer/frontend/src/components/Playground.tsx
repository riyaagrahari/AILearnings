import { useState } from "react";
import { ApiError, tokenizeText } from "../api/client";
import type { TokenizeResponse } from "../types";
import { Card } from "./Card";
import { CopyButton } from "./CopyButton";
import { Spinner } from "./StatusStates";

const PLACEHOLDER = "भारत is a great country.";

export function Playground() {
  const [text, setText] = useState(PLACEHOLDER);
  const [result, setResult] = useState<TokenizeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleEncode() {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const response = await tokenizeText(text);
      setResult(response);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to tokenize text.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card
      title="Tokenizer playground"
      subtitle="Runs the real tokenizer's encode()/decode() on whatever you type"
    >
      <div className="space-y-4">
        <textarea
          value={text}
          onChange={(event) => setText(event.target.value)}
          rows={3}
          placeholder="Type text in English, Hindi, Telugu or Tamil…"
          className="w-full resize-none rounded-xl border border-slate-200 bg-slate-50 p-3 text-sm text-slate-900 outline-none transition focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 dark:border-slate-700 dark:bg-slate-800/60 dark:text-slate-100 dark:focus:border-indigo-500 dark:focus:ring-indigo-900/40"
        />

        <div className="flex items-center justify-between gap-3">
          <button
            type="button"
            onClick={handleEncode}
            disabled={loading || !text.trim()}
            className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm shadow-indigo-500/30 transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading && <Spinner className="h-4 w-4" />}
            Encode
          </button>
          {result && (
            <span className="text-xs text-slate-500 dark:text-slate-400">
              {result.tokens.length} token{result.tokens.length === 1 ? "" : "s"} produced
            </span>
          )}
        </div>

        {error && (
          <p className="rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-700 dark:bg-rose-950/40 dark:text-rose-300">
            {error}
          </p>
        )}

        {result && (
          <div className="space-y-4 border-t border-slate-100 pt-4 dark:border-slate-800">
            <ResultRow label="Decoded output" copyText={result.decoded_text}>
              <p className="rounded-lg bg-slate-50 p-3 text-sm text-slate-800 dark:bg-slate-800/60 dark:text-slate-200">
                {result.decoded_text}
              </p>
            </ResultRow>

            <ResultRow label="Pretokens" copyText={result.pretokens.join(" | ")}>
              <TokenStrip items={result.pretokens} tone="slate" />
            </ResultRow>

            <ResultRow label="Tokens" copyText={result.tokens.join(" | ")}>
              <TokenStrip items={result.tokens} tone="indigo" />
            </ResultRow>

            <ResultRow label="Token IDs" copyText={result.ids.join(", ")}>
              <TokenStrip items={result.ids.map(String)} tone="violet" />
            </ResultRow>
          </div>
        )}
      </div>
    </Card>
  );
}

function ResultRow({
  label,
  copyText,
  children,
}: {
  label: string;
  copyText: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between">
        <p className="text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">
          {label}
        </p>
        <CopyButton text={copyText} />
      </div>
      {children}
    </div>
  );
}

function TokenStrip({ items, tone }: { items: string[]; tone: "slate" | "indigo" | "violet" }) {
  const toneClasses: Record<typeof tone, string> = {
    slate: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
    indigo: "bg-indigo-100 text-indigo-700 dark:bg-indigo-500/20 dark:text-indigo-300",
    violet: "bg-violet-100 text-violet-700 dark:bg-violet-500/20 dark:text-violet-300",
  };

  return (
    <div className="scroll-thin flex max-h-40 flex-wrap gap-1.5 overflow-y-auto rounded-lg bg-slate-50 p-3 dark:bg-slate-800/40">
      {items.map((item, index) => (
        <span
          key={`${item}-${index}`}
          className={`rounded-md px-2 py-1 font-mono-token text-xs ${toneClasses[tone]}`}
        >
          {item === "" ? "∅" : item}
        </span>
      ))}
    </div>
  );
}
