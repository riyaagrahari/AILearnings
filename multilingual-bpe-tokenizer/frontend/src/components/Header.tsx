import { API_BASE_URL } from "../api/client";
import { ThemeToggle } from "./ThemeToggle";

interface HeaderProps {
  connected: boolean | null; // null = still checking
}

export function Header({ connected }: HeaderProps) {
  return (
    <header className="border-b border-slate-200 bg-white/80 backdrop-blur-sm dark:border-slate-800 dark:bg-slate-950/80">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-4 sm:px-6">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 flex-none items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 text-sm font-bold text-white shadow-md shadow-indigo-500/30">
            BPE
          </div>
          <div>
            <h1 className="text-lg font-bold leading-tight text-slate-900 dark:text-slate-50 sm:text-xl">
              Shared Multilingual BPE Tokenizer
            </h1>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              English &middot; Hindi &middot; Telugu &middot; Tamil — built entirely from scratch
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <span
            className={`hidden items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium sm:inline-flex ${
              connected === null
                ? "bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400"
                : connected
                  ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300"
                  : "bg-rose-100 text-rose-700 dark:bg-rose-900/40 dark:text-rose-300"
            }`}
            title={API_BASE_URL}
          >
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                connected === null
                  ? "bg-slate-400"
                  : connected
                    ? "bg-emerald-500"
                    : "bg-rose-500"
              }`}
            />
            {connected === null ? "Checking API…" : connected ? "API connected" : "API offline"}
          </span>
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
