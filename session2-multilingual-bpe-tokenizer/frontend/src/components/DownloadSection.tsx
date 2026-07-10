import { downloadUrl, type DownloadFile } from "../api/client";
import { Card } from "./Card";

const DOWNLOADS: { file: DownloadFile; label: string; description: string }[] = [
  { file: "vocab.json", label: "vocab.json", description: "Token → id vocabulary" },
  { file: "merges.json", label: "merges.json", description: "Learned BPE merge rules, in rank order" },
  {
    file: "tokenizer.json",
    label: "tokenizer.json",
    description: "vocab + merges + training config, combined",
  },
];

export function DownloadSection() {
  return (
    <Card title="Download the tokenizer" subtitle="Straight from backend/artifacts/ — nothing regenerated">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        {DOWNLOADS.map(({ file, label, description }) => (
          <a
            key={file}
            href={downloadUrl(file)}
            download={file}
            className="group flex flex-col justify-between rounded-xl border border-slate-200 p-4 transition hover:border-indigo-300 hover:bg-indigo-50/50 dark:border-slate-700 dark:hover:border-indigo-600 dark:hover:bg-indigo-500/10"
          >
            <div className="flex items-center gap-2">
              <svg
                className="h-5 w-5 flex-none text-slate-400 transition group-hover:text-indigo-500"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  d="M10 3a.75.75 0 01.75.75v6.638l1.96-2.158a.75.75 0 111.08 1.04l-3.25 3.5a.75.75 0 01-1.08 0l-3.25-3.5a.75.75 0 111.08-1.04l1.96 2.158V3.75A.75.75 0 0110 3z"
                  clipRule="evenodd"
                />
                <path d="M3.5 12.75a.75.75 0 01.75.75v2.5c0 .138.112.25.25.25h11c.138 0 .25-.112.25-.25v-2.5a.75.75 0 011.5 0v2.5A1.75 1.75 0 0115.5 18h-11A1.75 1.75 0 012.75 16v-2.5a.75.75 0 01.75-.75z" />
              </svg>
              <span className="font-mono-token text-sm font-semibold text-slate-900 dark:text-slate-100">
                {label}
              </span>
            </div>
            <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">{description}</p>
          </a>
        ))}
      </div>
    </Card>
  );
}
