import type React from "react";
import { Link } from "react-router-dom";
import { ArrowUpRight, type LucideIcon } from "lucide-react";

export interface ExperimentCardProps {
  to: string;
  index: string;
  icon: LucideIcon;
  accent: string;
  glow: string;
  title: string;
  claim: string;
  proof: string;
  tags: string[];
}

export default function ExperimentCard({
  to,
  index,
  icon: Icon,
  accent,
  glow,
  title,
  claim,
  proof,
  tags,
}: ExperimentCardProps): React.ReactElement {
  return (
    <Link
      to={to}
      className={`group relative flex flex-col gap-5 overflow-hidden rounded-3xl border border-white/10 bg-slate-950/80 p-6 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl transition duration-300 hover:-translate-y-1 hover:border-white/20 ${glow}`}
    >
      <div
        className={`pointer-events-none absolute -right-16 -top-16 h-40 w-40 rounded-full ${accent} opacity-20 blur-3xl transition duration-300 group-hover:opacity-30`}
      />

      <div className="flex items-center justify-between">
        <span className={`inline-flex items-center gap-2 rounded-full border border-white/10 bg-slate-900/80 px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] ${accent.replace("bg-", "text-")}`}>
          {index}
        </span>
        <span className={`flex h-10 w-10 items-center justify-center rounded-2xl ${accent} bg-opacity-15 text-white`}>
          <Icon className="h-5 w-5" />
        </span>
      </div>

      <div>
        <h3 className="text-xl font-semibold text-white">{title}</h3>
        <p className="mt-2 text-sm leading-6 text-slate-300">{claim}</p>
      </div>

      <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">The proof</p>
        <p className="mt-2 text-sm leading-6 text-slate-200">{proof}</p>
      </div>

      <div className="flex flex-wrap gap-2">
        {tags.map((tag) => (
          <span key={tag} className="rounded-full border border-slate-700 bg-slate-900/80 px-3 py-1 text-xs text-slate-400">
            {tag}
          </span>
        ))}
      </div>

      <div className="mt-auto flex items-center gap-2 text-sm font-semibold text-white">
        Run the experiment
        <ArrowUpRight className="h-4 w-4 transition group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
      </div>
    </Link>
  );
}
