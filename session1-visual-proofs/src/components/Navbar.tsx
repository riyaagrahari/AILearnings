import type React from "react";
import { NavLink } from "react-router-dom";
import { BrainCircuit } from "lucide-react";

const LINKS: { to: string; label: string; accent: string }[] = [
  { to: "/relu", label: "ReLU", accent: "hover:text-cyan-300" },
  { to: "/depth", label: "Depth", accent: "hover:text-emerald-300" },
  { to: "/embeddings", label: "Embeddings", accent: "hover:text-violet-300" },
  { to: "/generalization", label: "Generalization", accent: "hover:text-sky-300" },
];

export default function Navbar(): React.ReactElement {
  return (
    <header className="sticky top-0 z-50 border-b border-white/10 bg-slate-950/80 backdrop-blur-xl">
      <nav className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-5 py-4">
        <NavLink to="/" className="flex items-center gap-2.5 text-white">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-400 via-violet-400 to-sky-400 text-slate-950 shadow-md shadow-cyan-500/20">
            <BrainCircuit className="h-5 w-5" />
          </span>
          <span className="hidden text-sm font-semibold uppercase tracking-[0.2em] text-slate-200 sm:inline">
            Visual Proofs
          </span>
        </NavLink>

        <div className="flex flex-wrap items-center gap-1 rounded-2xl border border-white/10 bg-slate-900/60 p-1 text-sm font-medium">
          {LINKS.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              className={({ isActive }) =>
                `rounded-xl px-3 py-2 transition ${link.accent} ${
                  isActive ? "bg-slate-800 text-white" : "text-slate-400"
                }`
              }
            >
              {link.label}
            </NavLink>
          ))}
        </div>
      </nav>
    </header>
  );
}
