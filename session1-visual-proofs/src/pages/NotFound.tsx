import type React from "react";
import { Link } from "react-router-dom";
import { Compass } from "lucide-react";

export default function NotFound(): React.ReactElement {
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center gap-6 px-5 text-center text-white">
      <span className="flex h-14 w-14 items-center justify-center rounded-2xl border border-white/10 bg-slate-900/80 text-cyan-300">
        <Compass className="h-7 w-7" />
      </span>
      <div>
        <h1 className="text-3xl font-semibold">Page not found</h1>
        <p className="mt-3 max-w-md text-sm leading-6 text-slate-400">
          That route doesn&apos;t exist. Head back home to pick one of the four visual proofs.
        </p>
      </div>
      <Link
        to="/"
        className="rounded-2xl bg-cyan-500 px-5 py-2.5 text-sm font-semibold text-slate-950 shadow-md shadow-cyan-500/20 transition hover:-translate-y-0.5 hover:bg-cyan-400"
      >
        Back to home
      </Link>
    </div>
  );
}
