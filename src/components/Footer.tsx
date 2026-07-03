import type React from "react";

export default function Footer(): React.ReactElement {
  return (
    <footer className="mt-12 border-t border-white/6 pt-6 pb-10">
      <div className="mx-auto flex max-w-7xl flex-col gap-4 px-5 sm:flex-row sm:items-center sm:justify-between">
        <div className="text-sm font-medium text-slate-300">Interactive Deep Learning Visualization</div>
        <div className="flex flex-col items-start gap-3 sm:flex-row sm:items-center">
          <div className="mr-3 text-xs text-slate-400">Built with</div>
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-sm font-semibold text-white">React 19</span>
            <span className="text-sm font-semibold text-white">TypeScript</span>
            <span className="text-sm font-semibold text-white">TensorFlow.js</span>
            <span className="text-sm font-semibold text-white">TailwindCSS</span>
          </div>
        </div>
      </div>
      <div className="mt-4 text-center text-xs text-slate-500">
        © {new Date().getFullYear()} Visual Proofs — Interactive Deep Learning
      </div>
    </footer>
  );
}
