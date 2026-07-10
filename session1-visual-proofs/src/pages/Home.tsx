import type React from "react";
import { Target, Layers, Network, SplitSquareHorizontal, Sparkles } from "lucide-react";
import ExperimentCard from "../components/ExperimentCard";

const EXPERIMENTS = [
  {
    to: "/relu",
    index: "S1-1",
    icon: Target,
    accent: "bg-cyan-500",
    glow: "hover:shadow-[0_30px_90px_-40px_rgba(34,211,238,0.35)]",
    title: "Activations exist for a reason",
    claim:
      "A model with no nonlinearity can only draw a straight boundary, so it cannot separate two interleaved rings.",
    proof:
      "Linear + sigmoid gets stuck near 55% with a straight line. Add one ReLU hidden layer and the boundary wraps the ring to ~99% — only the activation changed.",
    tags: ["Concentric circles", "Linear vs ReLU", "Decision boundary"],
  },
  {
    to: "/depth",
    index: "S1-2",
    icon: Layers,
    accent: "bg-emerald-500",
    glow: "hover:shadow-[0_30px_90px_-40px_rgba(52,211,153,0.35)]",
    title: "Depth without nonlinearity is a lie",
    claim:
      "Five stacked linear layers collapse to a single linear map — a 5-layer linear net is no stronger than 1 layer.",
    proof:
      "1-layer and 5-linear-layer accuracy and boundaries are identical (both a line). Insert ReLUs and it suddenly solves the ring. Bonus: the five weight matrices multiply into exactly one matrix.",
    tags: ["Matrix composition", "1 vs 5 layers", "ReLU breaks the tie"],
  },
  {
    to: "/embeddings",
    index: "S1-3",
    icon: Network,
    accent: "bg-violet-500",
    glow: "hover:shadow-[0_30px_90px_-40px_rgba(167,139,250,0.35)]",
    title: "Embeddings learn similarity from nothing but next-token",
    claim:
      "Trained only to predict the next token in a tiny synthetic grammar, the embedding table clusters related tokens.",
    proof:
      "Animals, fruits, and verbs land in their own clusters after 2D projection — even though similarity was never supplied as a label. Emergent clustering is the proof.",
    tags: ["Toy grammar", "Embedding table", "Nearest neighbours"],
  },
  {
    to: "/generalization",
    index: "S1-4",
    icon: SplitSquareHorizontal,
    accent: "bg-sky-500",
    glow: "hover:shadow-[0_30px_90px_-40px_rgba(56,189,248,0.35)]",
    title: "Memorization vs generalization, and data closes the gap",
    claim:
      "A high-capacity model on tiny data drives train loss to ~0 while held-out loss stays high — until data grows.",
    proof:
      "At 20 samples the train/test gap is huge. At 200 it shrinks. At 2000 it nearly closes. Data is everything.",
    tags: ["20 / 200 / 2000 samples", "Train vs test", "Generalization gap"],
  },
];

export default function Home(): React.ReactElement {
  return (
    <div className="px-5 py-14 text-white sm:py-20">
      <div className="mx-auto max-w-5xl text-center">
        <div className="mx-auto inline-flex items-center gap-2 rounded-full border border-white/10 bg-slate-900/70 px-4 py-2 text-xs font-semibold uppercase tracking-[0.3em] text-slate-300">
          <Sparkles className="h-3.5 w-3.5 text-cyan-300" />
          Four visual proofs, four claims
        </div>

        <h1 className="mt-6 text-4xl font-semibold leading-tight tracking-tight sm:text-6xl">
          Visual Proofs of{" "}
          <span className="bg-gradient-to-r from-cyan-300 via-violet-300 to-sky-300 bg-clip-text text-transparent">
            Deep Learning
          </span>
        </h1>

        <p className="mx-auto mt-6 max-w-2xl text-base leading-7 text-slate-300 sm:text-lg">
          Four claims about neural networks, each trained live in your browser and proven with a picture —
          not a citation. Nonlinearity, depth, embeddings, and generalization: watch the boundary bend,
          the matrices collapse, the clusters emerge, and the gap close.
        </p>
      </div>

      <div className="mx-auto mt-14 grid max-w-6xl gap-6 sm:grid-cols-2">
        {EXPERIMENTS.map((experiment) => (
          <ExperimentCard key={experiment.to} {...experiment} />
        ))}
      </div>

      <div className="mx-auto mt-16 max-w-4xl rounded-3xl border border-white/10 bg-slate-950/80 p-8 text-center shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
        <p className="text-xs font-semibold uppercase tracking-[0.3em] text-slate-500">Why this matters</p>
        <p className="mt-4 text-lg leading-8 text-slate-200">
          Every model here trains for real, in-browser, with TensorFlow.js — no pre-baked results. Pick an
          experiment, hit train, and watch the claim get proven or broken in front of you.
        </p>
      </div>
    </div>
  );
}
