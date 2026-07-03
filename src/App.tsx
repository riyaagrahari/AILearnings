import ReLUExperiment from "./pages/ReLUExperiment";
import LinearDepthExperiment from "./pages/LinearDepthExperiment";
import EmbeddingExperiment from "./pages/EmbeddingExperiment";
import GeneralizationExperiment from "./pages/GeneralizationExperiment";
import { useState } from "react";

function App() {
  const [page, setPage] = useState<"relu" | "depth" | "embedding" | "generalization">("relu");

  return (
    <div>
      <div className="p-4 flex gap-2">
        <button
          type="button"
          onClick={() => setPage("relu")}
          className={`px-3 py-1 rounded-md ${page === "relu" ? "bg-cyan-500 text-slate-900" : "bg-slate-800 text-slate-200"}`}
        >
          ReLU Experiment
        </button>
        <button
          type="button"
          onClick={() => setPage("depth")}
          className={`px-3 py-1 rounded-md ${page === "depth" ? "bg-emerald-500 text-slate-900" : "bg-slate-800 text-slate-200"}`}
        >
          Depth (Linear) Experiment
        </button>
        <button
          type="button"
          onClick={() => setPage("embedding")}
          className={`px-3 py-1 rounded-md ${page === "embedding" ? "bg-violet-500 text-slate-900" : "bg-slate-800 text-slate-200"}`}
        >
          Embedding Experiment
        </button>
        <button
          type="button"
          onClick={() => setPage("generalization")}
          className={`px-3 py-1 rounded-md ${page === "generalization" ? "bg-sky-500 text-slate-900" : "bg-slate-800 text-slate-200"}`}
        >
          Generalization Experiment
        </button>
      </div>

      {page === "relu" ? (
        <ReLUExperiment />
      ) : page === "depth" ? (
        <LinearDepthExperiment />
      ) : page === "embedding" ? (
        <EmbeddingExperiment />
      ) : (
        <GeneralizationExperiment />
      )}
    </div>
  );
}

export default App;