import { lazy, Suspense } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import Home from "./pages/Home";
import NotFound from "./pages/NotFound";

const ReLUExperiment = lazy(() => import("./pages/ReLUExperiment"));
const LinearDepthExperiment = lazy(() => import("./pages/LinearDepthExperiment"));
const EmbeddingExperiment = lazy(() => import("./pages/EmbeddingExperiment"));
const GeneralizationExperiment = lazy(() => import("./pages/GeneralizationExperiment"));

function PageLoadingFallback() {
  return (
    <div className="flex min-h-[60vh] items-center justify-center">
      <div className="flex items-center gap-3 text-slate-400">
        <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-cyan-400" />
        <span className="text-sm font-medium uppercase tracking-[0.2em]">Loading experiment…</span>
      </div>
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <div className="flex min-h-screen flex-col bg-slate-950">
        <Navbar />
        <main className="flex-1">
          <Suspense fallback={<PageLoadingFallback />}>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/relu" element={<ReLUExperiment />} />
              <Route path="/depth" element={<LinearDepthExperiment />} />
              <Route path="/embeddings" element={<EmbeddingExperiment />} />
              <Route path="/generalization" element={<GeneralizationExperiment />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </Suspense>
        </main>
        <Footer />
      </div>
    </BrowserRouter>
  );
}

export default App;
