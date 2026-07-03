import type React from "react";
import { useEffect, useRef, useState } from "react";
import { Database, Play, StopCircle, RefreshCcw, Sparkles, Layers } from "lucide-react";
import Plot from "react-plotly.js";
import type { LayersModel, Tensor } from "@tensorflow/tfjs";

type Category = "Animals" | "Fruits" | "Verbs";

type TokenPoint = {
  token: string;
  x: number;
  y: number;
  category: Category;
  index: number;
};

type DatasetInfo = {
  inputs: number[];
  labels: number[];
  sentenceCount: number;
  pairCount: number;
};

const TOKENS = ["cat", "dog", "cow", "apple", "mango", "banana", "eat", "chase", "see"];
const CATEGORY_MAP: Record<string, Category> = {
  cat: "Animals",
  dog: "Animals",
  cow: "Animals",
  apple: "Fruits",
  mango: "Fruits",
  banana: "Fruits",
  eat: "Verbs",
  chase: "Verbs",
  see: "Verbs",
};

const COLORS: Record<Category, string> = {
  Animals: "#66b3ff",
  Fruits: "#ff8a50",
  Verbs: "#34d399",
};

const TEMPLATES = [
  ["cat", "chase", "dog"],
  ["dog", "chase", "cat"],
  ["cow", "eat", "apple"],
  ["cow", "eat", "mango"],
  ["dog", "eat", "apple"],
  ["cat", "eat", "mango"],
  ["apple", "see", "cow"],
  ["mango", "see", "dog"],
  ["banana", "see", "cow"],
];

const TOKEN_TO_INDEX = TOKENS.reduce<Record<string, number>>((acc, token, index) => {
  acc[token] = index;
  return acc;
}, {} as Record<string, number>);

const vocabularySize = TOKENS.length;

function generateDataset(sentenceCount: number): DatasetInfo {
  const inputs: number[] = [];
  const labels: number[] = [];

  for (let i = 0; i < sentenceCount; i++) {
    const sentence = TEMPLATES[Math.floor(Math.random() * TEMPLATES.length)];
    for (let j = 0; j < sentence.length - 1; j++) {
      inputs.push(TOKEN_TO_INDEX[sentence[j]]);
      labels.push(TOKEN_TO_INDEX[sentence[j + 1]]);
    }
  }

  return {
    inputs,
    labels,
    sentenceCount,
    pairCount: inputs.length,
  };
}

export default function EmbeddingExperiment(): React.ReactElement {
  const [dataset, setDataset] = useState<DatasetInfo>(() => generateDataset(90));
  const [sentenceCount, setSentenceCount] = useState<number>(90);
  const [embeddingDim, setEmbeddingDim] = useState<number>(8);
  const [epochs, setEpochs] = useState<number>(25);
  const [batchSize, setBatchSize] = useState<number>(16);

  const [epochCount, setEpochCount] = useState<number>(0);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [trainingStatus, setTrainingStatus] = useState<string>("Idle");
  const [isReady, setIsReady] = useState<boolean>(false);

  const [tokenPoints, setTokenPoints] = useState<TokenPoint[] | null>(null);
  const [selectedToken, setSelectedToken] = useState<string | null>(null);
  const [nearestNeighbours, setNearestNeighbours] = useState<string[]>([]);
  const [embeddingVectors, setEmbeddingVectors] = useState<number[][]>([]);

  const modelRef = useRef<LayersModel | null>(null);
  const tensorsRef = useRef<{ xs?: Tensor; ys?: Tensor } | null>(null);
  const stopRef = useRef(false);
  const tfRef = useRef<any>(null);

  async function ensureTfLoaded() {
    if (!tfRef.current) {
      tfRef.current = await import("@tensorflow/tfjs");
    }
  }

  useEffect(() => {
    return () => {
      tensorsRef.current?.xs?.dispose?.();
      tensorsRef.current?.ys?.dispose?.();
      modelRef.current?.dispose();
    };
  }, []);

  async function buildModel() {
    await ensureTfLoaded();
    const tf = tfRef.current;

    modelRef.current?.dispose();
    const model = tf.sequential();
    model.add(
      tf.layers.embedding({
        inputDim: vocabularySize,
        outputDim: embeddingDim,
        inputLength: 1,
      }),
    );
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: vocabularySize, activation: "softmax" }));
    model.compile({ optimizer: tf.train.adam(), loss: "sparseCategoricalCrossentropy" });
    modelRef.current = model;
    setTokenPoints(null);
    setSelectedToken(null);
    setNearestNeighbours([]);
    setEmbeddingVectors([]);
  }

  async function prepareTensors() {
    await ensureTfLoaded();
    const tf = tfRef.current;

    tensorsRef.current?.xs?.dispose?.();
    tensorsRef.current?.ys?.dispose?.();

    tensorsRef.current = {
      xs: tf.tensor2d(dataset.inputs, [dataset.inputs.length, 1], "int32"),
      ys: tf.tensor1d(dataset.labels, "int32"),
    };
  }

  async function resetExperiment() {
    setEpochCount(0);
    setLossHistory([]);
    setTrainingStatus("Idle");
    setIsReady(false);
    setTokenPoints(null);
    setSelectedToken(null);
    setNearestNeighbours([]);
    await buildModel();
    await prepareTensors();
    setIsReady(true);
  }

  useEffect(() => {
    resetExperiment();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [embeddingDim]);

  useEffect(() => {
    (async () => {
      await prepareTensors();
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset]);

  async function generateNewDataset() {
    const updated = generateDataset(sentenceCount);
    setDataset(updated);
    setEpochCount(0);
    setLossHistory([]);
    setTrainingStatus("Idle");
    setIsReady(false);
    setTokenPoints(null);
    setSelectedToken(null);
    setNearestNeighbours([]);
    await buildModel();
    await prepareTensors();
    setIsReady(true);
  }

  async function trainModel() {
    if (!isReady) {
      await resetExperiment();
    }

    if (!modelRef.current || !tensorsRef.current) return;
    await ensureTfLoaded();
    const tf = tfRef.current;

    setTrainingStatus("Training");
    stopRef.current = false;

    const xs = tensorsRef.current.xs!;
    const ys = tensorsRef.current.ys!;

    for (let epoch = 0; epoch < epochs; epoch++) {
      if (stopRef.current) break;
      const history = await modelRef.current.fit(xs, ys, { epochs: 1, batchSize, shuffle: true });
      const loss = history.history.loss ? (history.history.loss as number[])[0] : NaN;

      setEpochCount(epoch + 1);
      setLossHistory((h) => [...h, Number(loss)]);
      setTrainingStatus(`Training (epoch ${epoch + 1}/${epochs})`);

      if (tf.nextFrame) await tf.nextFrame();
    }

    setTrainingStatus(stopRef.current ? "Stopped" : "Complete");
    if (!stopRef.current) {
      await renderEmbeddings();
    }
  }

  async function renderEmbeddings() {
    if (!modelRef.current) return;
    await ensureTfLoaded();
    const tf = tfRef.current;

    const embeddingLayer = modelRef.current.layers[0];
    const weights = (embeddingLayer as any).getWeights?.();
    if (!weights || weights.length === 0) return;

    const embeddings = weights[0] as Tensor;
    const projected = tf.tidy(() => {
      const limited = embeddings.slice([0, 0], [vocabularySize, Math.min(2, embeddingDim)]);
      if (embeddingDim >= 2) return limited;
      return tf.pad(limited, [[0, 0], [0, 2 - embeddingDim]]);
    });

    const values = (await projected.array()) as number[][];
    const vectors = (await embeddings.array()) as number[][];
    projected.dispose();

    const points = TOKENS.map((token, index) => ({
      token,
      x: values[index][0],
      y: values[index][1],
      category: CATEGORY_MAP[token],
      index,
    }));

    setEmbeddingVectors(vectors);
    setTokenPoints(points);
  }

  function selectToken(pointIndex: number) {
    if (!tokenPoints?.length) return;
    const token = tokenPoints[pointIndex].token;
    setSelectedToken(token);

    const selectedIndex = tokenPoints[pointIndex].index;
    const selectedVector = embeddingVectors[selectedIndex];
    if (!selectedVector) return;

    const neighbours = embeddingVectors
      .map((vector, index) => ({
        token: TOKENS[index],
        distance: Math.hypot(...vector.map((v, i) => v - selectedVector[i])),
      }))
      .filter((item) => item.token !== token)
      .sort((a, b) => a.distance - b.distance)
      .slice(0, 3)
      .map((item) => item.token);

    setNearestNeighbours(neighbours);
  }

  const progress = epochs > 0 ? Math.min(100, Math.round((epochCount / epochs) * 100)) : 0;

  return (
    <div className="p-5 text-white" style={{ fontFamily: "Inter, Arial" }}>
      <div className="grid gap-8 xl:grid-cols-[1.7fr_1fr]">
        <div className="space-y-5">
          <div className="inline-flex items-center gap-3 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-4 py-2 text-sm font-semibold uppercase tracking-[0.28em] text-emerald-300 shadow-sm shadow-emerald-400/10">
            <span>Experiment 3</span>
          </div>
          <p className="text-sm uppercase tracking-[0.28em] text-slate-400">Embeddings Learn Similarity from Nothing but Next Token</p>
          <h1 className="max-w-2xl text-4xl font-semibold leading-tight text-white sm:text-5xl">Embeddings Learn Similarity from Nothing but Next Token</h1>
          <p className="max-w-2xl text-sm leading-7 text-slate-300 sm:text-base">
            A network trained only to predict the next token learns useful word embeddings automatically. Similar words cluster together even though no similarity labels are provided.
          </p>
        </div>

        <div className="grid gap-4 sm:grid-cols-3">
          <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
            <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Vocabulary</p>
            <p className="mt-3 text-lg font-semibold text-white">{vocabularySize} tokens</p>
          </div>

          <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
            <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Categories</p>
            <p className="mt-3 text-lg font-semibold text-white">Animals, Fruits, Verbs</p>
          </div>

          <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
            <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Model</p>
            <p className="mt-3 text-lg font-semibold text-white">Embedding → Dense → Softmax</p>
          </div>
        </div>
      </div>

      <div className="mt-8 rounded-3xl border border-white/10 bg-slate-950/80 p-6 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <div className="text-sm uppercase tracking-[0.18em] text-slate-400">Epoch</div>
            <div className="mt-2 text-3xl font-semibold text-white">{epochCount} / {epochs}</div>
          </div>
          <div className="rounded-2xl bg-slate-900/90 px-4 py-2 text-sm font-semibold text-slate-200 ring-1 ring-slate-700">
            {trainingStatus}
          </div>
        </div>

        <div className="mt-6">
          <div className="mb-3 flex items-center justify-between gap-2 text-sm font-semibold uppercase tracking-[0.18em] text-slate-400">
            <span>Progress</span>
            <span>{progress}%</span>
          </div>
          <div className="h-3 overflow-hidden rounded-full bg-slate-900/90 ring-1 ring-slate-700">
            <div className="h-full rounded-full bg-gradient-to-r from-emerald-400 via-green-500 to-emerald-500 transition-all duration-700 ease-out" style={{ width: `${progress}%` }} />
          </div>
        </div>
      </div>

      <div className="mt-10 flex flex-col gap-16 xl:flex-row">
        <div className="flex flex-col gap-6 xl:w-96">
          <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
            <div className="mb-4 flex items-center gap-3 text-sm font-semibold uppercase tracking-[0.15em] text-slate-300">
              <Database className="h-4 w-4 text-emerald-300" />
              <span>Dataset</span>
            </div>
            <div className="grid gap-4">
              <div className="space-y-2">
                <label htmlFor="sentenceCount" className="text-sm font-medium text-slate-100">Sentence templates</label>
                <input id="sentenceCount" type="number" min={30} max={200} step={10} value={sentenceCount} onChange={(e) => setSentenceCount(Number(e.target.value))} className="w-full rounded-2xl border border-slate-700 bg-slate-900/90 px-4 py-2 text-slate-100 outline-none transition focus:border-emerald-400 focus:ring-2 focus:ring-emerald-500/20" />
                <p className="text-xs text-slate-500">Number of random sentences used to create next-token pairs.</p>
              </div>
              <div className="space-y-2">
                <label htmlFor="embeddingDim" className="text-sm font-medium text-slate-100">Embedding dimension</label>
                <div className="flex items-center gap-3">
                  <input id="embeddingDim" type="range" min={2} max={16} step={1} value={embeddingDim} onChange={(e) => setEmbeddingDim(Number(e.target.value))} className="w-full accent-emerald-400" />
                  <span className="text-sm text-slate-400">{embeddingDim}</span>
                </div>
                <p className="text-xs text-slate-500">Embedding vector dimension used for token representations.</p>
              </div>
            </div>
          </div>

          <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
            <div className="mb-4 flex items-center gap-3 text-sm font-semibold uppercase tracking-[0.15em] text-slate-300">
              <Sparkles className="h-4 w-4 text-emerald-300" />
              <span>Actions</span>
            </div>
            <div className="grid gap-3">
              <button type="button" onClick={generateNewDataset} className="inline-flex items-center justify-center gap-2 rounded-2xl bg-emerald-500 px-4 py-3 text-sm font-semibold text-slate-900 shadow-md shadow-emerald-500/20 transition transform-gpu hover:-translate-y-0.5 hover:bg-emerald-400 active:scale-95 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400/60 focus-visible:ring-offset-2">
                <RefreshCcw className="h-4 w-4" />
                Generate Dataset
              </button>
              <button type="button" onClick={trainModel} className="inline-flex items-center justify-center gap-2 rounded-2xl bg-cyan-500 px-4 py-3 text-sm font-semibold text-slate-900 shadow-md shadow-cyan-500/20 transition transform-gpu hover:-translate-y-0.5 hover:bg-cyan-400 active:scale-95 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/60 focus-visible:ring-offset-2">
                <Play className="h-4 w-4" />
                Train
              </button>
              <button type="button" onClick={resetExperiment} className="inline-flex items-center justify-center gap-2 rounded-2xl bg-slate-800 px-4 py-3 text-sm font-semibold text-slate-100 transition transform-gpu hover:-translate-y-0.5 hover:bg-slate-700 active:scale-95 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-500/50 focus-visible:ring-offset-2">
                <StopCircle className="h-4 w-4" />
                Reset
              </button>
            </div>
          </div>

          <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
            <div className="mb-4 flex items-center gap-3 text-sm font-semibold uppercase tracking-[0.15em] text-slate-300">
              <Layers className="h-4 w-4 text-amber-300" />
              <span>Training</span>
            </div>
            <div className="space-y-4">
              <div className="space-y-2">
                <label htmlFor="epochs" className="text-sm font-medium text-slate-100">Epochs</label>
                <input id="epochs" type="number" min={1} max={100} value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} className="w-full rounded-2xl border border-slate-700 bg-slate-900/90 px-4 py-2 text-slate-100 outline-none transition focus:border-emerald-400 focus:ring-2 focus:ring-emerald-500/20" />
                <p className="text-xs text-slate-500">Number of passes over the generated next-token pairs.</p>
              </div>
              <div className="space-y-2">
                <label htmlFor="batchSize" className="text-sm font-medium text-slate-100">Batch Size</label>
                <input id="batchSize" type="number" min={4} max={64} step={1} value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value))} className="w-full rounded-2xl border border-slate-700 bg-slate-900/90 px-4 py-2 text-slate-100 outline-none transition focus:border-emerald-400 focus:ring-2 focus:ring-emerald-500/20" />
                <p className="text-xs text-slate-500">Number of examples in each training update.</p>
              </div>
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-6">
          <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
            <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
              <div className="text-sm uppercase tracking-[0.18em] text-slate-400">Training Pairs</div>
              <div className="mt-3 text-3xl font-semibold text-white">{dataset.pairCount}</div>
              <div className="mt-2 text-sm text-slate-500">Input → next token pairs from randomly sampled template sentences.</div>
            </div>
            <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
              <div className="text-sm uppercase tracking-[0.18em] text-slate-400">Vocabulary</div>
              <div className="mt-3 text-3xl font-semibold text-white">{vocabularySize}</div>
              <div className="mt-2 text-sm text-slate-500">Three categories of words used by the model.</div>
            </div>
          </div>

          <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
            <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-4 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
              <Plot
                data={[
                  {
                    x: lossHistory.map((_, index) => index + 1),
                    y: lossHistory,
                    name: "Loss",
                    type: "scatter",
                    mode: "lines+markers",
                    marker: { color: "#66b3ff" },
                    line: { color: "#66b3ff" },
                  },
                ]}
                layout={{
                  width: 340,
                  height: 180,
                  margin: { l: 30, r: 10, t: 20, b: 30 },
                  paper_bgcolor: "#041026",
                  plot_bgcolor: "#041026",
                  xaxis: { title: "Epoch", color: "#94a3b8" },
                  yaxis: { title: "Loss", color: "#94a3b8" },
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ background: "#041026" }}
              />
            </div>
            <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-4 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
              <div className="font-semibold text-white mb-3">Embedding Visualization</div>
              <Plot
                data={[
                  {
                    x: tokenPoints?.map((point) => point.x) ?? [],
                    y: tokenPoints?.map((point) => point.y) ?? [],
                    text: tokenPoints?.map((point) => point.token) ?? [],
                    mode: "markers+text",
                    type: "scatter",
                    textposition: "top center",
                    marker: {
                      size: 12,
                      color: tokenPoints?.map((point) => COLORS[point.category]) ?? [],
                      line: { width: 1, color: "rgba(255,255,255,0.15)" },
                    },
                  },
                ]}
                layout={{
                  width: 340,
                  height: 220,
                  margin: { l: 30, r: 10, t: 20, b: 30 },
                  paper_bgcolor: "#041026",
                  plot_bgcolor: "#041026",
                  xaxis: { visible: true, color: "#94a3b8", title: "Dim 1" },
                  yaxis: { visible: true, color: "#94a3b8", title: "Dim 2" },
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ background: "#041026" }}
                onClick={(event) => {
                  const pointIndex = event.points?.[0]?.pointIndex;
                  if (pointIndex !== undefined) selectToken(pointIndex);
                }}
              />
            </div>
          </div>

          <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-6 shadow-[0_24px_100px_-60px_rgba(0,0,0,0.8)] backdrop-blur-xl">
            <div className="font-semibold text-white">Nearest Neighbours</div>
            {selectedToken ? (
              <div className="mt-3 space-y-2 text-sm text-slate-300">
                <div className="text-slate-100">Selected token: {selectedToken}</div>
                {nearestNeighbours.length > 0 ? (
                  <ol className="list-decimal list-inside space-y-1">
                    {nearestNeighbours.map((token) => (
                      <li key={token}>{token}</li>
                    ))}
                  </ol>
                ) : (
                  <div className="text-slate-500">Click a token in the plot to view nearest neighbours.</div>
                )}
              </div>
            ) : (
              <div className="mt-3 text-sm text-slate-500">Train the model and click a point to reveal its nearest neighbours.</div>
            )}
          </div>

          <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-6 shadow-[0_24px_100px_-60px_rgba(0,0,0,0.8)] backdrop-blur-xl">
            <div className="font-semibold text-white">Educational Summary</div>
            <div className="mt-3 space-y-3 text-sm text-slate-300">
              <div>
                Claim: A neural network trained only to predict the next token learns useful word embeddings automatically.
              </div>
              <div>
                Observation: Animals cluster together. Fruits cluster together. Verbs cluster together.
              </div>
              <div>
                Conclusion: Although the model never received similarity labels, it learned semantic structure because words sharing similar next-token distributions receive similar embeddings.
              </div>
              <div>
                Final conclusion: Word embeddings emerge naturally from next-token prediction.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
