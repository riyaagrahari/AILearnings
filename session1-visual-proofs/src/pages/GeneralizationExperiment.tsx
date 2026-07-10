import type React from "react";
import { useMemo, useRef, useState } from "react";
import { Database, Play, StopCircle, Sparkles, Layers, SlidersHorizontal } from "lucide-react";
import Plot from "react-plotly.js";
import type { Dataset } from "../experiments/relu/dataset";
import { generateConcentricCircles, datasetToTensors } from "../experiments/relu/dataset";
import type { LayersModel, Tensor } from "@tensorflow/tfjs";

const DATASET_SIZES = [20, 200, 2000] as const;
const RADIUS = 1.0;

type RunResult = {
  samples: number;
  trainLoss: number;
  testLoss: number;
  trainAccuracy: number;
  testAccuracy: number;
  accuracyGap: number;
};

function splitDataset(dataset: Dataset) {
  const total = dataset.points.length;
  const testCount = Math.max(1, Math.floor(total * 0.2));
  const trainCount = total - testCount;

  return {
    train: { points: dataset.points.slice(0, trainCount) },
    test: { points: dataset.points.slice(trainCount) },
  };
}

type TfNamespace = typeof import("@tensorflow/tfjs");

function buildGeneralizationModel(tf: TfNamespace): LayersModel {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 128, inputShape: [2], activation: "relu" }));
  model.add(tf.layers.dense({ units: 128, activation: "relu" }));
  model.add(tf.layers.dense({ units: 128, activation: "relu" }));
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
  return model as LayersModel;
}

async function computeAccuracy(model: LayersModel, xs: Tensor, labels: number[]) {
  const predictions = model.predict(xs) as Tensor;
  const values = await predictions.data();
  predictions.dispose();

  let correct = 0;
  for (let i = 0; i < values.length; i += 1) {
    const predicted = values[i] >= 0.5 ? 1 : 0;
    if (predicted === labels[i]) correct += 1;
  }

  return correct / Math.max(1, labels.length);
}

export default function GeneralizationExperiment(): React.ReactElement {
  const [datasetSize, setDatasetSize] = useState<number>(20);
  const [noise, setNoise] = useState<number>(0.08);
  const [epochs, setEpochs] = useState<number>(40);
  const [learningRate, setLearningRate] = useState<number>(0.01);
  const [batchSize, setBatchSize] = useState<number>(16);

  const [epochCount, setEpochCount] = useState<number>(0);
  const [trainLossHistory, setTrainLossHistory] = useState<number[]>([]);
  const [testLossHistory, setTestLossHistory] = useState<number[]>([]);
  const [trainAccHistory, setTrainAccHistory] = useState<number[]>([]);
  const [testAccHistory, setTestAccHistory] = useState<number[]>([]);
  const [trainingStatus, setTrainingStatus] = useState<string>("Idle");
  const [results, setResults] = useState<RunResult[]>([]);

  const modelRef = useRef<LayersModel | null>(null);
  const stopRef = useRef(false);
  // tfjs is lazy-loaded (see ensureTfLoaded) so the ref is populated at runtime, not statically typed here.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const tfRef = useRef<any>(null);

  async function ensureTfLoaded() {
    if (!tfRef.current) {
      tfRef.current = await import("@tensorflow/tfjs");
    }
  }

  const selectedResult = useMemo(
    () => results.find((result) => result.samples === datasetSize) ?? null,
    [datasetSize, results],
  );

  const gapPoints = useMemo(() => {
    return DATASET_SIZES.map((size) => {
      const result = results.find((row) => row.samples === size);
      return {
        x: size,
        y: result ? Math.round((result.accuracyGap || 0) * 100) : null,
      };
    });
  }, [results]);

  const progress = epochs > 0 ? Math.min(100, Math.round((epochCount / epochs) * 100)) : 0;

  async function resetExperiment() {
    stopRef.current = true;
    modelRef.current?.dispose();
    modelRef.current = null;
    setEpochCount(0);
    setTrainLossHistory([]);
    setTestLossHistory([]);
    setTrainAccHistory([]);
    setTestAccHistory([]);
    setTrainingStatus("Idle");
  }

  async function trainModel() {
    await ensureTfLoaded();
    const tf = tfRef.current;

    stopRef.current = false;
    setEpochCount(0);
    setTrainLossHistory([]);
    setTestLossHistory([]);
    setTrainAccHistory([]);
    setTestAccHistory([]);
    setTrainingStatus("Training");

    const dataset = generateConcentricCircles({ samples: datasetSize, radius: RADIUS, noise, seed: Date.now() });
    const { train, test } = splitDataset(dataset);

    const trainData = datasetToTensors(train);
    const testData = datasetToTensors(test);

    const xsTrain = tf.tensor2d(trainData.xs);
    const ysTrain = tf.tensor2d(trainData.ys, [trainData.ys.length, 1]);
    const xsTest = tf.tensor2d(testData.xs);
    const ysTest = tf.tensor2d(testData.ys, [testData.ys.length, 1]);

    modelRef.current?.dispose();
    const model = buildGeneralizationModel(tf);
    model.compile({ optimizer: tf.train.adam(learningRate), loss: "binaryCrossentropy" });
    modelRef.current = model;

    let lastTrainLoss = 0;
    let lastTestLoss = 0;
    let lastTrainAccuracy = 0;
    let lastTestAccuracy = 0;

    try {
      for (let epoch = 0; epoch < epochs; epoch += 1) {
        if (stopRef.current) break;

        const history = await model.fit(xsTrain, ysTrain, {
          epochs: 1,
          batchSize: Math.min(batchSize, trainData.ys.length),
          shuffle: true,
        });

        const trainLoss = Number(history.history.loss?.[0] ?? NaN);
        const testLossTensor = (model.evaluate(xsTest, ysTest, {
          batchSize: Math.min(batchSize, testData.ys.length),
        }) as Tensor);

        const testLossValues = await testLossTensor.data();
        const testLoss = Number(testLossValues[0] ?? NaN);
        testLossTensor.dispose();

        const trainAccuracy = await computeAccuracy(model, xsTrain, trainData.ys);
        const testAccuracy = await computeAccuracy(model, xsTest, testData.ys);

        lastTrainLoss = trainLoss;
        lastTestLoss = testLoss;
        lastTrainAccuracy = trainAccuracy;
        lastTestAccuracy = testAccuracy;

        setEpochCount(epoch + 1);
        setTrainLossHistory((historyArray) => [...historyArray, trainLoss]);
        setTestLossHistory((historyArray) => [...historyArray, testLoss]);
        setTrainAccHistory((historyArray) => [...historyArray, trainAccuracy]);
        setTestAccHistory((historyArray) => [...historyArray, testAccuracy]);
        setTrainingStatus(`Training (epoch ${epoch + 1}/${epochs})`);

        if (tf.nextFrame) await tf.nextFrame();
      }

      if (!stopRef.current) {
        const finalTrainAccuracy = lastTrainAccuracy;
        const finalTestAccuracy = lastTestAccuracy;
        const finalTrainLoss = lastTrainLoss;
        const finalTestLoss = lastTestLoss;

        const result: RunResult = {
          samples: datasetSize,
          trainLoss: finalTrainLoss,
          testLoss: finalTestLoss,
          trainAccuracy: finalTrainAccuracy,
          testAccuracy: finalTestAccuracy,
          accuracyGap: finalTrainAccuracy - finalTestAccuracy,
        };

        setResults((existing) => {
          const updated = existing.filter((item) => item.samples !== datasetSize);
          return [...updated, result];
        });
      }
    } finally {
      if (!stopRef.current) {
        setTrainingStatus("Complete");
      } else {
        setTrainingStatus("Stopped");
      }

      xsTrain.dispose();
      ysTrain.dispose();
      xsTest.dispose();
      ysTest.dispose();
    }
  }

  function formatPercent(value: number) {
    return `${Math.round(value * 100)}%`;
  }

  const currentObservation = selectedResult
    ? {
        claim: "A high-capacity network can memorize very small datasets but generalizes better with more data.",
        observation: `Dataset Size ${selectedResult.samples} – Train Accuracy ${formatPercent(
          selectedResult.trainAccuracy,
        )}, Test Accuracy ${formatPercent(selectedResult.testAccuracy)}, Gap ${formatPercent(
          selectedResult.accuracyGap,
        )}.`,
        conclusion:
          "More data reduces overfitting and improves generalization, shrinking the gap between train and test performance.",
      }
    : {
        claim: "A high-capacity network can memorize very small datasets but fail to generalize.",
        observation: "Train a dataset size to see the gap between training and testing performance.",
        conclusion: "Larger datasets help the same model generalize better.",
      };

  return (
    <div className="p-5 text-white" style={{ fontFamily: "Inter, Arial" }}>
      <div className="grid gap-8 xl:grid-cols-[1.7fr_1fr]">
        <div className="space-y-5">
          <div className="inline-flex items-center gap-3 rounded-full border border-cyan-400/20 bg-cyan-400/10 px-4 py-2 text-sm font-semibold uppercase tracking-[0.28em] text-cyan-300 shadow-sm shadow-cyan-400/10">
            <span>Experiment 4</span>
          </div>
          <p className="text-sm uppercase tracking-[0.28em] text-slate-400">Memorization vs Generalization</p>
          <h1 className="max-w-2xl text-4xl font-semibold leading-tight text-white sm:text-5xl">A high-capacity network memorizes small datasets but generalizes better with more data.</h1>
          <p className="max-w-2xl text-sm leading-7 text-slate-300 sm:text-base">
            Train the same over-parameterized model on different dataset sizes and watch how the testing gap shrinks as more examples are provided.
          </p>
        </div>

        <div className="grid gap-4 sm:grid-cols-3">
          <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
            <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Dataset</p>
            <p className="mt-3 text-lg font-semibold text-white">Noisy circles</p>
          </div>

          <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
            <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Model</p>
            <p className="mt-3 text-lg font-semibold text-white">4-layer overparameterized classifier</p>
          </div>

          <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
            <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Split</p>
            <p className="mt-3 text-lg font-semibold text-white">80% train / 20% test</p>
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
            <div className="h-full rounded-full bg-gradient-to-r from-cyan-400 via-sky-500 to-cyan-500 transition-all duration-700 ease-out" style={{ width: `${progress}%` }} />
          </div>
        </div>
      </div>

      <div className="mt-10 flex flex-col gap-16 xl:flex-row">
        <div className="flex flex-col gap-6 xl:w-96">
          <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
            <div className="mb-4 flex items-center gap-3 text-sm font-semibold uppercase tracking-[0.15em] text-slate-300">
              <Database className="h-4 w-4 text-cyan-300" />
              <span>Dataset Size</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {DATASET_SIZES.map((size) => (
                <button
                  key={size}
                  type="button"
                  onClick={() => setDatasetSize(size)}
                  className={`rounded-2xl px-4 py-2 text-sm font-semibold transition ${datasetSize === size ? "bg-cyan-500 text-slate-900" : "bg-slate-800 text-slate-200"}`}
                >
                  {size}
                </button>
              ))}
            </div>
          </div>

          <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
            <div className="mb-4 flex items-center gap-3 text-sm font-semibold uppercase tracking-[0.15em] text-slate-300">
              <SlidersHorizontal className="h-4 w-4 text-cyan-300" />
              <span>Controls</span>
            </div>
            <div className="space-y-4">
              <div className="space-y-2">
                <label htmlFor="noise" className="text-sm font-medium text-slate-100">Noise</label>
                <input
                  id="noise"
                  type="range"
                  min={0}
                  max={0.5}
                  step={0.01}
                  value={noise}
                  onChange={(event) => setNoise(Number(event.target.value))}
                  className="w-full accent-cyan-400"
                />
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span>Low</span>
                  <span>{noise.toFixed(2)}</span>
                  <span>High</span>
                </div>
              </div>

              <div className="space-y-2">
                <label htmlFor="epochs" className="text-sm font-medium text-slate-100">Epochs</label>
                <input
                  id="epochs"
                  type="number"
                  min={1}
                  max={200}
                  value={epochs}
                  onChange={(event) => setEpochs(Number(event.target.value))}
                  className="w-full rounded-2xl border border-slate-700 bg-slate-900/90 px-4 py-2 text-slate-100 outline-none transition focus:border-cyan-400 focus:ring-2 focus:ring-cyan-500/20"
                />
              </div>

              <div className="space-y-2">
                <label htmlFor="learningRate" className="text-sm font-medium text-slate-100">Learning Rate</label>
                <input
                  id="learningRate"
                  type="number"
                  min={0.0005}
                  max={0.1}
                  step={0.0005}
                  value={learningRate}
                  onChange={(event) => setLearningRate(Number(event.target.value))}
                  className="w-full rounded-2xl border border-slate-700 bg-slate-900/90 px-4 py-2 text-slate-100 outline-none transition focus:border-cyan-400 focus:ring-2 focus:ring-cyan-500/20"
                />
              </div>

              <div className="space-y-2">
                <label htmlFor="batchSize" className="text-sm font-medium text-slate-100">Batch Size</label>
                <input
                  id="batchSize"
                  type="number"
                  min={4}
                  max={256}
                  step={4}
                  value={batchSize}
                  onChange={(event) => setBatchSize(Number(event.target.value))}
                  className="w-full rounded-2xl border border-slate-700 bg-slate-900/90 px-4 py-2 text-slate-100 outline-none transition focus:border-cyan-400 focus:ring-2 focus:ring-cyan-500/20"
                />
              </div>
            </div>
          </div>

          <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
            <div className="mb-4 flex items-center gap-3 text-sm font-semibold uppercase tracking-[0.15em] text-slate-300">
              <Sparkles className="h-4 w-4 text-cyan-300" />
              <span>Actions</span>
            </div>
            <div className="grid gap-3">
              <button
                type="button"
                onClick={trainModel}
                className="inline-flex items-center justify-center gap-2 rounded-2xl bg-cyan-500 px-4 py-3 text-sm font-semibold text-slate-900 shadow-md shadow-cyan-500/20 transition transform-gpu hover:-translate-y-0.5 hover:bg-cyan-400 active:scale-95 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/60 focus-visible:ring-offset-2"
              >
                <Play className="h-4 w-4" />
                Train
              </button>
              <button
                type="button"
                onClick={resetExperiment}
                className="inline-flex items-center justify-center gap-2 rounded-2xl bg-slate-800 px-4 py-3 text-sm font-semibold text-slate-100 transition transform-gpu hover:-translate-y-0.5 hover:bg-slate-700 active:scale-95 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-500/50 focus-visible:ring-offset-2"
              >
                <StopCircle className="h-4 w-4" />
                Reset
              </button>
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-6">
          <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
            <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
              <div className="text-sm uppercase tracking-[0.18em] text-slate-400">Training Examples</div>
              <div className="mt-3 text-3xl font-semibold text-white">{datasetSize}</div>
              <div className="mt-2 text-sm text-slate-500">Number of training and testing examples for this run.</div>
            </div>
            <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
              <div className="text-sm uppercase tracking-[0.18em] text-slate-400">Noise</div>
              <div className="mt-3 text-3xl font-semibold text-white">{noise.toFixed(2)}</div>
              <div className="mt-2 text-sm text-slate-500">Gaussian noise added to the circle patterns.</div>
            </div>
          </div>

          <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
            <div className="mb-4 flex items-center gap-3 text-sm font-semibold uppercase tracking-[0.15em] text-slate-300">
              <Layers className="h-4 w-4 text-cyan-300" />
              <span>Current Run</span>
            </div>
            <div className="grid gap-3 text-sm text-slate-400">
              <div className="flex items-center justify-between gap-2">
                <span>Last train loss</span>
                <span>{trainLossHistory.length ? trainLossHistory[trainLossHistory.length - 1].toFixed(3) : "—"}</span>
              </div>
              <div className="flex items-center justify-between gap-2">
                <span>Last test loss</span>
                <span>{testLossHistory.length ? testLossHistory[testLossHistory.length - 1].toFixed(3) : "—"}</span>
              </div>
              <div className="flex items-center justify-between gap-2">
                <span>Last train accuracy</span>
                <span>{trainAccHistory.length ? formatPercent(trainAccHistory[trainAccHistory.length - 1]) : "—"}</span>
              </div>
              <div className="flex items-center justify-between gap-2">
                <span>Last test accuracy</span>
                <span>{testAccHistory.length ? formatPercent(testAccHistory[testAccHistory.length - 1]) : "—"}</span>
              </div>
            </div>
          </div>

          <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
            <div className="mb-4 flex items-center gap-3 text-sm font-semibold uppercase tracking-[0.15em] text-slate-300">
              <Sparkles className="h-4 w-4 text-cyan-300" />
              <span>Educational Panel</span>
            </div>
            <div className="space-y-3 text-sm text-slate-300">
              <div>
                <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Claim</p>
                <p className="mt-2 text-white">{currentObservation.claim}</p>
              </div>
              <div>
                <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Observation</p>
                <p className="mt-2 text-white">{currentObservation.observation}</p>
              </div>
              <div>
                <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Conclusion</p>
                <p className="mt-2 text-white">{currentObservation.conclusion}</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-10 grid gap-6 xl:grid-cols-[1fr_1fr]">
        <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
          <div className="font-semibold text-white mb-3">Training Loss</div>
          <Plot
            data={[
              {
                x: trainLossHistory.map((_, index) => index + 1),
                y: trainLossHistory,
                type: "scatter",
                mode: "lines+markers",
                marker: { color: "#38bdf8" },
                line: { shape: "spline", color: "#38bdf8" },
                name: "Train Loss",
              },
            ]}
            layout={{
              autosize: true,
              margin: { t: 20, b: 30, l: 40, r: 20 },
              plot_bgcolor: "#041026",
              paper_bgcolor: "#041026",
              font: { color: "#cbd5e1" },
              xaxis: { title: "Epoch", color: "#94a3b8" },
              yaxis: { title: "Loss", color: "#94a3b8" },
              showlegend: false,
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%", height: 320 }}
          />
        </div>

        <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
          <div className="font-semibold text-white mb-3">Testing Loss</div>
          <Plot
            data={[
              {
                x: testLossHistory.map((_, index) => index + 1),
                y: testLossHistory,
                type: "scatter",
                mode: "lines+markers",
                marker: { color: "#f472b6" },
                line: { shape: "spline", color: "#f472b6" },
                name: "Test Loss",
              },
            ]}
            layout={{
              autosize: true,
              margin: { t: 20, b: 30, l: 40, r: 20 },
              plot_bgcolor: "#041026",
              paper_bgcolor: "#041026",
              font: { color: "#cbd5e1" },
              xaxis: { title: "Epoch", color: "#94a3b8" },
              yaxis: { title: "Loss", color: "#94a3b8" },
              showlegend: false,
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%", height: 320 }}
          />
        </div>

        <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
          <div className="font-semibold text-white mb-3">Training Accuracy</div>
          <Plot
            data={[
              {
                x: trainAccHistory.map((_, index) => index + 1),
                y: trainAccHistory.map((value) => value * 100),
                type: "scatter",
                mode: "lines+markers",
                marker: { color: "#34d399" },
                line: { shape: "spline", color: "#34d399" },
                name: "Train Accuracy",
              },
            ]}
            layout={{
              autosize: true,
              margin: { t: 20, b: 30, l: 40, r: 20 },
              plot_bgcolor: "#041026",
              paper_bgcolor: "#041026",
              font: { color: "#cbd5e1" },
              xaxis: { title: "Epoch", color: "#94a3b8" },
              yaxis: { title: "Accuracy (%)", color: "#94a3b8" },
              showlegend: false,
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%", height: 320 }}
          />
        </div>

        <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
          <div className="font-semibold text-white mb-3">Testing Accuracy</div>
          <Plot
            data={[
              {
                x: testAccHistory.map((_, index) => index + 1),
                y: testAccHistory.map((value) => value * 100),
                type: "scatter",
                mode: "lines+markers",
                marker: { color: "#facc15" },
                line: { shape: "spline", color: "#facc15" },
                name: "Test Accuracy",
              },
            ]}
            layout={{
              autosize: true,
              margin: { t: 20, b: 30, l: 40, r: 20 },
              plot_bgcolor: "#041026",
              paper_bgcolor: "#041026",
              font: { color: "#cbd5e1" },
              xaxis: { title: "Epoch", color: "#94a3b8" },
              yaxis: { title: "Accuracy (%)", color: "#94a3b8" },
              showlegend: false,
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%", height: 320 }}
          />
        </div>
      </div>

      <div className="mt-6 rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
        <div className="font-semibold text-white mb-3">Generalization Gap</div>
        <Plot
          data={[
            {
              x: gapPoints.map((point) => point.x),
              y: gapPoints.map((point) => (point.y === null ? NaN : point.y)),
              type: "scatter",
              mode: "lines+markers",
              marker: { color: "#f97316" },
              line: { shape: "linear", color: "#f97316" },
              name: "Accuracy Gap",
              connectgaps: true,
            },
          ]}
          layout={{
            autosize: true,
            margin: { t: 20, b: 40, l: 50, r: 20 },
            plot_bgcolor: "#041026",
            paper_bgcolor: "#041026",
            font: { color: "#cbd5e1" },
            xaxis: { title: "Dataset Size", type: "category", color: "#94a3b8" },
            yaxis: { title: "Train - Test Accuracy (%)", color: "#94a3b8" },
            showlegend: false,
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: "100%", height: 360 }}
        />
      </div>
    </div>
  );
}
