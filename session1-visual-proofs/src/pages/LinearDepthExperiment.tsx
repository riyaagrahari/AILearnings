import type React from "react";
import { useEffect, useRef, useState } from "react";
import { Layers, Play, Pause, StopCircle, RefreshCcw, Shuffle, Sparkles, Database, SlidersHorizontal } from "lucide-react";
import type { Dataset, DatasetGeneratorOptions } from "../experiments/relu/dataset";
import { generateConcentricCircles, datasetToTensors } from "../experiments/relu/dataset";
import type { Tensor2D, Tensor, LayersModel } from "@tensorflow/tfjs";
import Plot from "react-plotly.js";

type TrainingState = "idle" | "running" | "paused";

const CANVAS_SIZE = 360;
const GRID_RES = 180;

type TfNamespace = typeof import("@tensorflow/tfjs");

function buildDeepLinearModel(tff: TfNamespace, depth: number, units: number): LayersModel {
  const model = tff.sequential();

  // stack `depth` linear layers (no activations), followed by sigmoid output
  for (let i = 0; i < depth; i++) {
    model.add(tff.layers.dense({ units: units, inputShape: i === 0 ? [2] : undefined, activation: undefined }));
  }

  model.add(tff.layers.dense({ units: 1, activation: "sigmoid" }));

  return model as LayersModel;
}

function buildSingleLinearModel(tff: TfNamespace): LayersModel {
  const model = tff.sequential();
  model.add(tff.layers.dense({ units: 1, inputShape: [2], activation: "sigmoid" }));
  return model as LayersModel;
}

function buildDeepReLUModel(tff: TfNamespace, depth: number, units: number): LayersModel {
  const model = tff.sequential();

  for (let i = 0; i < depth; i++) {
    model.add(tff.layers.dense({ units: units, inputShape: i === 0 ? [2] : undefined, activation: "relu" }));
  }

  model.add(tff.layers.dense({ units: 1, activation: "sigmoid" }));

  return model as LayersModel;
}

function drawAxes(ctx: CanvasRenderingContext2D, size: number) {
  ctx.save();
  ctx.strokeStyle = "rgba(200,200,200,0.25)";
  ctx.lineWidth = 1;

  for (let i = 0; i <= 10; i++) {
    const t = (i / 10) * size;
    ctx.beginPath();
    ctx.moveTo(t, 0);
    ctx.lineTo(t, size);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, t);
    ctx.lineTo(size, t);
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(200,200,200,0.6)";
  ctx.beginPath();
  ctx.moveTo(size / 2, 0);
  ctx.lineTo(size / 2, size);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(0, size / 2);
  ctx.lineTo(size, size / 2);
  ctx.stroke();

  ctx.restore();
}

function renderHeatmap(
  ctx: CanvasRenderingContext2D,
  size: number,
  gridResolution: number,
  probabilities: Float32Array,
) {
  const img = ctx.createImageData(gridResolution, gridResolution);

  for (let y = 0; y < gridResolution; y++) {
    for (let x = 0; x < gridResolution; x++) {
      const i = y * gridResolution + x;
      const p = probabilities[i];

      const blue = { r: 102, g: 179, b: 255 };
      const orange = { r: 255, g: 138, b: 80 };
      const r = Math.round(blue.r * (1 - p) + orange.r * p);
      const g = Math.round(blue.g * (1 - p) + orange.g * p);
      const b = Math.round(blue.b * (1 - p) + orange.b * p);

      const idx = (y * gridResolution + x) * 4;
      img.data[idx] = r;
      img.data[idx + 1] = g;
      img.data[idx + 2] = b;
      img.data[idx + 3] = 180;
    }
  }

  const tmp = document.createElement("canvas");
  tmp.width = gridResolution;
  tmp.height = gridResolution;
  const tctx = tmp.getContext("2d")!;
  tctx.putImageData(img, 0, 0);
  tctx.imageSmoothingEnabled = true;
  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.globalAlpha = 0.94;
  ctx.drawImage(tmp, 0, 0, size, size);
  ctx.restore();
}

function drawPoints(ctx: CanvasRenderingContext2D, size: number, points: Dataset["points"], radius: number) {
  ctx.save();

  const scale = size / (radius * 4);
  const center = size / 2;

  ctx.lineWidth = 1.5;
  ctx.shadowColor = "rgba(0, 0, 0, 0.25)";
  ctx.shadowBlur = 4;
  ctx.shadowOffsetX = 0;
  ctx.shadowOffsetY = 2;

  for (const p of points) {
    const cx = center + p.x * scale;
    const cy = center - p.y * scale;
    const fill = p.label === 1 ? "#ff8a50" : "#66b3ff";
    const stroke = p.label === 1 ? "rgba(255,138,80,0.9)" : "rgba(102,179,255,0.9)";

    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = stroke;
    ctx.stroke();
  }

  ctx.restore();
}

export default function LinearDepthExperiment(): React.ReactElement {
  const [dataset, setDataset] = useState<Dataset>(() =>
    generateConcentricCircles({ samples: 200, radius: 1.0, noise: 0.08, seed: 1 }),
  );

  const [options, setOptions] = useState<DatasetGeneratorOptions>({
    samples: 200,
    radius: 1.0,
    noise: 0.08,
    seed: 1,
  });

  const [learningRate] = useState<number>(0.01);
  const [epochs, setEpochs] = useState<number>(60);
  const [batchSize, setBatchSize] = useState<number>(32);
  const depth = 5;
  const [hiddenUnits, setHiddenUnits] = useState<number>(16);
  const [animationSpeed, setAnimationSpeed] = useState<number>(3);

  const [state, setState] = useState<TrainingState>("idle");

  const [epochCount, setEpochCount] = useState<number>(0);
  const [lossHistoryA, setLossHistoryA] = useState<number[]>([]);
  const [accHistoryA, setAccHistoryA] = useState<number[]>([]);
  const [lossHistoryB, setLossHistoryB] = useState<number[]>([]);
  const [accHistoryB, setAccHistoryB] = useState<number[]>([]);
  const [lossHistoryC, setLossHistoryC] = useState<number[]>([]);
  const [accHistoryC, setAccHistoryC] = useState<number[]>([]);
  const [trainingComplete, setTrainingComplete] = useState(false);
  const [effectiveMatrix, setEffectiveMatrix] = useState<{ shape: number[]; values: number[][] } | null>(null);

  const percentComplete = epochs > 0 ? Math.min(100, Math.round((epochCount / epochs) * 100)) : 0;
  const trainingStatusLabel = trainingComplete
    ? "✓ Training Complete"
    : state === "running"
    ? "Training"
    : state === "paused"
    ? "Paused"
    : "Idle";

  const canvasARef = useRef<HTMLCanvasElement | null>(null);
  const canvasBRef = useRef<HTMLCanvasElement | null>(null);
  const canvasCRef = useRef<HTMLCanvasElement | null>(null);

  const modelARef = useRef<LayersModel | null>(null);
  const modelBRef = useRef<LayersModel | null>(null);
  const modelCRef = useRef<LayersModel | null>(null);

  const tensorsRef = useRef<{ xs?: Tensor2D; ys?: Tensor } | null>(null);
  const stopRef = useRef(false);
  const pausedRef = useRef(false);

  // tfjs is lazy-loaded (see ensureTfLoaded) so the ref is populated at runtime, not statically typed here.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const tfRef = useRef<any>(null);

  async function ensureTfLoaded() {
    if (!tfRef.current) {
      tfRef.current = await import("@tensorflow/tfjs");
    }
  }

  useEffect(() => {
    let mounted = true;

    (async () => {
      await ensureTfLoaded();

      if (!mounted) return;

      const { xs, ys } = datasetToTensors(dataset);

      tensorsRef.current?.xs?.dispose();
      tensorsRef.current?.ys?.dispose();

      tensorsRef.current = {
        xs: tfRef.current.tensor2d(xs),
        ys: tfRef.current.tensor2d(ys, [ys.length, 1]),
      };
    })();

    return () => {
      mounted = false;
      tensorsRef.current?.xs?.dispose();
      tensorsRef.current?.ys?.dispose();
      tensorsRef.current = null;
    };
  }, [dataset]);

  useEffect(() => {
    return () => {
      modelARef.current?.dispose();
      modelBRef.current?.dispose();
      modelCRef.current?.dispose();
      tensorsRef.current?.xs?.dispose?.();
      tensorsRef.current?.ys?.dispose?.();
    };
  }, []);

  async function resetModels() {
    await ensureTfLoaded();

    tfRef.current.engine().startScope();
    modelARef.current?.dispose();
    modelBRef.current?.dispose();
    modelCRef.current?.dispose();

    // Model A: single linear layer -> Dense(1, sigmoid)
    const a = buildSingleLinearModel(tfRef.current);

    // Model B: stacked linear layers (no activation between), then final sigmoid
    const b = buildDeepLinearModel(tfRef.current, depth, hiddenUnits);

    // Model C: stacked layers with ReLU between, then final sigmoid
    const c = buildDeepReLUModel(tfRef.current, depth, hiddenUnits);

    const optA = tfRef.current.train.adam(learningRate);
    const optB = tfRef.current.train.adam(learningRate);
    const optC = tfRef.current.train.adam(learningRate);

    a.compile({ optimizer: optA, loss: "binaryCrossentropy" });
    b.compile({ optimizer: optB, loss: "binaryCrossentropy" });
    c.compile({ optimizer: optC, loss: "binaryCrossentropy" });

    modelARef.current = a;
    modelBRef.current = b;
    modelCRef.current = c;
    tfRef.current.engine().endScope();
  }

  async function computeAccuracy(model: LayersModel, xs: Tensor2D, ys: Tensor): Promise<number> {
    if (!tfRef.current) await ensureTfLoaded();

    const predictions = model.predict(xs) as Tensor;
    const threshold = tfRef.current.scalar(0.5);
    const binary = tfRef.current.greater(predictions, threshold).toInt();
    const labels = ys.toInt();
    const correct = tfRef.current.equal(binary, labels).toFloat();
    const accuracyTensor = correct.mean();

    const accuracy = (await accuracyTensor.data())[0];

    predictions.dispose();
    threshold.dispose();
    binary.dispose();
    labels.dispose();
    correct.dispose();
    accuracyTensor.dispose();

    return accuracy;
  }

  async function handleGenerate() {
    const d = generateConcentricCircles(options);
    setDataset(d);
    setEpochCount(0);
    setLossHistoryA([]);
    setAccHistoryA([]);
    setLossHistoryB([]);
    setAccHistoryB([]);
    setLossHistoryC([]);
    setAccHistoryC([]);
    setTrainingComplete(false);
    pausedRef.current = false;
    await resetModels();
  }

  async function trainLoop() {
    if (!tensorsRef.current || !modelARef.current || !modelBRef.current || !modelCRef.current) return;

    setState("running");
    setTrainingComplete(false);
    stopRef.current = false;

    const xs = tensorsRef.current!.xs!;
    const ys = tensorsRef.current!.ys!;

    for (let e = 0; e < epochs; e++) {
      if (stopRef.current) break;

      const [resA, resB, resC] = await Promise.all([
        modelARef.current!.fit(xs, ys, { epochs: 1, batchSize, shuffle: true }),
        modelBRef.current!.fit(xs, ys, { epochs: 1, batchSize, shuffle: true }),
        modelCRef.current!.fit(xs, ys, { epochs: 1, batchSize, shuffle: true }),
      ]);

      const lossA = resA.history.loss ? (resA.history.loss as number[]).slice(-1)[0] : NaN;
      const lossB = resB.history.loss ? (resB.history.loss as number[]).slice(-1)[0] : NaN;
      const lossC = resC.history.loss ? (resC.history.loss as number[]).slice(-1)[0] : NaN;

      const accA = await computeAccuracy(modelARef.current!, xs, ys);
      const accB = await computeAccuracy(modelBRef.current!, xs, ys);
      const accC = await computeAccuracy(modelCRef.current!, xs, ys);

      setEpochCount((p) => p + 1);
      setLossHistoryA((h) => [...h, Number(lossA ?? NaN)]);
      setAccHistoryA((h) => [...h, Number(accA ?? NaN)]);
      setLossHistoryB((h) => [...h, Number(lossB ?? NaN)]);
      setAccHistoryB((h) => [...h, Number(accB ?? NaN)]);
      setLossHistoryC((h) => [...h, Number(lossC ?? NaN)]);
      setAccHistoryC((h) => [...h, Number(accC ?? NaN)]);

      if ((e + 1) % 2 === 0) {
        await updateDecisionBoundaries();
      }

      if (tfRef.current && tfRef.current.nextFrame) await tfRef.current.nextFrame();

      while (pausedRef.current) {
        await new Promise((r) => setTimeout(r, 100));
        if (stopRef.current) break;
      }
    }

    setTrainingComplete(!stopRef.current);
    setState("idle");
  }

  async function updateDecisionBoundaries() {
    if (!modelARef.current || !modelBRef.current || !modelCRef.current) return;

    const gridRes = GRID_RES;
    const total = gridRes * gridRes;

    const coords: number[] = new Array(total * 2);

    const radius = options.radius;

    let idx = 0;
    for (let y = 0; y < gridRes; y++) {
      for (let x = 0; x < gridRes; x++) {
        const nx = (x / (gridRes - 1)) * 2 - 1; // -1..1
        const ny = (y / (gridRes - 1)) * 2 - 1;

        coords[idx++] = nx * radius * 2;
        coords[idx++] = -ny * radius * 2;
      }
    }

    await ensureTfLoaded();

    const tcoords = tfRef.current.tensor2d(coords, [total, 2]);

    const pa = modelARef.current!.predict(tcoords) as Tensor;
    const pb = modelBRef.current!.predict(tcoords) as Tensor;
    const pc = modelCRef.current!.predict(tcoords) as Tensor;

    const [arrA, arrB, arrC] = await Promise.all([
      pa.data() as Promise<Float32Array>,
      pb.data() as Promise<Float32Array>,
      pc.data() as Promise<Float32Array>,
    ]);

    const ca = canvasARef.current!;
    const cb = canvasBRef.current!;
    const cc = canvasCRef.current!;

    const ctxA = ca.getContext("2d")!;
    const ctxB = cb.getContext("2d")!;
    const ctxC = cc.getContext("2d")!;

    const tmpA = document.createElement("canvas");
    tmpA.width = gridRes;
    tmpA.height = gridRes;
    const tA = tmpA.getContext("2d")!;

    const tmpB = document.createElement("canvas");
    tmpB.width = gridRes;
    tmpB.height = gridRes;
    const tB = tmpB.getContext("2d")!;

    const tmpC = document.createElement("canvas");
    tmpC.width = gridRes;
    tmpC.height = gridRes;
    const tC = tmpC.getContext("2d")!;

    renderHeatmap(tA, gridRes, gridRes, arrA);
    renderHeatmap(tB, gridRes, gridRes, arrB);
    renderHeatmap(tC, gridRes, gridRes, arrC);

    const skipTicks = Math.max(1, 11 - animationSpeed);
    async function animateBlend(ctx: CanvasRenderingContext2D, src: HTMLCanvasElement) {
      const frames = 8;
      for (let i = 0; i <= frames; i++) {
        const alpha = i / frames;
        ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(src, 0, 0, CANVAS_SIZE, CANVAS_SIZE);
        ctx.restore();
        for (let tick = 0; tick < skipTicks; tick++) {
          await new Promise((r) => requestAnimationFrame(r));
        }
      }
    }

    await Promise.all([animateBlend(ctxA, tmpA), animateBlend(ctxB, tmpB), animateBlend(ctxC, tmpC)]);

    drawAxes(ctxA, CANVAS_SIZE);
    drawAxes(ctxB, CANVAS_SIZE);
    drawAxes(ctxC, CANVAS_SIZE);

    drawPoints(ctxA, CANVAS_SIZE, dataset.points, options.radius);
    drawPoints(ctxB, CANVAS_SIZE, dataset.points, options.radius);
    drawPoints(ctxC, CANVAS_SIZE, dataset.points, options.radius);

    pa.dispose();
    pb.dispose();
    pc.dispose();
    tcoords.dispose();
  }

  function handlePause() {
    pausedRef.current = true;
    if (state === "running") setState("paused");
  }

  function handleResume() {
    pausedRef.current = false;
    if (state === "paused") setState("running");
  }

  function handleStop() {
    stopRef.current = true;
    pausedRef.current = false;
    setTrainingComplete(false);
    setState("idle");
  }

  async function handleReset() {
    handleStop();
    setEpochCount(0);
    setLossHistoryA([]);
    setAccHistoryA([]);
    setLossHistoryB([]);
    setAccHistoryB([]);
    setLossHistoryC([]);
    setAccHistoryC([]);
    setTrainingComplete(false);
    await resetModels();
    await updateDecisionBoundaries();
  }

  useEffect(() => {
    resetModels();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [depth, hiddenUnits, learningRate]);

  useEffect(() => {
    const ca = canvasARef.current!;
    const cb = canvasBRef.current!;
    const cc = canvasCRef.current!;
    if (!ca || !cb || !cc) return;

    const ctxA = ca.getContext("2d")!;
    const ctxB = cb.getContext("2d")!;
    const ctxC = cc.getContext("2d")!;

    ctxA.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    ctxB.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    ctxC.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    drawAxes(ctxA, CANVAS_SIZE);
    drawAxes(ctxB, CANVAS_SIZE);
    drawAxes(ctxC, CANVAS_SIZE);

    drawPoints(ctxA, CANVAS_SIZE, dataset.points, options.radius);
    drawPoints(ctxB, CANVAS_SIZE, dataset.points, options.radius);
    drawPoints(ctxC, CANVAS_SIZE, dataset.points, options.radius);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [canvasARef.current, canvasBRef.current, canvasCRef.current, dataset]);

  useEffect(() => {
    // Derive effectiveMatrix from trainingComplete: clear it immediately when training
    // isn't complete, then compute it asynchronously once it is.
    if (!trainingComplete) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setEffectiveMatrix(null);
      return;
    }

    (async () => {
      if (!modelBRef.current) return;
      await ensureTfLoaded();
      const tff = tfRef.current;

      try {
        const combined = tff.tidy(() => {
          // gather kernel weights from modelB's layers (keep order)
          const kernels: Tensor[] = [];
          for (const layer of modelBRef.current!.layers) {
            const w = layer.getWeights();
            if (w && w.length > 0) {
              kernels.push(w[0]);
            }
          }

          if (kernels.length === 0) return tff.tensor([]);

          let m = kernels[0];
          for (let i = 1; i < kernels.length; i++) {
            m = m.matMul(kernels[i]);
          }

          return m.clone();
        });

        const shape = combined.shape.slice();
        const values = (await combined.array()) as number[][];
        combined.dispose();

        setEffectiveMatrix({ shape, values });
      } catch {
        // ignore: tensors may already be disposed if a reset happened mid-computation
      }
    })();
  }, [trainingComplete]);

  return (
    <div className="p-5 text-white" style={{ fontFamily: "Inter, Arial" }}>
      <div className="grid gap-8 xl:grid-cols-[1.7fr_1fr]">
        <div className="space-y-5">
          <div className="inline-flex items-center gap-3 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-4 py-2 text-sm font-semibold uppercase tracking-[0.28em] text-emerald-300 shadow-sm shadow-emerald-400/10">
            <span>Experiment 2</span>
          </div>
          <p className="text-sm uppercase tracking-[0.28em] text-slate-400">Depth without Nonlinearity is a Lie</p>
          <h1 className="max-w-2xl text-4xl font-semibold leading-tight text-white sm:text-5xl">Depth without Nonlinearity is a Lie</h1>
          <p className="max-w-2xl text-sm leading-7 text-slate-300 sm:text-base">
            Stacking multiple linear layers without nonlinear activations does not increase representational power. A deep linear network is equivalent to a single linear transform — add ReLU between layers and the network can learn nonlinear decision boundaries.
          </p>
        </div>

        <div className="grid gap-4 sm:grid-cols-3">
          <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
            <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Dataset</p>
            <p className="mt-3 text-lg font-semibold text-white">Concentric Circles</p>
          </div>

          <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
            <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Models</p>
            <p className="mt-3 text-lg font-semibold text-white">Single Linear vs Five Linear vs Five+ReLU</p>
          </div>

          <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
            <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Framework</p>
            <p className="mt-3 text-lg font-semibold text-white">TensorFlow.js</p>
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
            {trainingStatusLabel}
          </div>
        </div>

        <div className="mt-6">
          <div className="mb-3 flex items-center justify-between gap-2 text-sm font-semibold uppercase tracking-[0.18em] text-slate-400">
            <span>Progress</span>
            <span>{percentComplete}%</span>
          </div>
          <div className="h-3 overflow-hidden rounded-full bg-slate-900/90 ring-1 ring-slate-700">
            <div className="h-full rounded-full bg-gradient-to-r from-emerald-400 via-green-500 to-emerald-500 transition-all duration-700 ease-out" style={{ width: `${percentComplete}%` }} />
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
                <label htmlFor="samples" className="text-sm font-medium text-slate-100">Samples</label>
                <input id="samples" type="number" min={50} step={10} value={options.samples} onChange={(e) => setOptions({ ...options, samples: Number(e.target.value) })} className="w-full rounded-2xl border border-slate-700 bg-slate-900/90 px-4 py-2 text-slate-100 outline-none transition focus:border-emerald-400 focus:ring-2 focus:ring-emerald-500/20" />
                <p className="text-xs text-slate-500">Number of points in the circle dataset.</p>
              </div>

              <div className="space-y-2">
                <label htmlFor="noise" className="text-sm font-medium text-slate-100">Noise</label>
                <div className="flex items-center gap-3">
                  <input id="noise" type="range" min={0} max={0.3} step={0.01} value={options.noise} onChange={(e) => setOptions({ ...options, noise: Number(e.target.value) })} className="w-full accent-emerald-400" />
                  <span className="text-sm text-slate-400">{options.noise.toFixed(2)}</span>
                </div>
                <p className="text-xs text-slate-500">Adds radial noise to the points.</p>
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <label htmlFor="radius" className="text-sm font-medium text-slate-100">Radius</label>
                  <input id="radius" type="number" step={0.1} value={options.radius} onChange={(e) => setOptions({ ...options, radius: Number(e.target.value) })} className="w-full rounded-2xl border border-slate-700 bg-slate-900/90 px-4 py-2 text-slate-100 outline-none transition focus:border-emerald-400 focus:ring-2 focus:ring-emerald-500/20" />
                  <p className="text-xs text-slate-500">Circle radius for class separation.</p>
                </div>

                <div className="space-y-2">
                  <label htmlFor="seed" className="text-sm font-medium text-slate-100">Seed</label>
                  <input id="seed" type="number" value={options.seed} onChange={(e) => setOptions({ ...options, seed: Number(e.target.value) })} className="w-full rounded-2xl border border-slate-700 bg-slate-900/90 px-4 py-2 text-slate-100 outline-none transition focus:border-emerald-400 focus:ring-2 focus:ring-emerald-500/20" />
                  <p className="text-xs text-slate-500">Control reproducible dataset generation.</p>
                </div>
              </div>
            </div>
          </div>

          <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
            <div className="mb-4 flex items-center gap-3 text-sm font-semibold uppercase tracking-[0.15em] text-slate-300">
              <Layers className="h-4 w-4 text-amber-300" />
              <span>Model</span>
            </div>
            <div className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-slate-100">Depth (layers)</label>
                <div className="text-sm text-slate-400">5 (fixed)</div>
                <p className="text-xs text-slate-500">Fixed to five stacked layers for the assignment.</p>
              </div>

              <div className="space-y-2">
                <label htmlFor="hiddenUnits" className="text-sm font-medium text-slate-100">Hidden Units</label>
                <div className="flex items-center gap-3">
                  <input id="hiddenUnits" type="range" min={4} max={64} step={1} value={hiddenUnits} onChange={(e) => setHiddenUnits(Number(e.target.value))} className="w-full accent-amber-400" />
                  <span className="text-sm text-slate-400">{hiddenUnits}</span>
                </div>
                <p className="text-xs text-slate-500">Width of each hidden layer.</p>
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <label htmlFor="batchSize" className="text-sm font-medium text-slate-100">Batch Size</label>
                  <input id="batchSize" type="number" value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value))} className="w-full rounded-2xl border border-slate-700 bg-slate-900/90 px-4 py-2 text-slate-100 outline-none transition focus:border-amber-400 focus:ring-2 focus:ring-amber-500/20" />
                  <p className="text-xs text-slate-500">Examples processed per training step.</p>
                </div>

                <div className="space-y-2">
                  <label htmlFor="epochs" className="text-sm font-medium text-slate-100">Epochs</label>
                  <input id="epochs" type="number" value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} className="w-full rounded-2xl border border-slate-700 bg-slate-900/90 px-4 py-2 text-slate-100 outline-none transition focus:border-amber-400 focus:ring-2 focus:ring-amber-500/20" />
                  <p className="text-xs text-slate-500">Total passes over the dataset.</p>
                </div>
              </div>
            </div>
          </div>

          <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
            <div className="mb-4 flex items-center gap-3 text-sm font-semibold uppercase tracking-[0.15em] text-slate-300">
              <SlidersHorizontal className="h-4 w-4 text-violet-300" />
              <span>Animation</span>
            </div>
            <div className="space-y-3">
              <label htmlFor="animationSpeed" className="text-sm font-medium text-slate-100">Animation Speed</label>
              <div className="flex items-center gap-3">
                <input id="animationSpeed" type="range" min={1} max={10} step={1} value={animationSpeed} onChange={(e) => setAnimationSpeed(Number(e.target.value))} className="w-full accent-violet-400" />
                <span className="text-sm text-slate-400">{animationSpeed}</span>
              </div>
              <p className="text-xs text-slate-500">Controls how quickly the heatmap updates.</p>
            </div>
          </div>

          <div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
            <div className="mb-4 flex items-center gap-3 text-sm font-semibold uppercase tracking-[0.15em] text-slate-300">
              <Sparkles className="h-4 w-4 text-emerald-300" />
              <span>Actions</span>
            </div>
            <div className="grid gap-3">
              <button type="button" onClick={handleGenerate} className="inline-flex items-center justify-center gap-2 rounded-2xl bg-emerald-500 px-4 py-3 text-sm font-semibold text-slate-900 shadow-md shadow-emerald-500/20 transition transform-gpu hover:-translate-y-0.5 hover:bg-emerald-400 active:scale-95 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400/60 focus-visible:ring-offset-2" aria-label="Generate dataset">
                <RefreshCcw className="h-4 w-4" />
                Generate Dataset
              </button>

              <button type="button" onClick={() => { setOptions({ ...options, seed: options.seed + 1 }); setDataset(generateConcentricCircles({ ...options, seed: options.seed + 1 })); }} className="inline-flex items-center justify-center gap-2 rounded-2xl border border-slate-700 bg-slate-900/90 px-4 py-3 text-sm font-semibold text-slate-100 transition transform-gpu hover:-translate-y-0.5 hover:border-slate-500 hover:bg-slate-800 active:scale-95 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-500/50 focus-visible:ring-offset-2" aria-label="Shuffle dataset">
                <Shuffle className="h-4 w-4" />
                Shuffle
              </button>

              <button type="button" onClick={handleReset} className="inline-flex items-center justify-center gap-2 rounded-2xl border border-slate-700 bg-slate-900/90 px-4 py-3 text-sm font-semibold text-slate-100 transition transform-gpu hover:-translate-y-0.5 hover:border-slate-500 hover:bg-slate-800 active:scale-95 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-500/50 focus-visible:ring-offset-2" aria-label="Reset training">
                <StopCircle className="h-4 w-4" />
                Reset
              </button>
            </div>

            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              <button type="button" onClick={() => { pausedRef.current = false; setState("running"); stopRef.current = false; trainLoop(); }} className="inline-flex items-center justify-center gap-2 rounded-2xl bg-emerald-500 px-4 py-3 text-sm font-semibold text-slate-900 shadow-md shadow-emerald-500/20 transition transform-gpu hover:-translate-y-0.5 hover:bg-emerald-400 active:scale-95 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400/60 focus-visible:ring-offset-2">
                <Play className="h-4 w-4" />
                Train
              </button>

              <button type="button" onClick={handlePause} className="inline-flex items-center justify-center gap-2 rounded-2xl bg-slate-800 px-4 py-3 text-sm font-semibold text-slate-100 transition transform-gpu hover:-translate-y-0.5 hover:bg-slate-700 active:scale-95 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-500/50 focus-visible:ring-offset-2">
                <Pause className="h-4 w-4" />
                Pause
              </button>

              <button type="button" onClick={handleResume} className="inline-flex items-center justify-center gap-2 rounded-2xl bg-slate-800 px-4 py-3 text-sm font-semibold text-slate-100 transition transform-gpu hover:-translate-y-0.5 hover:bg-slate-700 active:scale-95 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-500/50 focus-visible:ring-offset-2">
                <Play className="h-4 w-4 rotate-180" />
                Resume
              </button>

              <button type="button" onClick={handleStop} className="inline-flex items-center justify-center gap-2 rounded-2xl bg-rose-500 px-4 py-3 text-sm font-semibold text-slate-900 shadow-md shadow-rose-500/20 transition transform-gpu hover:-translate-y-0.5 hover:bg-rose-400 active:scale-95 focus:outline-none focus-visible:ring-2 focus-visible:ring-rose-400/60 focus-visible:ring-offset-2">
                <StopCircle className="h-4 w-4" />
                Stop
              </button>
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-6">
          <div className="flex flex-col gap-8">
            <div className="flex flex-wrap gap-3 items-center">
              <div className="inline-flex items-center gap-3 bg-slate-900/95 border border-white/8 rounded-xl px-3 py-2">
                <div className="w-2.5 h-2.5 rounded-full bg-[#66b3ff]" />
                <div className="text-slate-300 text-xs">Blue = Class 0</div>
              </div>
              <div className="inline-flex items-center gap-3 bg-slate-900/95 border border-white/8 rounded-xl px-3 py-2">
                <div className="w-2.5 h-2.5 rounded-full bg-[#ff8a50]" />
                <div className="text-slate-300 text-xs">Orange = Class 1</div>
              </div>
            </div>

            <div className="flex gap-4 items-start">
              <div className="bg-[#071029] p-3 rounded-lg ring-1 ring-slate-800 w-[220px]" role="group" aria-label="Single linear model metrics">
                <div className="font-semibold mb-2 text-white">Single Linear</div>
                <div className="flex gap-3 mb-3">
                  <div className="text-sm text-slate-400">Loss</div>
                  <div className="font-medium text-white">{lossHistoryA.slice(-1)[0] ?? "–"}</div>
                </div>
                <div className="flex gap-3">
                  <div className="text-sm text-slate-400">Acc</div>
                  <div className="font-medium text-white">{Math.round((accHistoryA.slice(-1)[0] ?? 0) * 100)}%</div>
                </div>
              </div>
              <div className="rounded-lg overflow-hidden ring-1 ring-slate-800">
                <canvas ref={canvasARef} width={CANVAS_SIZE} height={CANVAS_SIZE} role="img" aria-label="Single linear model decision boundary" tabIndex={0} />
              </div>
            </div>

            <div className="flex gap-4 items-start mt-6">
              <div className="bg-[#071029] p-3 rounded-lg ring-1 ring-slate-800 w-[220px]" role="group" aria-label="Five linear model metrics">
                <div className="font-semibold mb-2 text-white">Five Linear</div>
                <div className="flex gap-3 mb-3">
                  <div className="text-sm text-slate-400">Loss</div>
                  <div className="font-medium text-white">{lossHistoryB.slice(-1)[0] ?? "–"}</div>
                </div>
                <div className="flex gap-3">
                  <div className="text-sm text-slate-400">Acc</div>
                  <div className="font-medium text-white">{Math.round((accHistoryB.slice(-1)[0] ?? 0) * 100)}%</div>
                </div>
              </div>
              <div className="rounded-lg overflow-hidden ring-1 ring-slate-800">
                <canvas ref={canvasBRef} width={CANVAS_SIZE} height={CANVAS_SIZE} role="img" aria-label="Five linear model decision boundary" tabIndex={0} />
              </div>
            </div>

            <div className="flex gap-4 items-start mt-6">
              <div className="bg-[#071029] p-3 rounded-lg ring-1 ring-slate-800 w-[220px]" role="group" aria-label="Five ReLU model metrics">
                <div className="font-semibold mb-2 text-white">Five Layer + ReLU</div>
                <div className="flex gap-3 mb-3">
                  <div className="text-sm text-slate-400">Loss</div>
                  <div className="font-medium text-white">{lossHistoryC.slice(-1)[0] ?? "–"}</div>
                </div>
                <div className="flex gap-3">
                  <div className="text-sm text-slate-400">Acc</div>
                  <div className="font-medium text-white">{Math.round((accHistoryC.slice(-1)[0] ?? 0) * 100)}%</div>
                </div>
              </div>
              <div className="rounded-lg overflow-hidden ring-1 ring-slate-800">
                <canvas ref={canvasCRef} width={CANVAS_SIZE} height={CANVAS_SIZE} role="img" aria-label="Five ReLU model decision boundary" tabIndex={0} />
              </div>
            </div>
          </div>

          <div className="flex flex-wrap gap-3">
            <div className="w-[360px] h-[120px] bg-[#041027] p-3 rounded-lg ring-1 ring-slate-800">
              <Plot
                data={[
                  {
                    x: lossHistoryA.map((_, i) => i),
                    y: lossHistoryA,
                    name: "Single Linear",
                    type: "scatter",
                    mode: "lines+markers",
                    marker: { color: "#66b3ff" },
                    line: { color: "#66b3ff" },
                  },
                  {
                    x: lossHistoryB.map((_, i) => i),
                    y: lossHistoryB,
                    name: "Five Linear",
                    type: "scatter",
                    mode: "lines+markers",
                    marker: { color: "#ff8a50" },
                    line: { color: "#ff8a50" },
                  },
                  {
                    x: lossHistoryC.map((_, i) => i),
                    y: lossHistoryC,
                    name: "Five+ReLU",
                    type: "scatter",
                    mode: "lines+markers",
                    marker: { color: "#a78bfa" },
                    line: { color: "#a78bfa" },
                  },
                ]}
                layout={{
                  width: 340,
                  height: 80,
                  margin: { l: 30, r: 10, t: 10, b: 20 },
                  paper_bgcolor: "#041026",
                  plot_bgcolor: "#041026",
                  xaxis: { visible: false },
                  yaxis: { range: [0, Math.max(...lossHistoryA, ...lossHistoryB, ...lossHistoryC, 1)] },
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ background: "#041026" }}
              />
            </div>

            <div className="w-[360px] h-[120px] bg-[#041027] p-3 rounded-lg ring-1 ring-slate-800">
              <Plot
                data={[
                  {
                    x: accHistoryA.map((_, i) => i),
                    y: accHistoryA,
                    name: "Single Linear",
                    type: "scatter",
                    mode: "lines+markers",
                    marker: { color: "#66b3ff" },
                    line: { color: "#66b3ff" },
                  },
                  {
                    x: accHistoryB.map((_, i) => i),
                    y: accHistoryB,
                    name: "Five Linear",
                    type: "scatter",
                    mode: "lines+markers",
                    marker: { color: "#ff8a50" },
                    line: { color: "#ff8a50" },
                  },
                  {
                    x: accHistoryC.map((_, i) => i),
                    y: accHistoryC,
                    name: "Five+ReLU",
                    type: "scatter",
                    mode: "lines+markers",
                    marker: { color: "#a78bfa" },
                    line: { color: "#a78bfa" },
                  },
                ]}
                layout={{
                  width: 340,
                  height: 80,
                  margin: { l: 30, r: 10, t: 10, b: 20 },
                  paper_bgcolor: "#041026",
                  plot_bgcolor: "#041026",
                  xaxis: { visible: false },
                  yaxis: { range: [0, 1] },
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ background: "#041026" }}
              />
            </div>

            <div className={`w-full rounded-3xl border border-white/10 bg-slate-950/80 p-6 shadow-[0_24px_100px_-60px_rgba(0,0,0,0.8)] backdrop-blur-xl transition-all duration-700 ${trainingComplete ? "opacity-100" : "opacity-40"}`}>
              <div className="font-semibold text-white">Summary</div>
              <div className="mt-2 text-sm text-slate-300">
                {trainingComplete ? (
                  <>
                    <div className="mb-2">Single Linear Accuracy: {Math.round((accHistoryA.slice(-1)[0] ?? 0) * 100)}%</div>
                    <div className="mb-2">Five Linear Accuracy: {Math.round((accHistoryB.slice(-1)[0] ?? 0) * 100)}%</div>
                    <div className="mb-2">Five Layer + ReLU Accuracy: {Math.round((accHistoryC.slice(-1)[0] ?? 0) * 100)}%</div>
                    <div>
                      The single linear model and the five-layer linear model produced nearly identical straight decision boundaries and similar accuracies, demonstrating that stacking linear layers does not increase representational power.
                      The five-layer ReLU network successfully learned the nonlinear circular decision boundary, proving that nonlinear activation functions are what make deep networks expressive.
                    </div>
                  </>
                ) : (
                  "Stacking linear layers without nonlinearities does not increase expressive power; the deep linear model should perform similarly to a single linear layer. Adding ReLU between layers enables nonlinear decision boundaries that can capture the concentric pattern."
                )}
              </div>
            </div>
            {effectiveMatrix && (
              <div className="w-full rounded-3xl border border-white/10 bg-slate-950/80 p-6 shadow-[0_24px_100px_-60px_rgba(0,0,0,0.8)] backdrop-blur-xl mt-4">
                <div className="font-semibold text-white">Effective Weight Matrix (Five Linear)</div>
                <div className="mt-3 text-sm text-slate-300">
                  <div className="mb-2">Dimensions: {effectiveMatrix.shape.join(" × ")}</div>
                  <div className="mb-2">Combined matrix:</div>
                  <div className="overflow-auto rounded-md bg-[#021021] p-3 text-xs text-slate-200">
                    {effectiveMatrix.values.map((row, i) => (
                      <div key={i} className="whitespace-pre">{row.map((v) => v.toFixed(4)).join("\t")}</div>
                    ))}
                  </div>
                  <div className="mt-3 text-sm text-slate-300">A composition of linear transformations is itself a linear transformation. Therefore a five-layer linear network is mathematically equivalent to a single linear transformation.</div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
