import type React from "react";
import { useEffect, useRef, useState } from "react";
import { Layers, Play, Pause, StopCircle, RefreshCcw, Shuffle, Sparkles, Database, SlidersHorizontal } from "lucide-react";
import type { Dataset, DatasetGeneratorOptions } from "../experiments/relu/dataset";
import { generateConcentricCircles, datasetToTensors } from "../experiments/relu/dataset";
import type { Tensor2D, Tensor, LayersModel } from "@tensorflow/tfjs";
import Plot from "react-plotly.js";

type TrainingState = "idle" | "running" | "paused";

const CANVAS_SIZE = 360;
const GRID_RES = 96;

function buildLinearModel(tff: any): LayersModel {
	const model = tff.sequential();

	model.add(
		tff.layers.dense({ units: 1, inputShape: [2], activation: "sigmoid" }),
	);

	return model as LayersModel;
}

function buildReLUModel(tff: any, hiddenUnits: number): LayersModel {
	const model = tff.sequential();

	model.add(tff.layers.dense({ units: hiddenUnits, inputShape: [2], activation: "relu" }));

	model.add(tff.layers.dense({ units: 1, activation: "sigmoid" }));

	return model as LayersModel;
}

function drawAxes(ctx: CanvasRenderingContext2D, size: number) {
	ctx.save();
	ctx.strokeStyle = "rgba(200,200,200,0.25)";
	ctx.lineWidth = 1;

	// grid
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

	// axes
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

			// color: interpolate between blue (class 0) and orange (class 1)
			const r = Math.round(255 * p);
			const g = Math.round(120 * (1 - p));
			const b = Math.round(255 * (1 - p));

			const idx = (y * gridResolution + x) * 4;
			img.data[idx] = r;
			img.data[idx + 1] = g;
			img.data[idx + 2] = b;
			img.data[idx + 3] = 200;
		}
	}

	// scale up image to canvas
	const tmp = document.createElement("canvas");
	tmp.width = gridResolution;
	tmp.height = gridResolution;
	const tctx = tmp.getContext("2d")!;
	tctx.putImageData(img, 0, 0);
	ctx.imageSmoothingEnabled = true;
	ctx.drawImage(tmp, 0, 0, size, size);
}

function drawPoints(ctx: CanvasRenderingContext2D, size: number, points: Dataset["points"], radius: number) {
	ctx.save();

	const scale = size / (radius * 4);
	const center = size / 2;

	for (const p of points) {
		const cx = center + p.x * scale;
		const cy = center - p.y * scale;

		ctx.beginPath();
		ctx.fillStyle = p.label === 1 ? "#ff8a50" : "#66b3ff";
		ctx.arc(cx, cy, 4, 0, Math.PI * 2);
		ctx.fill();
		ctx.strokeStyle = "rgba(0,0,0,0.2)";
		ctx.stroke();
	}

	ctx.restore();
}

export default function ReLUExperiment(): React.ReactElement {
	const [dataset, setDataset] = useState<Dataset>(() =>
		generateConcentricCircles({ samples: 200, radius: 1.0, noise: 0.08, seed: 1 }),
	);

	const [options, setOptions] = useState<DatasetGeneratorOptions>({
		samples: 200,
		radius: 1.0,
		noise: 0.08,
		seed: 1,
	});

	const [learningRate, setLearningRate] = useState<number>(0.01);
	const [epochs, setEpochs] = useState<number>(60);
	const [batchSize, setBatchSize] = useState<number>(32);
	const [hiddenUnits, setHiddenUnits] = useState<number>(8);
	const [animationSpeed, setAnimationSpeed] = useState<number>(3);

	const [state, setState] = useState<TrainingState>("idle");

	const [epochCount, setEpochCount] = useState<number>(0);
	const [lossHistoryA, setLossHistoryA] = useState<number[]>([]);
	const [accHistoryA, setAccHistoryA] = useState<number[]>([]);
	const [lossHistoryB, setLossHistoryB] = useState<number[]>([]);
	const [accHistoryB, setAccHistoryB] = useState<number[]>([]);

	const canvasARef = useRef<HTMLCanvasElement | null>(null);
	const canvasBRef = useRef<HTMLCanvasElement | null>(null);

	const modelARef = useRef<LayersModel | null>(null);
	const modelBRef = useRef<LayersModel | null>(null);

	const tensorsRef = useRef<{ xs?: Tensor2D; ys?: Tensor } | null>(null);
	const stopRef = useRef(false);
	const pausedRef = useRef(false);

	const tfRef = useRef<any>(null);

	async function ensureTfLoaded() {
		if (!tfRef.current) {
			tfRef.current = await import("@tensorflow/tfjs");
		}
	}

	useEffect(() => {
		// create tensors for dataset (lazy-load tfjs first)
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
		// regenerate when dataset changes
	}, [dataset]);

	useEffect(() => {
		return () => {
			// cleanup models on unmount
			modelARef.current?.dispose();
			modelBRef.current?.dispose();
			tensorsRef.current?.xs?.dispose();
			tensorsRef.current?.ys?.dispose();
		};
	}, []);

	async function resetModels() {
		await ensureTfLoaded();

		tfRef.current.engine().startScope();
		modelARef.current?.dispose();
		modelBRef.current?.dispose();

		const a = buildLinearModel(tfRef.current);
		const b = buildReLUModel(tfRef.current, hiddenUnits);

		const optA = tfRef.current.train.adam(learningRate);
		const optB = tfRef.current.train.adam(learningRate);

		a.compile({ optimizer: optA, loss: "binaryCrossentropy", metrics: ["accuracy"] });
		b.compile({ optimizer: optB, loss: "binaryCrossentropy", metrics: ["accuracy"] });

		modelARef.current = a;
		modelBRef.current = b;
		tfRef.current.engine().endScope();
	}

	async function handleGenerate() {
		const d = generateConcentricCircles(options);
		setDataset(d);
		setEpochCount(0);
		setLossHistoryA([]);
		setAccHistoryA([]);
		setLossHistoryB([]);
		setAccHistoryB([]);
		pausedRef.current = false;
		await resetModels();
	}

	async function trainLoop() {
		if (!tensorsRef.current || !modelARef.current || !modelBRef.current) return;

		setState("running");
		stopRef.current = false;

		const xs = tensorsRef.current!.xs!;
		const ys = tensorsRef.current!.ys!;

		for (let e = 0; e < epochs; e++) {
			if (stopRef.current) break;

			// single epoch train both models sequentially
			const [resA, resB] = await Promise.all([
				modelARef.current!.fit(xs, ys, { epochs: 1, batchSize, shuffle: true }),
				modelBRef.current!.fit(xs, ys, { epochs: 1, batchSize, shuffle: true }),
			]);

			// compute metrics from ReLU model (resB)
			const lossA = resA.history.loss ? (resA.history.loss as number[]).slice(-1)[0] : NaN;
			const accA = resA.history.accuracy ? (resA.history.accuracy as number[]).slice(-1)[0] : NaN;
			const lossB = resB.history.loss ? (resB.history.loss as number[]).slice(-1)[0] : NaN;
			const accB = resB.history.accuracy ? (resB.history.accuracy as number[]).slice(-1)[0] : NaN;

			setEpochCount((p) => p + 1);
			setLossHistoryA((h) => [...h, Number(lossA ?? NaN)]);
			setAccHistoryA((h) => [...h, Number(accA ?? NaN)]);
			setLossHistoryB((h) => [...h, Number(lossB ?? NaN)]);
			setAccHistoryB((h) => [...h, Number(accB ?? NaN)]);

			// update decision boundary occasionally based on animation speed
			// render decision boundaries every 2 epochs
			if ((e + 1) % 2 === 0) {
				await updateDecisionBoundaries();
			}

			// yield to the UI to keep React responsive
			if (tfRef.current && tfRef.current.nextFrame) await tfRef.current.nextFrame();

			// pause handling (use ref to avoid stale closure)
			// eslint-disable-next-line no-await-in-loop
			while (pausedRef.current) {
				// small sleep while paused
				// eslint-disable-next-line no-await-in-loop
				await new Promise((r) => setTimeout(r, 100));
				if (stopRef.current) break;
			}
		}

		setState(stopRef.current ? "idle" : "idle");
	}

	async function updateDecisionBoundaries() {
		if (!modelARef.current || !modelBRef.current) return;

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

		const [arrA, arrB] = await Promise.all([pa.data() as Promise<Float32Array>, pb.data() as Promise<Float32Array>]);

		// draw on offscreen canvases and animate blend into visible canvases
		const ca = canvasARef.current!;
		const cb = canvasBRef.current!;

		const ctxA = ca.getContext("2d")!;
		const ctxB = cb.getContext("2d")!;

		// prepare temporary canvases at grid resolution
		const tmpA = document.createElement("canvas");
		tmpA.width = gridRes;
		tmpA.height = gridRes;
		const tA = tmpA.getContext("2d")!;

		const tmpB = document.createElement("canvas");
		tmpB.width = gridRes;
		tmpB.height = gridRes;
		const tB = tmpB.getContext("2d")!;

		// render heatmaps into temporary contexts (they internally scale)
		renderHeatmap(tA, gridRes, gridRes, arrA);
		renderHeatmap(tB, gridRes, gridRes, arrB);

		// animate blend for both canvases
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
				// yield to next frame
				await new Promise((r) => requestAnimationFrame(r));
			}
		};

		await Promise.all([animateBlend(ctxA, tmpA), animateBlend(ctxB, tmpB)]);

		// overlay axes and points after blend
		drawAxes(ctxA, CANVAS_SIZE);
		drawAxes(ctxB, CANVAS_SIZE);

		drawPoints(ctxA, CANVAS_SIZE, dataset.points, options.radius);
		drawPoints(ctxB, CANVAS_SIZE, dataset.points, options.radius);

		pa.dispose();
		pb.dispose();
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
		setState("idle");
	}

	function handleReset() {
		handleStop();
		setEpochCount(0);
		setLossHistoryA([]);
		setAccHistoryA([]);
		setLossHistoryB([]);
		setAccHistoryB([]);
		resetModels();
		updateDecisionBoundaries();
	}

	useEffect(() => {
		// initialize models when hiddenUnits or learningRate changes
		resetModels();
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [hiddenUnits, learningRate]);

	useEffect(() => {
		// initial draw
		const ca = canvasARef.current!;
		const cb = canvasBRef.current!;
		if (!ca || !cb) return;

		const ctxA = ca.getContext("2d")!;
		const ctxB = cb.getContext("2d")!;

		ctxA.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
		ctxB.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

		drawAxes(ctxA, CANVAS_SIZE);
		drawAxes(ctxB, CANVAS_SIZE);

		drawPoints(ctxA, CANVAS_SIZE, dataset.points, options.radius);
		drawPoints(ctxB, CANVAS_SIZE, dataset.points, options.radius);
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [canvasARef.current, canvasBRef.current, dataset]);

	return (
		<div style={{ padding: 20, color: "white", fontFamily: "Inter, Arial" }}>
			<h2 style={{ fontSize: 24, marginBottom: 8 }}>Why ReLU Matters</h2>

			<div className="flex flex-col gap-16 xl:flex-row">
				<div className="flex flex-col gap-6 xl:w-96">
				<div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
					<div className="mb-4 flex items-center gap-3 text-sm font-semibold uppercase tracking-[0.15em] text-slate-300">
						<Database className="h-4 w-4 text-cyan-300" />
						<span>Dataset</span>
					</div>
					<div className="grid gap-4">
						<div className="space-y-2">
							<label htmlFor="samples" className="text-sm font-medium text-slate-100">
								Samples
							</label>
							<input
								id="samples"
								type="number"
								min={50}
								step={10}
								value={options.samples}
								onChange={(e) => setOptions({ ...options, samples: Number(e.target.value) })}
								className="w-full rounded-2xl border border-slate-700 bg-slate-900/90 px-4 py-2 text-slate-100 outline-none transition focus:border-cyan-400 focus:ring-2 focus:ring-cyan-500/20"
							/>
							<p className="text-xs text-slate-500">Number of points in the circle dataset.</p>
						</div>

						<div className="space-y-2">
							<label htmlFor="noise" className="text-sm font-medium text-slate-100">
								Noise
							</label>
							<div className="flex items-center gap-3">
								<input
									id="noise"
									type="range"
									min={0}
									max={0.3}
									step={0.01}
									value={options.noise}
									onChange={(e) => setOptions({ ...options, noise: Number(e.target.value) })}
									className="w-full accent-cyan-400"
								/>
								<span className="text-sm text-slate-400">{options.noise.toFixed(2)}</span>
							</div>
							<p className="text-xs text-slate-500">Adds radial noise to the points.</p>
						</div>

						<div className="grid gap-4 sm:grid-cols-2">
							<div className="space-y-2">
								<label htmlFor="radius" className="text-sm font-medium text-slate-100">
									Radius
								</label>
								<input
									id="radius"
									type="number"
									step={0.1}
									value={options.radius}
									onChange={(e) => setOptions({ ...options, radius: Number(e.target.value) })}
									className="w-full rounded-2xl border border-slate-700 bg-slate-900/90 px-4 py-2 text-slate-100 outline-none transition focus:border-cyan-400 focus:ring-2 focus:ring-cyan-500/20"
								/>
								<p className="text-xs text-slate-500">Circle radius for class separation.</p>
							</div>

							<div className="space-y-2">
								<label htmlFor="seed" className="text-sm font-medium text-slate-100">
									Seed
								</label>
								<input
									id="seed"
									type="number"
									value={options.seed}
									onChange={(e) => setOptions({ ...options, seed: Number(e.target.value) })}
									className="w-full rounded-2xl border border-slate-700 bg-slate-900/90 px-4 py-2 text-slate-100 outline-none transition focus:border-cyan-400 focus:ring-2 focus:ring-cyan-500/20"
								/>
								<p className="text-xs text-slate-500">Control reproducible dataset generation.</p>
							</div>
						</div>
					</div>
				</div>

				<div className="rounded-3xl border border-white/10 bg-slate-950/80 p-5 shadow-[0_20px_80px_-50px_rgba(0,0,0,0.8)] backdrop-blur-xl">
					<div className="mb-4 flex items-center gap-3 text-sm font-semibold uppercase tracking-[0.15em] text-slate-300">
						<Layers className="h-4 w-4 text-orange-300" />
						<span>Model</span>
					</div>
					<div className="space-y-4">
						<div className="space-y-2">
							<label htmlFor="learningRate" className="text-sm font-medium text-slate-100">
								Learning Rate
							</label>
							<div className="flex items-center gap-3">
								<input
									id="learningRate"
									type="range"
									min={0.0005}
									max={0.1}
									step={0.0005}
									value={learningRate}
									onChange={(e) => setLearningRate(Number(e.target.value))}
									className="w-full accent-orange-400"
								/>
								<span className="text-sm text-slate-400">{learningRate.toFixed(4)}</span>
							</div>
							<p className="text-xs text-slate-500">Step size for optimizer updates.</p>
						</div>

						<div className="space-y-2">
							<label htmlFor="hiddenUnits" className="text-sm font-medium text-slate-100">
								Hidden Units
							</label>
							<div className="flex items-center gap-3">
								<input
									id="hiddenUnits"
									type="range"
									min={4}
									max={32}
									step={1}
									value={hiddenUnits}
									onChange={(e) => setHiddenUnits(Number(e.target.value))}
									className="w-full accent-orange-400"
								/>
								<span className="text-sm text-slate-400">{hiddenUnits}</span>
							</div>
							<p className="text-xs text-slate-500">Controls model complexity.</p>
						</div>

						<div className="grid gap-4 sm:grid-cols-2">
							<div className="space-y-2">
								<label htmlFor="batchSize" className="text-sm font-medium text-slate-100">
									Batch Size
								</label>
								<input
									id="batchSize"
									type="number"
									value={batchSize}
									onChange={(e) => setBatchSize(Number(e.target.value))}
									className="w-full rounded-2xl border border-slate-700 bg-slate-900/90 px-4 py-2 text-slate-100 outline-none transition focus:border-orange-400 focus:ring-2 focus:ring-orange-500/20"
								/>
								<p className="text-xs text-slate-500">Examples processed per training step.</p>
							</div>

							<div className="space-y-2">
								<label htmlFor="epochs" className="text-sm font-medium text-slate-100">
									Epochs
								</label>
								<input
									id="epochs"
									type="number"
									value={epochs}
									onChange={(e) => setEpochs(Number(e.target.value))}
									className="w-full rounded-2xl border border-slate-700 bg-slate-900/90 px-4 py-2 text-slate-100 outline-none transition focus:border-orange-400 focus:ring-2 focus:ring-orange-500/20"
								/>
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
						<label htmlFor="animationSpeed" className="text-sm font-medium text-slate-100">
							Animation Speed
						</label>
						<div className="flex items-center gap-3">
							<input
								id="animationSpeed"
								type="range"
								min={1}
								max={10}
								step={1}
								value={animationSpeed}
								onChange={(e) => setAnimationSpeed(Number(e.target.value))}
								className="w-full accent-violet-400"
							/>
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
						<button
							onClick={handleGenerate}
							className="inline-flex items-center justify-center gap-2 rounded-2xl bg-cyan-500 px-4 py-3 text-sm font-semibold text-slate-950 shadow-lg shadow-cyan-500/20 transition hover:-translate-y-0.5 hover:bg-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-400/50"
						>
							<RefreshCcw className="h-4 w-4" />
							Generate Dataset
						</button>

						<button
							onClick={() => {
							setOptions({ ...options, seed: options.seed + 1 });
							setDataset(generateConcentricCircles({ ...options, seed: options.seed + 1 }));
						}}
							className="inline-flex items-center justify-center gap-2 rounded-2xl border border-slate-700 bg-slate-900/90 px-4 py-3 text-sm font-semibold text-slate-100 transition hover:-translate-y-0.5 hover:border-slate-500 hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-slate-500/40"
						>
							<Shuffle className="h-4 w-4" />
							Shuffle
						</button>

						<button
							onClick={handleReset}
							className="inline-flex items-center justify-center gap-2 rounded-2xl border border-slate-700 bg-slate-900/90 px-4 py-3 text-sm font-semibold text-slate-100 transition hover:-translate-y-0.5 hover:border-slate-500 hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-slate-500/40"
						>
							<StopCircle className="h-4 w-4" />
							Reset
						</button>
					</div>

					<div className="mt-4 grid gap-3 sm:grid-cols-2">
						<button
							onClick={() => {
							pausedRef.current = false;
							setState("running");
							stopRef.current = false;
							trainLoop();
						}}
							className="inline-flex items-center justify-center gap-2 rounded-2xl bg-emerald-500 px-4 py-3 text-sm font-semibold text-slate-950 shadow-lg shadow-emerald-500/20 transition hover:-translate-y-0.5 hover:bg-emerald-400 focus:outline-none focus:ring-2 focus:ring-emerald-400/50"
						>
							<Play className="h-4 w-4" />
							Train
						</button>

						<button
							onClick={handlePause}
							className="inline-flex items-center justify-center gap-2 rounded-2xl bg-slate-800 px-4 py-3 text-sm font-semibold text-slate-100 transition hover:-translate-y-0.5 hover:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-slate-500/40"
						>
							<Pause className="h-4 w-4" />
							Pause
						</button>

						<button
							onClick={handleResume}
							className="inline-flex items-center justify-center gap-2 rounded-2xl bg-slate-800 px-4 py-3 text-sm font-semibold text-slate-100 transition hover:-translate-y-0.5 hover:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-slate-500/40"
						>
							<Play className="h-4 w-4 rotate-180" />
							Resume
						</button>

						<button
							onClick={handleStop}
							className="inline-flex items-center justify-center gap-2 rounded-2xl bg-rose-500 px-4 py-3 text-sm font-semibold text-slate-950 shadow-lg shadow-rose-500/20 transition hover:-translate-y-0.5 hover:bg-rose-400 focus:outline-none focus:ring-2 focus:ring-rose-400/50"
						>
							<StopCircle className="h-4 w-4" />
							Stop
						</button>
					</div>
				</div>

				</div>

				<div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
					<div style={{ display: "flex", gap: 8 }}>
						<div style={{ background: "#071029", padding: 8 }}>
							<div style={{ fontWeight: 600, marginBottom: 6 }}>Linear</div>
							<div style={{ display: "flex", gap: 12, marginBottom: 8 }}>
								<div style={{ background: "#011025", padding: 8, borderRadius: 6 }}>
									<div style={{ fontSize: 12, color: "#9fb6d9" }}>Epoch</div>
									<div style={{ fontWeight: 700 }}>{epochCount}</div>
								</div>
								<div style={{ background: "#011025", padding: 8, borderRadius: 6 }}>
									<div style={{ fontSize: 12, color: "#9fb6d9" }}>Loss</div>
									<div style={{ fontWeight: 700 }}>{(lossHistoryA.slice(-1)[0] ?? "-")}</div>
								</div>
								<div style={{ background: "#011025", padding: 8, borderRadius: 6 }}>
									<div style={{ fontSize: 12, color: "#9fb6d9" }}>Accuracy</div>
									<div style={{ fontWeight: 700 }}>{Math.round((accHistoryA.slice(-1)[0] ?? 0) * 100)}%</div>
								</div>
							</div>
							<canvas ref={canvasARef} width={CANVAS_SIZE} height={CANVAS_SIZE} />
						</div>

						<div style={{ background: "#071029", padding: 8 }}>
							<div style={{ fontWeight: 600, marginBottom: 6 }}>ReLU</div>
							<div style={{ display: "flex", gap: 12, marginBottom: 8 }}>
								<div style={{ background: "#011025", padding: 8, borderRadius: 6 }}>
									<div style={{ fontSize: 12, color: "#9fb6d9" }}>Epoch</div>
									<div style={{ fontWeight: 700 }}>{epochCount}</div>
								</div>
								<div style={{ background: "#011025", padding: 8, borderRadius: 6 }}>
									<div style={{ fontSize: 12, color: "#9fb6d9" }}>Loss</div>
									<div style={{ fontWeight: 700 }}>{(lossHistoryB.slice(-1)[0] ?? "-")}</div>
								</div>
								<div style={{ background: "#011025", padding: 8, borderRadius: 6 }}>
									<div style={{ fontSize: 12, color: "#9fb6d9" }}>Accuracy</div>
									<div style={{ fontWeight: 700 }}>{Math.round((accHistoryB.slice(-1)[0] ?? 0) * 100)}%</div>
								</div>
							</div>
							<canvas ref={canvasBRef} width={CANVAS_SIZE} height={CANVAS_SIZE} />
						</div>
					</div>

					<div style={{ display: "flex", gap: 8 }}>
						<div style={{ width: 360, height: 120, background: "#041027", padding: 8 }}>
							<div style={{ fontWeight: 600 }}>Live Loss</div>
							<Plot
								data={[
									{
										x: lossHistoryA.map((_, i) => i),
										y: lossHistoryA,
										name: "Linear",
										type: "scatter",
										mode: "lines+markers",
										marker: { color: "#66b3ff" },
										line: { color: "#66b3ff" },
									},
									{
										x: lossHistoryB.map((_, i) => i),
										y: lossHistoryB,
										name: "ReLU",
										type: "scatter",
										mode: "lines+markers",
										marker: { color: "#ff8a50" },
										line: { color: "#ff8a50" },
									},
								]}
								layout={{
									width: 340,
									height: 80,
									margin: { l: 30, r: 10, t: 10, b: 20 },
									paper_bgcolor: "#041026",
									plot_bgcolor: "#041026",
									xaxis: { visible: false },
									yaxis: { range: [0, Math.max(...lossHistoryA, ...lossHistoryB, 1)] },
								}}
								config={{ displayModeBar: false, responsive: true }}
								style={{ background: "#041026" }}
							/>
						</div>

						<div style={{ width: 360, height: 120, background: "#041027", padding: 8 }}>
							<div style={{ fontWeight: 600 }}>Live Accuracy</div>
							<Plot
								data={[
									{
										x: accHistoryA.map((_, i) => i),
										y: accHistoryA,
										name: "Linear",
										type: "scatter",
										mode: "lines+markers",
										marker: { color: "#66b3ff" },
										line: { color: "#66b3ff" },
									},
									{
										x: accHistoryB.map((_, i) => i),
										y: accHistoryB,
										name: "ReLU",
										type: "scatter",
										mode: "lines+markers",
										marker: { color: "#ff8a50" },
										line: { color: "#ff8a50" },
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
					</div>

					<div style={{ width: 740, background: "#021022", padding: 12 }}>
						<div style={{ fontWeight: 700, marginBottom: 6 }}>Educational Panel</div>
						<div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
							<div>
								<div style={{ fontWeight: 600 }}>Claim</div>
								<div>Linear models cannot separate concentric circles.</div>
							</div>

							<div>
								<div style={{ fontWeight: 600 }}>Observation</div>
								<div>Linear model decision boundary remains linear; ReLU network learns non-linear boundary.</div>
							</div>

							<div>
								<div style={{ fontWeight: 600 }}>Conclusion</div>
								<div>Non-linear activation (ReLU) enables learning complex decision boundaries.</div>
							</div>
						</div>
					</div>
				</div>
			</div>
		</div>
	);
}
