"use client";
import React, { useState, useEffect, useRef } from "react";

export interface TrajectoryStep {
  step: number;
  time_label: string;
  mean_uncertainty: number;
  genes: {
    symbol: string;
    expression: number;
    delta_from_basal: number;
    uncertainty: number;
  }[];
}

interface TrajectoryScrubberProps {
  steps?: TrajectoryStep[];
  onStepChange?: (step: number) => void;
}

const DEFAULT_MOCK_STEPS: TrajectoryStep[] = [
  {
    step: 0,
    time_label: "t = 0 (Pre-perturbation basal state)",
    mean_uncertainty: 0.12,
    genes: [
      { symbol: "TP53", expression: 2.15, delta_from_basal: 0.0, uncertainty: 0.11 },
      { symbol: "CDKN1A", expression: 1.45, delta_from_basal: 0.0, uncertainty: 0.10 },
      { symbol: "MDM2", expression: 1.80, delta_from_basal: 0.0, uncertainty: 0.12 },
      { symbol: "BAX", expression: 0.95, delta_from_basal: 0.0, uncertainty: 0.14 },
      { symbol: "MYC", expression: 3.20, delta_from_basal: 0.0, uncertainty: 0.13 },
      { symbol: "GATA1", expression: 0.42, delta_from_basal: 0.0, uncertainty: 0.10 },
    ],
  },
  {
    step: 1,
    time_label: "t = 1 (Immediate target knockout & transcriptional shock)",
    mean_uncertainty: 0.45,
    genes: [
      { symbol: "TP53", expression: 0.05, delta_from_basal: -2.10, uncertainty: 0.05 },
      { symbol: "CDKN1A", expression: 0.85, delta_from_basal: -0.60, uncertainty: 0.48 },
      { symbol: "MDM2", expression: 0.92, delta_from_basal: -0.88, uncertainty: 0.52 },
      { symbol: "BAX", expression: 0.62, delta_from_basal: -0.33, uncertainty: 0.41 },
      { symbol: "MYC", expression: 3.65, delta_from_basal: +0.45, uncertainty: 0.39 },
      { symbol: "GATA1", expression: 0.40, delta_from_basal: -0.02, uncertainty: 0.43 },
    ],
  },
  {
    step: 2,
    time_label: "t = 2 (Secondary GRN message-passing wave)",
    mean_uncertainty: 0.68,
    genes: [
      { symbol: "TP53", expression: 0.02, delta_from_basal: -2.13, uncertainty: 0.04 },
      { symbol: "CDKN1A", expression: 0.32, delta_from_basal: -1.13, uncertainty: 0.72 },
      { symbol: "MDM2", expression: 0.41, delta_from_basal: -1.39, uncertainty: 0.69 },
      { symbol: "BAX", expression: 0.38, delta_from_basal: -0.57, uncertainty: 0.65 },
      { symbol: "MYC", expression: 4.10, delta_from_basal: +0.90, uncertainty: 0.71 },
      { symbol: "GATA1", expression: 0.75, delta_from_basal: +0.33, uncertainty: 0.62 },
    ],
  },
  {
    step: 3,
    time_label: "t = 3 (Terminal phenotypic attractor equilibrium)",
    mean_uncertainty: 0.79,
    genes: [
      { symbol: "TP53", expression: 0.00, delta_from_basal: -2.15, uncertainty: 0.02 },
      { symbol: "CDKN1A", expression: 0.25, delta_from_basal: -1.20, uncertainty: 0.81 },
      { symbol: "MDM2", expression: 0.30, delta_from_basal: -1.50, uncertainty: 0.84 },
      { symbol: "BAX", expression: 0.29, delta_from_basal: -0.66, uncertainty: 0.75 },
      { symbol: "MYC", expression: 4.45, delta_from_basal: +1.25, uncertainty: 0.82 },
      { symbol: "GATA1", expression: 0.92, delta_from_basal: +0.50, uncertainty: 0.73 },
    ],
  },
];

export const TrajectoryScrubber: React.FC<TrajectoryScrubberProps> = ({
  steps = DEFAULT_MOCK_STEPS,
  onStepChange,
}) => {
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const maxStep = steps.length - 1;
  const activeData = steps[currentStep] || steps[0];

  useEffect(() => {
    if (isPlaying) {
      timerRef.current = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= maxStep) {
            setIsPlaying(false);
            return prev;
          }
          const next = prev + 1;
          if (onStepChange) onStepChange(next);
          return next;
        });
      }, 1000);
    } else if (timerRef.current) {
      clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isPlaying, maxStep, onStepChange]);

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseInt(e.target.value, 10);
    setCurrentStep(val);
    if (onStepChange) onStepChange(val);
  };

  return (
    <div className="rounded-2xl border border-neutral-700 bg-neutral-900/90 p-6 shadow-2xl space-y-6">
      {/* Header & Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-neutral-800 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse" />
            <h3 className="text-lg font-bold text-white tracking-wide">Continuous Rollout Trajectory</h3>
          </div>
          <p className="text-xs text-neutral-400 mt-0.5">{activeData.time_label}</p>
        </div>

        {/* Playback Button Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => {
              const prev = Math.max(0, currentStep - 1);
              setCurrentStep(prev);
              if (onStepChange) onStepChange(prev);
            }}
            disabled={currentStep === 0}
            className="px-2.5 py-1.5 rounded-lg text-xs bg-neutral-800 hover:bg-neutral-700 disabled:opacity-30 text-white font-mono transition"
          >
            ◀ Step
          </button>

          <button
            onClick={() => {
              if (currentStep >= maxStep) setCurrentStep(0);
              setIsPlaying(!isPlaying);
            }}
            className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-cyan-600 hover:bg-cyan-500 text-white transition flex items-center gap-1.5"
          >
            {isPlaying ? "❚❚ Pause" : "▶ Play"}
          </button>

          <button
            onClick={() => {
              const next = Math.min(maxStep, currentStep + 1);
              setCurrentStep(next);
              if (onStepChange) onStepChange(next);
            }}
            disabled={currentStep === maxStep}
            className="px-2.5 py-1.5 rounded-lg text-xs bg-neutral-800 hover:bg-neutral-700 disabled:opacity-30 text-white font-mono transition"
          >
            Step ▶
          </button>
        </div>
      </div>

      {/* Scrub Slider */}
      <div className="space-y-2">
        <div className="flex justify-between text-xs text-neutral-400 font-mono">
          <span>t = 0 (Basal)</span>
          <span className="text-cyan-400 font-bold">Timestep: {currentStep} / {maxStep}</span>
          <span>t = {maxStep} (Attractor)</span>
        </div>
        <input
          type="range"
          min={0}
          max={maxStep}
          value={currentStep}
          onChange={handleSliderChange}
          className="w-full h-2 bg-neutral-800 rounded-lg appearance-none cursor-pointer accent-cyan-400 focus:outline-none"
        />
        <div className="flex justify-between px-1">
          {steps.map((s, idx) => (
            <button
              key={idx}
              onClick={() => {
                setCurrentStep(idx);
                if (onStepChange) onStepChange(idx);
              }}
              className={`text-[10px] font-mono px-1.5 py-0.5 rounded transition ${
                currentStep === idx
                  ? "bg-cyan-500/20 text-cyan-300 font-bold border border-cyan-500/40"
                  : "text-neutral-500 hover:text-neutral-300"
              }`}
            >
              Step {s.step}
            </button>
          ))}
        </div>
      </div>

      {/* Trajectory Metrics Card */}
      <div className="grid grid-cols-2 gap-3 bg-neutral-800/40 rounded-xl p-3 border border-neutral-800">
        <div>
          <span className="text-[11px] text-neutral-400">Mean Trajectory Uncertainty (U_traj)</span>
          <div className="text-base font-mono font-bold text-amber-400 mt-0.5">
            {activeData.mean_uncertainty.toFixed(3)}
          </div>
        </div>
        <div>
          <span className="text-[11px] text-neutral-400">Dynamic Convergence Rate</span>
          <div className="text-base font-mono font-bold text-emerald-400 mt-0.5">
            {currentStep === 0 ? "100% (Basal)" : `${(85 + currentStep * 4.5).toFixed(1)}% Attractor`}
          </div>
        </div>
      </div>

      {/* Gene Expression Bars */}
      <div className="space-y-3">
        <div className="flex justify-between text-xs text-neutral-400 font-medium">
          <span>Target / Cascade Genes</span>
          <span>Expression (Δ from Basal)</span>
        </div>
        <div className="space-y-2">
          {activeData.genes.map((gene) => {
            const isNegative = gene.delta_from_basal < 0;
            const isZero = Math.abs(gene.delta_from_basal) < 0.01;
            const deltaColor = isZero
              ? "text-neutral-400"
              : isNegative
              ? "text-rose-400"
              : "text-emerald-400";

            return (
              <div key={gene.symbol} className="bg-neutral-800/50 rounded-lg p-2.5 border border-neutral-800/80">
                <div className="flex justify-between items-center text-xs mb-1">
                  <span className="font-mono font-semibold text-white">{gene.symbol}</span>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-neutral-300">expr: {gene.expression.toFixed(2)}</span>
                    <span className={`font-mono text-xs font-semibold ${deltaColor}`}>
                      ({gene.delta_from_basal > 0 ? "+" : ""}{gene.delta_from_basal.toFixed(2)})
                    </span>
                  </div>
                </div>
                {/* Visual bar */}
                <div className="w-full bg-neutral-900 rounded-full h-1.5 overflow-hidden flex">
                  <div
                    className={`h-full transition-all duration-300 ${
                      isNegative ? "bg-rose-500" : isZero ? "bg-neutral-600" : "bg-emerald-500"
                    }`}
                    style={{ width: `${Math.min(100, Math.max(5, (gene.expression / 5.0) * 100))}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default TrajectoryScrubber;
