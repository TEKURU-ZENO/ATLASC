"use client";
import React, { useState, useEffect, useRef, useCallback } from "react";

const API_BASE = "http://localhost:8000";

// ─── Types ────────────────────────────────────────────────────────────────────
interface SimResult {
  simulation_id: string;
  status: string;
  steps: number;
  uncertainty_mean?: number;
  mean_expression_summary?: {
    top_affected_genes: { gene_idx: number; abs_expression: number }[];
    total_genes: number;
  };
  elapsed_s?: number;
}
interface RecommendResult {
  selected_genes: number[];
  explanations: Record<string, { rank: number; Total_Score: number; Utraj_Norm: number }>;
}
interface GraphData {
  nodes: { id: number }[];
  edges: { source: number; target: number }[];
  total_edges: number;
}
interface ModelStatus {
  status: string;
  ensemble_members?: number;
  num_genes?: number;
  device?: string;
  checkpoints?: { file: string; size_kb: number }[];
  error?: string;
}
type Panel = "home" | "simulate" | "explain" | "recommend" | "graph";
type ReqState = "idle" | "loading" | "ok" | "error";

// ─── Utility components ───────────────────────────────────────────────────────
function Badge({ label, color }: { label: string; color: string }) {
  return <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${color}`}>{label}</span>;
}
function StatusIndicator({ state }: { state: ReqState }) {
  if (state === "loading") return <span className="inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />;
  if (state === "ok") return <span className="text-emerald-400">✓</span>;
  if (state === "error") return <span className="text-red-400">✗</span>;
  return null;
}
function FormGroup({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-sm font-medium text-neutral-300 mb-2">{label}</label>
      {children}
    </div>
  );
}
function ActionButton({ id, label, color, state, onClick }: {
  id: string; label: string; color: string; state: ReqState; onClick: () => void;
}) {
  return (
    <button id={id} onClick={onClick} disabled={state === "loading"}
      className={`w-full py-3 rounded-xl font-semibold text-sm bg-${color}-600 hover:bg-${color}-500 disabled:opacity-50 disabled:cursor-not-allowed text-white transition-all flex items-center justify-center gap-2`}>
      <StatusIndicator state={state} />
      {state === "loading" ? "Processing…" : label}
    </button>
  );
}
function ResultBox({ children }: { children: React.ReactNode }) {
  return <div className="rounded-xl border border-neutral-700 bg-neutral-800/50 p-5 text-sm">{children}</div>;
}
function ErrorBox({ message }: { message: string }) {
  return (
    <div className="rounded-xl border border-red-900 bg-red-950/40 p-4 text-sm">
      <p className="text-red-400 font-semibold">❌ {message}</p>
      <p className="text-red-300/70 text-xs mt-1">Make sure the backend is running: <code className="bg-red-950 px-1 rounded">python -m uvicorn main:app --reload --port 8000</code> from <code className="bg-red-950 px-1 rounded">apps/api/</code></p>
    </div>
  );
}
function KVRow({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between text-sm py-1 border-b border-neutral-700/50 last:border-0">
      <span className="text-neutral-400">{k}</span>
      <span className="text-white font-mono">{v}</span>
    </div>
  );
}
function PanelWrapper({ icon, title, color, description, children }: {
  icon: string; title: string; color: string; description: string; children: React.ReactNode;
}) {
  return (
    <div className="max-w-2xl mx-auto">
      <div className="flex items-center gap-3 mb-2">
        <span className="text-3xl">{icon}</span>
        <h2 className={`text-2xl font-bold text-${color}-300`}>{title}</h2>
      </div>
      <p className="text-neutral-400 text-sm mb-8 leading-relaxed">{description}</p>
      <div className="space-y-6">{children}</div>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function Home() {
  const [panel, setPanel] = useState<Panel>("home");
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);

  // Health check once on mount
  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(r => r.ok ? setApiOnline(true) : setApiOnline(false))
      .catch(() => setApiOnline(false));
  }, []);

  const navItems: [Panel, string, string][] = [
    ["simulate", "⚡ Simulate", "blue"],
    ["explain", "🔍 Explain", "emerald"],
    ["recommend", "🧬 Recommend", "purple"],
    ["graph", "🕸 GRN Graph", "cyan"],
  ];

  return (
    <div className="min-h-screen bg-[#080b14] text-neutral-100 font-sans flex flex-col">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-neutral-800/60 bg-[#080b14]/90 sticky top-0 z-50 backdrop-blur">
        <button onClick={() => setPanel("home")} className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-emerald-500 flex items-center justify-center font-bold text-sm">A</div>
          <span className="text-lg font-bold tracking-tight">ATLAS</span>
        </button>

        <nav className="flex gap-1 flex-wrap justify-end">
          {navItems.map(([id, label, color]) => (
            <button key={id} onClick={() => setPanel(id)}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${panel === id ? `bg-${color}-700 text-white` : "text-neutral-400 hover:text-white hover:bg-neutral-800"
                }`}>{label}</button>
          ))}
        </nav>

        {/* API status pill */}
        <div className={`hidden sm:flex items-center gap-1.5 text-xs px-3 py-1 rounded-full border ${apiOnline === null ? "border-neutral-700 text-neutral-500" :
            apiOnline ? "border-emerald-800 bg-emerald-950/40 text-emerald-400" :
              "border-red-800 bg-red-950/40 text-red-400"}`}>
          <span className={`w-1.5 h-1.5 rounded-full ${apiOnline === null ? "bg-neutral-500" : apiOnline ? "bg-emerald-400 animate-pulse" : "bg-red-400"}`} />
          {apiOnline === null ? "Checking API…" : apiOnline ? "API Online" : "API Offline"}
        </div>
      </header>

      {/* Body */}
      <main className="flex-1 px-6 py-10 max-w-5xl mx-auto w-full">
        {panel === "home" && <HomePanel setPanel={setPanel} apiOnline={apiOnline} />}
        {panel === "simulate" && <SimulatePanel />}
        {panel === "explain" && <ExplainPanel />}
        {panel === "recommend" && <RecommendPanel />}
        {panel === "graph" && <GraphPanel />}
      </main>

      <footer className="py-4 text-center text-neutral-600 text-xs border-t border-neutral-800/60">
        ATLAS · Causal GNN Ensemble ·{" "}
        <a href={`${API_BASE}/docs`} target="_blank" rel="noreferrer" className="text-blue-500 hover:underline">API Docs ↗</a>
        {" · "}
        <a href={`${API_BASE}/model_status`} target="_blank" rel="noreferrer" className="text-blue-500 hover:underline">Model Status ↗</a>
      </footer>
    </div>
  );
}

// ─── Home Panel ───────────────────────────────────────────────────────────────
function HomePanel({ setPanel, apiOnline }: { setPanel: (p: Panel) => void; apiOnline: boolean | null }) {
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);

  useEffect(() => {
    if (apiOnline) {
      fetch(`${API_BASE}/model_status`)
        .then(r => r.json())
        .then(d => setModelStatus(d))
        .catch(() => setModelStatus(null));
    }
  }, [apiOnline]);

  const cards = [
    { id: "simulate" as Panel, icon: "⚡", title: "Simulate Interventions", desc: "Silence one or more genes and propagate the causal effect forward in time using the trained Causal GNN ensemble.", action: "Run Simulation", color: "blue", badge: "Core Feature", badgeColor: "bg-blue-900/60 text-blue-300" },
    { id: "explain" as Panel, icon: "🔍", title: "Explainability & Attribution", desc: "Inspect causal pathways and epistemic uncertainty. Understand why the model predicts a particular gene expression outcome.", action: "Explore Attribution", color: "emerald", badge: "XAI Layer", badgeColor: "bg-emerald-900/60 text-emerald-300" },
    { id: "recommend" as Panel, icon: "🧬", title: "Active Learning Planner", desc: "Discover the highest-value unobserved perturbations using uncertainty-driven acquisition scoring.", action: "Get Recommendations", color: "purple", badge: "AL Engine", badgeColor: "bg-purple-900/60 text-purple-300" },
    { id: "graph" as Panel, icon: "🕸", title: "GRN Graph Explorer", desc: "Visualize the TRRUST Gene Regulatory Network topology as a force-directed graph. Inspect hub genes and their connectivity.", action: "Explore Graph", color: "cyan", badge: "Structural Layer", badgeColor: "bg-cyan-900/60 text-cyan-300" },
  ];

  return (
    <div>
      <div className="text-center mb-12 mt-4">
        <div className={`inline-flex items-center gap-2 text-xs px-3 py-1 rounded-full mb-6 border ${apiOnline ? "bg-blue-950/60 text-blue-300 border-blue-900/60" : "bg-neutral-900 text-neutral-400 border-neutral-700"}`}>
          <span className={`w-2 h-2 rounded-full ${apiOnline ? "bg-emerald-400 animate-pulse" : "bg-neutral-500"}`} />
          {apiOnline ? "System Online · Ensemble K=3 Loaded" : apiOnline === false ? "Backend Offline — start uvicorn to connect" : "Checking backend status…"}
        </div>
        <h1 className="text-5xl font-extrabold tracking-tight mb-4">
          <span className="bg-gradient-to-r from-blue-400 via-cyan-300 to-emerald-400 bg-clip-text text-transparent">ATLAS</span>
        </h1>
        <p className="text-neutral-400 text-lg max-w-xl mx-auto">A scalable causal AI simulation platform for gene perturbation prediction with calibrated epistemic uncertainty.</p>
      </div>

      {/* Model Status Widget */}
      {modelStatus && modelStatus.status === "loaded" && (
        <div className="mb-8 rounded-2xl border border-emerald-900/40 bg-emerald-950/20 p-4">
          <p className="text-xs text-emerald-400 font-semibold uppercase tracking-widest mb-3">✓ Model Status</p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              ["Ensemble Members", `K = ${modelStatus.ensemble_members}`],
              ["Gene Vocabulary", `${modelStatus.num_genes?.toLocaleString()} genes`],
              ["Device", modelStatus.device ?? "—"],
              ["Checkpoints", `${modelStatus.checkpoints?.length ?? 0} loaded`],
            ].map(([label, value]) => (
              <div key={label} className="bg-neutral-900/60 rounded-xl px-3 py-2 text-xs">
                <span className="text-neutral-500 block">{label}</span>
                <span className="text-emerald-300 font-bold font-mono">{value}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
        {cards.map((c) => (
          <div key={c.id} className="relative group rounded-2xl border border-neutral-800 bg-neutral-900/60 p-5 flex flex-col gap-4 hover:border-neutral-700 hover:-translate-y-1 transition-all duration-200">
            <div className="text-3xl">{c.icon}</div>
            <div>
              <div className="flex items-center gap-2 mb-1 flex-wrap">
                <h2 className="text-base font-bold">{c.title}</h2>
                <Badge label={c.badge} color={c.badgeColor} />
              </div>
              <p className="text-xs text-neutral-400 leading-relaxed">{c.desc}</p>
            </div>
            <button onClick={() => setPanel(c.id)} id={`btn-${c.id}`}
              className={`mt-auto w-full py-2 rounded-xl font-semibold text-xs bg-${c.color}-600 hover:bg-${c.color}-500 text-white transition-colors`}>
              {c.action}
            </button>
          </div>
        ))}
      </div>

      <div className="mt-12 rounded-2xl border border-neutral-800 bg-neutral-900/40 p-6">
        <h3 className="font-bold text-base mb-4 text-neutral-200">System Architecture — 7 Layers</h3>
        <div className="flex flex-wrap gap-2">
          {[["1", "Data Ingestion", "blue"], ["2", "Representation", "cyan"], ["3", "Causal Modeling", "emerald"], ["4", "Dynamics (GNN)", "teal"], ["5", "Trust & Uncertainty", "yellow"], ["6", "Decision Support", "orange"], ["7", "Application API", "purple"]].map(([n, label, c]) => (
            <div key={n} className={`flex items-center gap-2 bg-neutral-800 rounded-xl px-3 py-1.5 text-sm border border-${c}-900/50`}>
              <span className={`text-xs font-bold text-${c}-400 bg-${c}-900/40 w-5 h-5 rounded-full flex items-center justify-center`}>{n}</span>
              <span className="text-neutral-300 text-xs">{label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Simulate Panel (FIXED: correct request format) ──────────────────────────
function SimulatePanel() {
  const [genes, setGenes] = useState("0,1,5");
  const [steps, setSteps] = useState(3);
  const [state, setState] = useState<ReqState>("idle");
  const [result, setResult] = useState<SimResult | null>(null);
  const [error, setError] = useState("");

  const run = async () => {
    setState("loading"); setError("");
    try {
      // Build perturbation specs: "gene_idx:0" for knock-out
      const perturbations = genes
        .split(",")
        .map(g => g.trim())
        .filter(g => g !== "")
        .map(g => `${parseInt(g, 10)}:0`);

      const res = await fetch(`${API_BASE}/simulate/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ perturbations, steps, dataset_id: "norman" }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
        throw new Error(err.detail ?? `HTTP ${res.status}`);
      }
      setResult(await res.json()); setState("ok");
    } catch (e: unknown) { setError((e as Error).message); setState("error"); }
  };

  return (
    <PanelWrapper icon="⚡" title="Simulate Gene Interventions" color="blue"
      description="Silence (knock-out) one or more genes and propagate the causal effect forward in time using the trained Causal GNN ensemble.">
      <FormGroup label="Gene Indices to Knock Out (comma-separated)">
        <input id="input-gene-indices" value={genes} onChange={e => setGenes(e.target.value)} placeholder="e.g. 0, 5, 12, 99"
          className="w-full bg-neutral-800 border border-neutral-700 rounded-xl px-4 py-2 text-sm outline-none focus:border-blue-500 transition-colors" />
        <p className="text-xs text-neutral-500 mt-1">Integer gene indices — each will be perturbed to 0 (knock-out). Sent as <code className="bg-neutral-700 px-1 rounded">GENE_IDX:0</code> to the API.</p>
      </FormGroup>
      <FormGroup label={`Rollout Steps: ${steps}`}>
        <input id="input-steps" type="range" min={1} max={10} value={steps} onChange={e => setSteps(Number(e.target.value))} className="w-full accent-blue-500" />
        <div className="flex justify-between text-xs text-neutral-500"><span>1 step</span><span>10 steps</span></div>
      </FormGroup>
      <ActionButton id="btn-run-simulation" label="Run Simulation" color="blue" state={state} onClick={run} />
      {state === "error" && <ErrorBox message={error} />}
      {state === "ok" && result && (
        <ResultBox>
          <p className="text-sm text-emerald-400 font-semibold mb-3">✓ Simulation Complete</p>
          <KVRow k="Simulation ID" v={result.simulation_id} />
          <KVRow k="Status" v={result.status} />
          <KVRow k="Rollout Steps" v={`${result.steps}`} />
          {result.uncertainty_mean !== undefined && (
            <KVRow k="Mean Epistemic Uncertainty" v={result.uncertainty_mean.toFixed(4)} />
          )}
          {result.elapsed_s !== undefined && (
            <KVRow k="Elapsed" v={`${result.elapsed_s}s`} />
          )}
          {result.mean_expression_summary?.top_affected_genes && (
            <div className="mt-4">
              <p className="text-xs text-neutral-500 font-semibold uppercase tracking-widest mb-2">
                Top Affected Genes (by absolute expression)
              </p>
              <div className="flex flex-wrap gap-1.5">
                {result.mean_expression_summary.top_affected_genes.slice(0, 10).map(g => (
                  <span key={g.gene_idx}
                    className="bg-blue-900/40 border border-blue-800 text-blue-200 text-xs px-2 py-0.5 rounded-full font-mono">
                    Gene {g.gene_idx} ({g.abs_expression.toFixed(3)})
                  </span>
                ))}
              </div>
            </div>
          )}
        </ResultBox>
      )}
    </PanelWrapper>
  );
}

// ─── Explain Panel (FIXED: requires simulation_id + target_gene) ──────────────
function ExplainPanel() {
  const [simId, setSimId] = useState("");
  const [targetGene, setTargetGene] = useState("TP53");
  const [state, setState] = useState<ReqState>("idle");
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState("");

  const run = async () => {
    setState("loading"); setError("");
    if (!simId.trim()) {
      setError("Please enter a Simulation ID. Run a simulation first to get one."); setState("error"); return;
    }
    try {
      const res = await fetch(`${API_BASE}/explain/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ simulation_id: simId.trim(), target_gene: targetGene.trim() }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
        throw new Error(err.detail ?? `HTTP ${res.status}`);
      }
      setResult(await res.json()); setState("ok");
    } catch (e: unknown) { setError((e as Error).message); setState("error"); }
  };

  return (
    <PanelWrapper icon="🔍" title="Explainability & Causal Attribution" color="emerald"
      description="Inspect how the model attributes its predictions. Enter a Simulation ID from a completed simulation, then provide a target gene name to get attribution scores.">
      <FormGroup label="Simulation ID (from a completed simulation)">
        <input id="input-sim-id" value={simId} onChange={e => setSimId(e.target.value)}
          placeholder="e.g. a3f2c1b0"
          className="w-full bg-neutral-800 border border-neutral-700 rounded-xl px-4 py-2 text-sm outline-none focus:border-emerald-500 transition-colors" />
        <p className="text-xs text-neutral-500 mt-1">Run a simulation first — the Simulation ID is shown in the result.</p>
      </FormGroup>
      <FormGroup label="Target Gene Name">
        <input id="input-target-gene" value={targetGene} onChange={e => setTargetGene(e.target.value)}
          placeholder="e.g. TP53, MYC, BRCA1"
          className="w-full bg-neutral-800 border border-neutral-700 rounded-xl px-4 py-2 text-sm outline-none focus:border-emerald-500 transition-colors" />
      </FormGroup>
      <ActionButton id="btn-explain" label="Explain Attribution" color="emerald" state={state} onClick={run} />
      {state === "error" && <ErrorBox message={error} />}
      {state === "ok" && result && (
        <ResultBox>
          <p className="text-sm text-emerald-400 font-semibold mb-3">✓ Attribution computed</p>
          <pre className="text-xs text-neutral-300 overflow-auto whitespace-pre-wrap max-h-64">{JSON.stringify(result, null, 2)}</pre>
        </ResultBox>
      )}
    </PanelWrapper>
  );
}

// ─── Recommend Panel ──────────────────────────────────────────────────────────
function RecommendPanel() {
  const [candidates, setCandidates] = useState("10,20,30,40,50");
  const [k, setK] = useState(3);
  const [horizon, setHorizon] = useState(3);
  const [state, setState] = useState<ReqState>("idle");
  const [result, setResult] = useState<RecommendResult | null>(null);
  const [error, setError] = useState("");

  const run = async () => {
    setState("loading"); setError("");
    try {
      const res = await fetch(`${API_BASE}/recommend/rank_perturbations`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ candidate_genes: candidates.split(",").map(g => parseInt(g.trim(), 10)), k_to_select: k, rollout_horizon: horizon }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
        throw new Error(err.detail ?? `HTTP ${res.status}`);
      }
      setResult(await res.json()); setState("ok");
    } catch (e: unknown) { setError((e as Error).message); setState("error"); }
  };

  return (
    <PanelWrapper icon="🧬" title="Active Learning Planner" color="purple"
      description="Rank candidate gene perturbations by epistemic value. Combines uncertainty, structural hub importance, and diversity.">
      <FormGroup label="Candidate Gene Indices (comma-separated)">
        <input id="input-candidate-genes" value={candidates} onChange={e => setCandidates(e.target.value)} placeholder="e.g. 10, 25, 100"
          className="w-full bg-neutral-800 border border-neutral-700 rounded-xl px-4 py-2 text-sm outline-none focus:border-purple-500 transition-colors" />
      </FormGroup>
      <div className="grid grid-cols-2 gap-4">
        <FormGroup label={`Top-K Selections: ${k}`}>
          <input id="input-k" type="range" min={1} max={10} value={k} onChange={e => setK(Number(e.target.value))} className="w-full accent-purple-500" />
        </FormGroup>
        <FormGroup label={`Rollout Horizon: ${horizon}`}>
          <input id="input-horizon" type="range" min={1} max={10} value={horizon} onChange={e => setHorizon(Number(e.target.value))} className="w-full accent-purple-500" />
        </FormGroup>
      </div>
      <ActionButton id="btn-recommend" label="Get Recommendations" color="purple" state={state} onClick={run} />
      {state === "error" && <ErrorBox message={error} />}
      {state === "ok" && result && (
        <ResultBox>
          <p className="text-sm text-purple-400 font-semibold mb-3">✓ Top-{result.selected_genes.length} genes selected</p>
          <div className="flex flex-wrap gap-2 mb-4">
            {result.selected_genes.map(g => (
              <span key={g} className="bg-purple-900/50 border border-purple-800 text-purple-200 text-xs px-3 py-1 rounded-full font-mono">Gene {g}</span>
            ))}
          </div>
          {Object.entries(result.explanations).length > 0 && (
            <div className="space-y-2">
              <p className="text-xs text-neutral-500 font-semibold uppercase tracking-widest">Acquisition Scores</p>
              {Object.entries(result.explanations).map(([gene, exp]) => (
                <div key={gene} className="flex items-center justify-between bg-neutral-800 rounded-lg px-3 py-2 text-xs">
                  <span className="text-neutral-300 font-mono">Gene {gene}</span>
                  <div className="flex gap-4">
                    <span className="text-purple-300">Score: {exp.Total_Score?.toFixed(4) ?? "—"}</span>
                    <span className="text-blue-300">U_traj: {exp.Utraj_Norm?.toFixed(4) ?? "—"}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </ResultBox>
      )}
    </PanelWrapper>
  );
}

// ─── Graph Panel (Force-Directed SVG) ────────────────────────────────────────
interface NodePos { id: number; x: number; y: number; vx: number; vy: number; }

function GraphPanel() {
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [loadState, setLoadState] = useState<ReqState>("idle");
  const [error, setError] = useState("");
  const [nodes, setNodes] = useState<NodePos[]>([]);
  const [highlight, setHighlight] = useState<number | null>(null);
  const animRef = useRef<number>(0);
  const nodesRef = useRef<NodePos[]>([]);
  const W = 760, H = 500;

  const loadGraph = useCallback(async () => {
    setLoadState("loading"); setError("");
    try {
      const res = await fetch(`${API_BASE}/graph`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: GraphData = await res.json();
      setGraphData(data);

      // Initialize node positions randomly
      const nodeMap = new Map<number, NodePos>();
      data.nodes.forEach(n => {
        nodeMap.set(n.id, { id: n.id, x: Math.random() * W, y: Math.random() * H, vx: 0, vy: 0 });
      });
      const initial = Array.from(nodeMap.values());
      nodesRef.current = initial;
      setNodes([...initial]);
      setLoadState("ok");
    } catch (e: unknown) { setError((e as Error).message); setLoadState("error"); }
  }, []);

  // Simple force simulation
  useEffect(() => {
    if (!graphData || nodesRef.current.length === 0) return;

    const edges = graphData.edges;

    const tick = () => {
      const ns = nodesRef.current;
      const idxMap = new Map(ns.map((n, i) => [n.id, i]));
      const next = ns.map(n => ({ ...n }));
      const k = 20, repel = 800;

      // Repulsion
      for (let i = 0; i < next.length; i++) {
        for (let j = i + 1; j < next.length; j++) {
          const dx = next[j].x - next[i].x || 0.01;
          const dy = next[j].y - next[i].y || 0.01;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const force = repel / (dist * dist);
          next[i].vx -= (dx / dist) * force;
          next[i].vy -= (dy / dist) * force;
          next[j].vx += (dx / dist) * force;
          next[j].vy += (dy / dist) * force;
        }
      }

      // Attraction (spring)
      edges.forEach(e => {
        const si = idxMap.get(e.source), ti = idxMap.get(e.target);
        if (si === undefined || ti === undefined) return;
        const dx = next[ti].x - next[si].x;
        const dy = next[ti].y - next[si].y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const force = (dist - k) * 0.05;
        next[si].vx += (dx / dist) * force;
        next[si].vy += (dy / dist) * force;
        next[ti].vx -= (dx / dist) * force;
        next[ti].vy -= (dy / dist) * force;
      });

      // Center gravity
      next.forEach(n => {
        n.vx += (W / 2 - n.x) * 0.003;
        n.vy += (H / 2 - n.y) * 0.003;
        n.vx *= 0.85; n.vy *= 0.85; // damping
        n.x = Math.max(8, Math.min(W - 8, n.x + n.vx));
        n.y = Math.max(8, Math.min(H - 8, n.y + n.vy));
      });

      nodesRef.current = next;
      setNodes([...next]);
      animRef.current = requestAnimationFrame(tick);
    };

    animRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animRef.current);
  }, [graphData]);

  const idxMap = new Map(nodes.map((n, i) => [n.id, i]));

  // Degree count for node sizing
  const degree = new Map<number, number>();
  graphData?.edges.forEach(e => {
    degree.set(e.source, (degree.get(e.source) ?? 0) + 1);
    degree.set(e.target, (degree.get(e.target) ?? 0) + 1);
  });

  const highlightNeighbors = highlight !== null
    ? new Set(graphData?.edges.filter(e => e.source === highlight || e.target === highlight).flatMap(e => [e.source, e.target]))
    : null;

  return (
    <div className="max-w-4xl mx-auto">
      <div className="flex items-center gap-3 mb-2">
        <span className="text-3xl">🕸</span>
        <h2 className="text-2xl font-bold text-cyan-300">Gene Regulatory Network (GRN) Explorer</h2>
      </div>
      <p className="text-neutral-400 text-sm mb-6 leading-relaxed">
        Visualize the TRRUST biological GRN as a force-directed graph. Each <strong className="text-cyan-300">node</strong> is a gene; each <strong className="text-cyan-300">edge</strong> is a regulatory interaction. Larger nodes = higher connectivity (hub genes). Hover to highlight neighbours.
      </p>

      {loadState === "idle" && (
        <button id="btn-load-graph" onClick={loadGraph}
          className="w-full py-3 rounded-xl font-semibold text-sm bg-cyan-600 hover:bg-cyan-500 text-white transition-all flex items-center justify-center gap-2">
          🕸 Load GRN Graph (first 500 edges)
        </button>
      )}
      {loadState === "loading" && (
        <div className="flex items-center justify-center gap-3 py-20 text-cyan-400">
          <span className="w-6 h-6 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
          Building force layout…
        </div>
      )}
      {loadState === "error" && <ErrorBox message={error} />}
      {loadState === "ok" && graphData && (
        <div>
          {/* Stats bar */}
          <div className="flex gap-4 mb-4 flex-wrap">
            {[["Nodes", `${graphData.nodes.length}`], ["Edges shown", `${graphData.edges.length}`], ["Total edges", `${graphData.total_edges}`]].map(([k, v]) => (
              <div key={k} className="bg-neutral-800 rounded-xl px-4 py-2 text-sm flex flex-col">
                <span className="text-neutral-500 text-xs">{k}</span>
                <span className="text-cyan-300 font-bold font-mono">{v}</span>
              </div>
            ))}
            <div className="text-xs text-neutral-500 self-center ml-auto">Hover a node to highlight neighbours</div>
          </div>

          {/* SVG canvas */}
          <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 overflow-hidden">
            <svg width="100%" viewBox={`0 0 ${W} ${H}`} className="block" style={{ background: "none" }}>
              {/* Edges */}
              {graphData.edges.map((e, i) => {
                const si = idxMap.get(e.source), ti = idxMap.get(e.target);
                if (si === undefined || ti === undefined) return null;
                const s = nodes[si], t = nodes[ti];
                const isHighlighted = highlight !== null && (e.source === highlight || e.target === highlight);
                return (
                  <line key={i} x1={s.x} y1={s.y} x2={t.x} y2={t.y}
                    stroke={isHighlighted ? "#22d3ee" : "#334155"}
                    strokeWidth={isHighlighted ? 1.5 : 0.5}
                    strokeOpacity={highlight !== null ? (isHighlighted ? 0.9 : 0.1) : 0.4}
                  />
                );
              })}
              {/* Nodes */}
              {nodes.map(n => {
                const deg = degree.get(n.id) ?? 0;
                const r = Math.min(3 + deg * 0.6, 10);
                const isHighlighted = highlightNeighbors?.has(n.id);
                const isFocus = n.id === highlight;
                return (
                  <circle key={n.id} cx={n.x} cy={n.y} r={isFocus ? r + 3 : r}
                    fill={isFocus ? "#22d3ee" : isHighlighted ? "#60a5fa" : "#1e40af"}
                    stroke={isFocus ? "#e0f2fe" : isHighlighted ? "#93c5fd" : "#3b82f6"}
                    strokeWidth={isFocus ? 2 : 0.5}
                    opacity={highlight !== null && !isHighlighted && !isFocus ? 0.3 : 1}
                    className="cursor-pointer transition-all"
                    onMouseEnter={() => setHighlight(n.id)}
                    onMouseLeave={() => setHighlight(null)}
                  >
                    <title>Gene {n.id} | Degree: {deg}</title>
                  </circle>
                );
              })}
            </svg>
          </div>

          {/* Legend */}
          <div className="mt-3 flex gap-4 text-xs text-neutral-500 flex-wrap">
            <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full bg-blue-700 border border-blue-500" />Low connectivity</span>
            <span className="flex items-center gap-1.5"><span className="w-4 h-4 rounded-full bg-blue-700 border border-blue-500" />Hub gene (high degree)</span>
            <span className="flex items-center gap-1.5"><span className="w-4 h-0.5 bg-slate-600" />Regulatory edge</span>
          </div>

          <button onClick={loadGraph} className="mt-4 text-xs text-neutral-500 hover:text-neutral-300 underline">Reload with new random layout</button>
        </div>
      )}
    </div>
  );
}
