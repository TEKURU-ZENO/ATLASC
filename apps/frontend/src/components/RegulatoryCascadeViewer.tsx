"use client";
import React, { useState } from "react";

export interface CascadeNode {
  id: string;
  name: string;
  role: "knockout" | "repressed" | "activated" | "unaffected";
  foldChange: number;
  x: number;
  y: number;
  type: "TF" | "Effector" | "Kinase";
}

export interface CascadeEdge {
  source: string;
  target: string;
  type: "activation" | "repression";
  weight: number;
}

const DEFAULT_NODES: CascadeNode[] = [
  { id: "TP53", name: "TP53", role: "knockout", foldChange: -2.15, x: 250, y: 50, type: "TF" },
  { id: "CDKN1A", name: "CDKN1A", role: "repressed", foldChange: -1.20, x: 120, y: 150, type: "Effector" },
  { id: "MDM2", name: "MDM2", role: "repressed", foldChange: -1.50, x: 250, y: 160, type: "Effector" },
  { id: "BAX", name: "BAX", role: "repressed", foldChange: -0.66, x: 380, y: 150, type: "Effector" },
  { id: "MYC", name: "MYC", role: "activated", foldChange: +1.25, x: 180, y: 260, type: "TF" },
  { id: "BCL2", name: "BCL2", role: "activated", foldChange: +0.85, x: 340, y: 260, type: "Effector" },
  { id: "CCND1", name: "CCND1", role: "activated", foldChange: +1.40, x: 260, y: 350, type: "Effector" },
];

const DEFAULT_EDGES: CascadeEdge[] = [
  { source: "TP53", target: "CDKN1A", type: "activation", weight: 1.0 },
  { source: "TP53", target: "MDM2", type: "activation", weight: 0.95 },
  { source: "TP53", target: "BAX", type: "activation", weight: 0.88 },
  { source: "MDM2", target: "TP53", type: "repression", weight: -0.9 },
  { source: "CDKN1A", target: "MYC", type: "repression", weight: -0.85 },
  { source: "BAX", target: "BCL2", type: "repression", weight: -0.75 },
  { source: "MYC", target: "CCND1", type: "activation", weight: 0.92 },
];

export const RegulatoryCascadeViewer: React.FC = () => {
  const [nodes, setNodes] = useState<CascadeNode[]>(DEFAULT_NODES);
  const [edges, setEdges] = useState<CascadeEdge[]>(DEFAULT_EDGES);
  const [selectedNode, setSelectedNode] = useState<CascadeNode | null>(nodes[0]);
  const [filterMode, setFilterMode] = useState<"all" | "active">("all");

  const getNodeColor = (role: CascadeNode["role"]) => {
    switch (role) {
      case "knockout":
        return { fill: "#dc2626", stroke: "#ef4444", text: "Knockout" };
      case "repressed":
        return { fill: "#e11d48", stroke: "#f43f5e", text: "Down-regulated" };
      case "activated":
        return { fill: "#059669", stroke: "#10b981", text: "Up-regulated" };
      default:
        return { fill: "#475569", stroke: "#64748b", text: "Basal" };
    }
  };

  const handleNodeClick = (node: CascadeNode) => {
    setSelectedNode(node);
  };

  const filteredEdges = edges.filter((e) => {
    if (filterMode === "all") return true;
    if (!selectedNode) return true;
    return e.source === selectedNode.id || e.target === selectedNode.id;
  });

  return (
    <div className="rounded-2xl border border-neutral-700 bg-neutral-900/90 p-6 shadow-2xl space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-neutral-800 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-indigo-400" />
            <h3 className="text-lg font-bold text-white tracking-wide">Biological Regulatory Cascade</h3>
          </div>
          <p className="text-xs text-neutral-400 mt-0.5">
            TRRUST Directed & Signed Multi-Hop Intervention Flow
          </p>
        </div>

        {/* Filters */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setFilterMode("all")}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition ${
              filterMode === "all"
                ? "bg-indigo-600 text-white"
                : "bg-neutral-800 text-neutral-400 hover:text-white"
            }`}
          >
            Full Cascade
          </button>
          <button
            onClick={() => setFilterMode("active")}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition ${
              filterMode === "active"
                ? "bg-indigo-600 text-white"
                : "bg-neutral-800 text-neutral-400 hover:text-white"
            }`}
          >
            Direct Paths
          </button>
        </div>
      </div>

      {/* SVG Canvas */}
      <div className="relative bg-neutral-950 rounded-xl border border-neutral-800 overflow-hidden flex justify-center py-4">
        <svg viewBox="0 0 500 420" className="w-full max-w-[500px] h-[360px]">
          <defs>
            <marker
              id="arrow-act"
              markerWidth="8"
              markerHeight="8"
              refX="18"
              refY="4"
              orient="auto"
            >
              <polygon points="0 0, 8 4, 0 8" fill="#10b981" />
            </marker>
            <marker
              id="arrow-rep"
              markerWidth="8"
              markerHeight="8"
              refX="18"
              refY="4"
              orient="auto"
            >
              <line x1="8" y1="0" x2="8" y2="8" stroke="#ef4444" strokeWidth="2.5" />
            </marker>
          </defs>

          {/* Render Edges */}
          {filteredEdges.map((e, idx) => {
            const sNode = nodes.find((n) => n.id === e.source);
            const tNode = nodes.find((n) => n.id === e.target);
            if (!sNode || !tNode) return null;

            const isRep = e.type === "repression";
            const strokeColor = isRep ? "#ef4444" : "#10b981";
            const markerUrl = isRep ? "url(#arrow-rep)" : "url(#arrow-act)";

            return (
              <line
                key={`edge-${idx}`}
                x1={sNode.x}
                y1={sNode.y}
                x2={tNode.x}
                y2={tNode.y}
                stroke={strokeColor}
                strokeWidth={Math.abs(e.weight) * 2}
                strokeDasharray={isRep ? "4,3" : "none"}
                markerEnd={markerUrl}
                opacity={0.8}
              />
            );
          })}

          {/* Render Nodes */}
          {nodes.map((node) => {
            const { fill, stroke } = getNodeColor(node.role);
            const isSelected = selectedNode?.id === node.id;

            return (
              <g
                key={node.id}
                transform={`translate(${node.x}, ${node.y})`}
                onClick={() => handleNodeClick(node)}
                className="cursor-pointer transition-transform hover:scale-110"
              >
                <circle
                  r={isSelected ? 22 : 18}
                  fill={fill}
                  stroke={isSelected ? "#ffffff" : stroke}
                  strokeWidth={isSelected ? 3 : 2}
                  className="transition-all duration-200"
                />
                <text
                  textAnchor="middle"
                  dy=".3em"
                  fill="#ffffff"
                  fontSize={10}
                  fontWeight="bold"
                  className="pointer-events-none select-none font-mono"
                >
                  {node.name}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Legend */}
        <div className="absolute bottom-3 left-3 bg-neutral-900/90 border border-neutral-800 rounded-lg p-2 text-[10px] space-y-1 font-mono text-neutral-300">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-red-600" />
            <span>Knockout Target</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
            <span>Up-regulated (+)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-rose-500" />
            <span>Down-regulated (−)</span>
          </div>
        </div>
      </div>

      {/* Selected Node Details Card */}
      {selectedNode && (
        <div className="bg-neutral-800/60 rounded-xl p-4 border border-neutral-700/60 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3">
          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono font-bold text-white text-base">{selectedNode.name}</span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-neutral-700 text-neutral-300">
                {selectedNode.type}
              </span>
              <span
                className={`text-[10px] font-mono px-2 py-0.5 rounded-full ${
                  selectedNode.role === "knockout"
                    ? "bg-red-500/20 text-red-300 border border-red-500/40"
                    : selectedNode.role === "activated"
                    ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                    : "bg-rose-500/20 text-rose-300 border border-rose-500/40"
                }`}
              >
                {getNodeColor(selectedNode.role).text}
              </span>
            </div>
            <p className="text-xs text-neutral-400 mt-1">
              Cascade Delta Fold Change:{" "}
              <span className="font-mono font-semibold text-white">
                {selectedNode.foldChange > 0 ? "+" : ""}
                {selectedNode.foldChange.toFixed(2)}
              </span>
            </p>
          </div>

          <div className="flex gap-2 text-xs">
            <span className="text-neutral-400">
              Regulators:{" "}
              <span className="text-white font-mono font-semibold">
                {edges.filter((e) => e.target === selectedNode.id).length}
              </span>
            </span>
            <span className="text-neutral-500">•</span>
            <span className="text-neutral-400">
              Downstream Targets:{" "}
              <span className="text-white font-mono font-semibold">
                {edges.filter((e) => e.source === selectedNode.id).length}
              </span>
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default RegulatoryCascadeViewer;
