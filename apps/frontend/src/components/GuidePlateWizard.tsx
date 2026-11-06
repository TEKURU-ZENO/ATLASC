"use client";
import React, { useState } from "react";

interface GuidePlateWizardProps {
  selectedGenes?: number[];
  apiBase?: string;
}

export const GuidePlateWizard: React.FC<GuidePlateWizardProps> = ({
  selectedGenes = [12, 45, 89, 102, 143, 210, 312],
  apiBase = "http://localhost:8000",
}) => {
  const [formatType, setFormatType] = useState<"96" | "384">("96");
  const [replicates, setReplicates] = useState<number>(2);
  const [isExporting, setIsExporting] = useState<boolean>(false);
  const [downloadSuccess, setDownloadSuccess] = useState<boolean>(false);

  const rows = formatType === "96" ? 8 : 16;
  const cols = formatType === "96" ? 12 : 24;
  const totalWells = rows * cols;
  const controlWells = formatType === "96" ? 8 : 16;
  const blankWells = 4;

  const handleExportCSV = async () => {
    setIsExporting(true);
    setDownloadSuccess(false);
    try {
      const res = await fetch(`${apiBase}/recommend/export_plate_csv`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          selected_genes: selectedGenes,
          format_type: formatType,
          control_wells: controlWells,
          blank_wells: blankWells,
          replicates: replicates,
        }),
      });

      if (!res.ok) throw new Error("Failed to export plate CSV");

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `atlas_plate_layout_${formatType}well.csv`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
      setDownloadSuccess(true);
    } catch (err) {
      console.error(err);
      alert("Failed to export plate layout. Make sure the API backend is running.");
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="rounded-2xl border border-neutral-700 bg-neutral-900/90 p-6 shadow-2xl space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-neutral-800 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
            <h3 className="text-lg font-bold text-white tracking-wide">
              Wet-Lab Microplate Designer & Guide Wizard
            </h3>
          </div>
          <p className="text-xs text-neutral-400 mt-0.5">
            Automated CRISPR guide mapping for liquid handlers (Echo / Biomek)
          </p>
        </div>

        {/* Format Selector */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setFormatType("96")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold font-mono transition ${
              formatType === "96"
                ? "bg-emerald-600 text-white"
                : "bg-neutral-800 text-neutral-400 hover:text-white"
            }`}
          >
            96-Well (8×12)
          </button>
          <button
            onClick={() => setFormatType("384")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold font-mono transition ${
              formatType === "384"
                ? "bg-emerald-600 text-white"
                : "bg-neutral-800 text-neutral-400 hover:text-white"
            }`}
          >
            384-Well (16×24)
          </button>
        </div>
      </div>

      {/* Plate Specifications Bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-neutral-800/40 rounded-xl p-3 border border-neutral-800 text-xs">
        <div>
          <span className="text-neutral-400">Total Plate Capacity</span>
          <p className="font-mono font-bold text-white mt-0.5">{totalWells} Wells</p>
        </div>
        <div>
          <span className="text-neutral-400">Replicates / Guide</span>
          <div className="flex items-center gap-1.5 mt-0.5">
            <select
              value={replicates}
              onChange={(e) => setReplicates(parseInt(e.target.value, 10))}
              className="bg-neutral-800 border border-neutral-700 rounded px-1.5 py-0.5 text-xs text-white font-mono focus:outline-none"
            >
              <option value={1}>1x</option>
              <option value={2}>2x</option>
              <option value={3}>3x</option>
            </select>
          </div>
        </div>
        <div>
          <span className="text-neutral-400">Controls (NegCtrl / Blank)</span>
          <p className="font-mono font-bold text-amber-400 mt-0.5">
            {controlWells} Ctrl + {blankWells} Blank
          </p>
        </div>
        <div>
          <span className="text-neutral-400">Guides Accommodated</span>
          <p className="font-mono font-bold text-emerald-400 mt-0.5">
            {Math.floor((totalWells - controlWells - blankWells) / replicates)} Targets
          </p>
        </div>
      </div>

      {/* Microplate Visual Grid Preview */}
      <div className="space-y-2">
        <div className="flex justify-between items-center text-xs text-neutral-400">
          <span>Plate Grid Preview</span>
          <div className="flex items-center gap-3 font-mono text-[10px]">
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-amber-500" /> NegCtrl
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-neutral-600" /> Blank
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-indigo-500" /> Perturbation
            </span>
          </div>
        </div>

        <div className="bg-neutral-950 p-4 rounded-xl border border-neutral-800 overflow-x-auto">
          <div
            className="grid gap-1 min-w-[320px]"
            style={{
              gridTemplateColumns: `repeat(${cols}, minmax(14px, 1fr))`,
            }}
          >
            {Array.from({ length: totalWells }).map((_, idx) => {
              let wellColor = "bg-indigo-600/70 border-indigo-500/40";
              let title = `Perturbation Target (Rep)`;

              if (idx < controlWells) {
                wellColor = "bg-amber-600/80 border-amber-500/50";
                title = `NegCtrl #${idx + 1}`;
              } else if (idx < controlWells + blankWells) {
                wellColor = "bg-neutral-700/80 border-neutral-600/50";
                title = `Blank Media #${idx - controlWells + 1}`;
              }

              return (
                <div
                  key={idx}
                  title={title}
                  className={`h-4 rounded-[3px] border ${wellColor} hover:scale-125 transition-transform cursor-pointer`}
                />
              );
            })}
          </div>
        </div>
      </div>

      {/* Export Action */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-2">
        <span className="text-xs text-neutral-400">
          Ready for robot automation: Compatible with Echo 525, Tecan, and Hamilton STAR.
        </span>

        <button
          onClick={handleExportCSV}
          disabled={isExporting}
          className="w-full sm:w-auto px-5 py-2.5 rounded-xl font-semibold text-xs bg-emerald-600 hover:bg-emerald-500 text-white transition flex items-center justify-center gap-2 shadow-lg disabled:opacity-50"
        >
          {isExporting ? (
            <span className="animate-spin w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full" />
          ) : (
            <span>⬇ Export Plate CSV</span>
          )}
          {downloadSuccess && <span className="text-emerald-300">✓ Downloaded</span>}
        </button>
      </div>
    </div>
  );
};

export default GuidePlateWizard;
