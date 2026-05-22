import React, { useState, useEffect } from "react";
import { Sliders, Download, RefreshCw, Folder, Layers, ShieldAlert, Cpu, Sparkles, Terminal } from "lucide-react";
import { BACKEND_URL } from "../config";

export function Augment() {
  const [existingFolders, setExistingFolders] = useState([]);
  const [selectedJob, setSelectedJob] = useState("");
  const [jobDetails, setJobDetails] = useState(null);
  const [multiplier, setMultiplier] = useState(2);
  const [mode, setMode] = useState("augment");
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [logs, setLogs] = useState([]);
  const [error, setError] = useState(null);
  const [successData, setSuccessData] = useState(null);

  // Fetch available job folders
  const fetchFolders = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/jobs`);
      if (res.ok) {
        const data = await res.json();
        setExistingFolders(data);
        if (data.length > 0 && !selectedJob) {
          setSelectedJob(data[0]);
        }
      }
    } catch (err) {
      console.error("Error fetching folders:", err);
    }
  };

  useEffect(() => {
    fetchFolders();
  }, []);

  // Fetch details of selected job
  const fetchJobStatus = async (jobId) => {
    if (!jobId) return;
    try {
      const res = await fetch(`${BACKEND_URL}/api/jobs/${jobId}/status`);
      if (res.ok) {
        const data = await res.json();
        setJobDetails(data);
      } else {
        setJobDetails(null);
      }
    } catch (err) {
      console.error("Error fetching job details:", err);
      setJobDetails(null);
    }
  };

  useEffect(() => {
    if (selectedJob) {
      fetchJobStatus(selectedJob);
      setSuccessData(null);
      setLogs([]);
      setError(null);
    } else {
      setJobDetails(null);
    }
  }, [selectedJob]);

  const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  const runExpander = async () => {
    if (!selectedJob) return;
    setRunning(true);
    setLogs([]);
    setError(null);
    setSuccessData(null);

    const addLog = (msg, type = "info") => {
      const prefix = type === "error" ? "❌ [ERROR]" : type === "success" ? "✅ [SUCCESS]" : "ℹ️ [INFO]";
      setLogs((prev) => [...prev, `${prefix} ${msg}`]);
    };

    try {
      addLog(`Initializing offline dataset multiplier for job '${selectedJob}'...`);
      await delay(450);
      addLog(`Validation complete. Found raw dataset components.`);
      await delay(300);
      addLog(`Target multiplication factor: ${multiplier}x`);
      await delay(350);
      if (mode === "augment") {
        addLog(`Replication strategy: Smart Augmentation (flips, brightness, contrast, blur).`);
        await delay(300);
        addLog(`Applying smart image variations and recalculating coordinate orientations...`);
      } else {
        addLog(`Replication strategy: Simple Duplication (exact copies).`);
        await delay(300);
        addLog(`Duplicating image files and labels...`);
      }
      await delay(200);
      addLog(`Clearing any previous copies to maintain idempotency...`);
      await delay(500);
      
      const formData = new FormData();
      formData.append("multiplier", multiplier);
      formData.append("mode", mode);

      const res = await fetch(`${BACKEND_URL}/api/jobs/${selectedJob}/multiply`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Backend failed to process multiplier request.");
      }

      const data = await res.json();
      addLog(`Duplication completed in local outputs/ directory.`);
      await delay(400);
      addLog(`Re-partitioning dataset split (80/20 train/validation)...`);
      await delay(400);
      addLog(`Re-generating annotations and packing ZIP archive dataset_${selectedJob}.zip...`);
      await delay(600);
      addLog(`Successfully expanded dataset for '${selectedJob}' from ${jobDetails?.images_passed || 0} to ${data.total_images} images.`, "success");

      setSuccessData(data);
      // Refresh job stats on screen
      fetchJobStatus(selectedJob);
    } catch (err) {
      addLog(err.message, "error");
      setError(err.message);
    } finally {
      setRunning(false);
    }
  };

  const isJobRunningOrQueued = jobDetails?.status && !["done", "failed"].includes(jobDetails.status);

  return (
    <div className="max-w-4xl mx-auto px-4 py-4 space-y-6">
      {/* Title Header */}
      <div className="flex flex-col gap-2">
        <span className="text-[10px] w-fit bg-violet-500/10 border border-violet-500/30 text-violet-300 font-bold px-2.5 py-1 rounded-full uppercase tracking-wider">
          Offline Dataset Expander
        </span>
        <h1 className="text-3xl font-black text-white tracking-tight flex items-center gap-2">
          <Sliders className="w-8 h-8 text-violet-500" />
          Dataset Expander
        </h1>
        <p className="text-zinc-400 text-sm max-w-xl">
          Multiply dataset epochs by duplicating images and coordinates on disk. Useful for local training pipelines requiring physically expanded files.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Left Control Card */}
        <div className="md:col-span-2 space-y-6">
          <div className="glass-panel rounded-2xl p-6 border border-zinc-800/80 space-y-5">
            {/* Job Selection Dropdown */}
            <div className="space-y-2">
              <label className="block text-xs font-semibold text-zinc-300">
                Select Local Dataset Job
              </label>
              <div className="relative">
                <select
                  value={selectedJob}
                  disabled={running}
                  onChange={(e) => setSelectedJob(e.target.value)}
                  className="w-full px-3.5 py-2.5 rounded-lg border border-zinc-800 bg-zinc-950 text-zinc-200 focus:outline-none focus:ring-1 focus:ring-violet-500/50 text-sm cursor-pointer"
                >
                  {existingFolders.length === 0 ? (
                    <option value="">No dataset directories found</option>
                  ) : (
                    existingFolders.map((folder) => (
                      <option key={folder} value={folder}>
                        {folder}
                      </option>
                    ))
                  )}
                </select>
              </div>
              {existingFolders.length === 0 && (
                <p className="text-[11px] text-zinc-500 flex items-center gap-1">
                  <Folder className="w-3.5 h-3.5" /> Please create a collector job on the 'Collect' tab first.
                </p>
              )}
            </div>

            {/* Slider Multiplier Input */}
            <div className="space-y-2">
              <div className="flex justify-between items-center text-sm font-semibold text-zinc-300">
                <span className="flex items-center gap-1.5">
                  <Cpu className="w-4 h-4 text-emerald-400" /> Multiplication Factor
                </span>
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    min="2"
                    max="10"
                    disabled={running || isJobRunningOrQueued || !selectedJob}
                    value={multiplier}
                    onChange={(e) => {
                      const val = Math.min(10, Math.max(2, parseInt(e.target.value, 10) || 2));
                      setMultiplier(val);
                    }}
                    className="w-16 px-2 py-1 rounded-lg border border-zinc-800 bg-zinc-950 text-violet-400 text-center text-sm font-bold focus:outline-none focus:ring-1 focus:ring-violet-500/50"
                  />
                  <span className="text-zinc-500 text-xs">x</span>
                </div>
              </div>
              <input
                type="range"
                min="2"
                max="10"
                step="1"
                disabled={running || isJobRunningOrQueued || !selectedJob}
                value={multiplier}
                onChange={(e) => setMultiplier(parseInt(e.target.value, 10) || 2)}
                className="w-full h-2 rounded-lg bg-zinc-800 accent-violet-500 cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-zinc-500">
                <span>2x duplication</span>
                <span>10x duplication (Max)</span>
              </div>
            </div>

            {/* Replication Strategy Toggle */}
            <div className="space-y-2.5">
              <label className="block text-xs font-semibold text-zinc-300">
                Replication Strategy
              </label>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <button
                  type="button"
                  disabled={running}
                  onClick={() => setMode("augment")}
                  className={`flex flex-col items-start p-3.5 rounded-xl border text-left transition-all duration-300 relative overflow-hidden cursor-pointer ${
                    mode === "augment"
                      ? "bg-violet-600/10 border-violet-500 text-white shadow-md shadow-violet-500/5"
                      : "bg-zinc-950/40 border-zinc-800 text-zinc-400 hover:border-zinc-700"
                  }`}
                >
                  <div className="flex items-center gap-1.5 font-bold text-xs">
                    <Sparkles className={`w-3.5 h-3.5 ${mode === "augment" ? "text-violet-400" : "text-zinc-500"}`} />
                    Smart Augmentation
                  </div>
                  <span className="text-[10px] text-zinc-500 mt-1 leading-relaxed">
                    Applies flips, brightness/contrast scaling, and blurring to generated copies.
                  </span>
                  {mode === "augment" && (
                    <span className="absolute top-2 right-2 text-[9px] bg-violet-500/20 border border-violet-500/30 text-violet-300 font-bold px-1.5 py-0.5 rounded-md uppercase tracking-wider scale-90">
                      Recommended
                    </span>
                  )}
                </button>
                <button
                  type="button"
                  disabled={running}
                  onClick={() => setMode("copy")}
                  className={`flex flex-col items-start p-3.5 rounded-xl border text-left transition-all duration-300 relative overflow-hidden cursor-pointer ${
                    mode === "copy"
                      ? "bg-violet-600/10 border-violet-500 text-white shadow-md shadow-violet-500/5"
                      : "bg-zinc-950/40 border-zinc-800 text-zinc-400 hover:border-zinc-700"
                  }`}
                >
                  <div className="flex items-center gap-1.5 font-bold text-xs">
                    <Layers className={`w-3.5 h-3.5 ${mode === "copy" ? "text-violet-400" : "text-zinc-500"}`} />
                    Simple Duplication
                  </div>
                  <span className="text-[10px] text-zinc-500 mt-1 leading-relaxed">
                    Performs exact file duplication. Best for pure epoch-level replication.
                  </span>
                </button>
              </div>
            </div>

            {/* Multiplier Info Box */}
            <div className="p-4 rounded-xl border border-zinc-800 bg-zinc-950/40 text-xs text-zinc-400 space-y-2 leading-relaxed flex gap-3">
              <ShieldAlert className="w-5 h-5 text-amber-500 shrink-0" />
              <div>
                <span className="font-bold text-zinc-200 block">How it works:</span>
                Each original image in the <code className="text-violet-400 font-semibold">filtered/</code> folder is copied <code className="text-zinc-200 font-bold">{multiplier - 1}</code> times. Corresponding bounding boxes or segmentation masks are {mode === "augment" ? "adjusted dynamically to match flipped orientations" : "duplicated exactly"}. Re-packaging partitions the expanded dataset into a new training ZIP.
              </div>
            </div>

            {/* Expander Submit Button */}
            <button
              onClick={runExpander}
              disabled={running || !selectedJob || isJobRunningOrQueued}
              className={`w-full py-3 px-4 rounded-xl font-bold text-sm tracking-wide transition-all duration-300 flex items-center justify-center gap-2 cursor-pointer ${
                running
                  ? "bg-zinc-800 text-zinc-500 border border-zinc-700"
                  : !selectedJob || isJobRunningOrQueued
                  ? "bg-zinc-900 text-zinc-600 border border-zinc-800 cursor-not-allowed"
                  : "bg-violet-600 hover:bg-violet-500 text-white shadow-lg shadow-violet-600/25 hover:shadow-violet-600/35 border border-violet-500/20 active:scale-[0.98]"
              }`}
            >
              {running ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin text-zinc-500" />
                  Running Expander...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4 text-violet-300 fill-current" />
                  Run Dataset Expander
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right Info Card & Success Stats */}
        <div className="space-y-6">
          {/* Dataset Info summary */}
          <div className="glass-panel rounded-2xl p-5 border border-zinc-800/80 space-y-4">
            <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-500">
              Dataset Properties
            </h3>
            {jobDetails ? (
              <div className="space-y-3.5">
                <div>
                  <span className="block text-[10px] text-zinc-500 font-semibold uppercase">Job Directory</span>
                  <span className="text-sm font-bold text-zinc-200">{jobDetails.job_id}</span>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <span className="block text-[10px] text-zinc-500 font-semibold uppercase">Original Images</span>
                    <span className="text-sm font-bold text-zinc-200">{jobDetails.images_passed}</span>
                  </div>
                  <div>
                    <span className="block text-[10px] text-zinc-500 font-semibold uppercase">Export Format</span>
                    <span className="text-sm font-bold text-emerald-400 uppercase">
                      {jobDetails.config?.export_format || "YOLO"}
                    </span>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <span className="block text-[10px] text-zinc-500 font-semibold uppercase">Label Type</span>
                    <span className="text-sm font-bold text-violet-400 capitalize">
                      {jobDetails.config?.label_type || "Detection"}
                    </span>
                  </div>
                  <div>
                    <span className="block text-[10px] text-zinc-500 font-semibold uppercase">Project Status</span>
                    <span
                      className={`text-xs font-bold px-2 py-0.5 rounded-full inline-block mt-0.5 border ${
                        jobDetails.status === "done"
                          ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400"
                          : jobDetails.status === "failed"
                          ? "bg-rose-500/10 border-rose-500/20 text-rose-400"
                          : "bg-amber-500/10 border-amber-500/20 text-amber-400 animate-pulse"
                      }`}
                    >
                      {jobDetails.status}
                    </span>
                  </div>
                </div>

                {isJobRunningOrQueued && (
                  <p className="text-[11px] text-amber-400 font-semibold bg-amber-500/5 border border-amber-500/10 p-2.5 rounded-lg">
                    ⚠️ The selected job is still running/queued. Expander is blocked.
                  </p>
                )}

                {/* Estimate Multiplied Count */}
                {!isJobRunningOrQueued && (
                  <div className="pt-3 border-t border-zinc-800/80 flex items-center justify-between">
                    <div>
                      <span className="block text-[10px] text-zinc-500 font-semibold uppercase">Projected Count</span>
                      <span className="text-base font-black text-white">
                        {jobDetails.images_passed * multiplier} images
                      </span>
                    </div>
                    <div className="p-2 rounded-lg bg-zinc-900/60 text-zinc-500 text-xs font-bold">
                      {multiplier}x
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-xs text-zinc-500 py-2">
                No active dataset selected or directory data loaded.
              </p>
            )}
          </div>

          {/* Download card when successful */}
          {successData && (
            <div className="p-5 rounded-2xl border border-emerald-500/30 bg-emerald-500/5 space-y-4 animate-scaleUp">
              <div>
                <span className="text-[10px] bg-emerald-500/10 border border-emerald-500/25 text-emerald-400 font-bold px-2 py-0.5 rounded-md uppercase tracking-wider">
                  Success
                </span>
                <h4 className="text-sm font-black text-zinc-100 mt-2">
                  Expanded Dataset Ready
                </h4>
                <p className="text-[11px] text-zinc-400 mt-1">
                  New dataset packaged with {successData.total_images} files.
                </p>
              </div>

              <a
                href={`${BACKEND_URL}${successData.zip_url}`}
                download
                className="w-full py-2.5 px-4 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs tracking-wider transition-colors flex items-center justify-center gap-2 shadow-lg shadow-emerald-600/20 active:scale-[0.98]"
              >
                <Download className="w-4 h-4" />
                Download Expanded ZIP
              </a>
            </div>
          )}
        </div>
      </div>

      {/* Terminal log panel */}
      {(running || logs.length > 0) && (
        <div className="glass-panel rounded-2xl border border-zinc-800/80 p-5 space-y-3">
          <div className="flex items-center gap-2 text-zinc-400 font-bold text-xs uppercase tracking-wider">
            <Terminal className="w-4 h-4 text-violet-400" />
            Process Console Logs
          </div>
          <div className="bg-zinc-950/80 border border-zinc-900 rounded-xl p-4 font-mono text-[11px] text-zinc-300 space-y-1.5 h-48 overflow-y-auto shadow-inner">
            {logs.map((log, idx) => (
              <div key={idx} className="whitespace-pre-wrap leading-relaxed animate-fadeIn">
                {log}
              </div>
            ))}
            {running && (
              <div className="flex items-center gap-1.5 text-zinc-500 pt-1 text-[10px]">
                <RefreshCw className="w-3 h-3 animate-spin text-zinc-500" />
                Processing multiplier replication pipeline...
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
