import React, { useEffect, useState } from "react";
import { Loader2, ArrowLeft, Image as ImageIcon, CheckCircle, AlertTriangle, Download, ChevronRight } from "lucide-react";
import { useJob } from "../hooks/useJob";
import { BACKEND_URL } from "../config";

function ProgressImage({ imageUrl, filename }) {
  const [error, setError] = useState(false);
  if (error) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center bg-zinc-900 text-zinc-500 p-2 text-center">
        <svg className="w-8 h-8 text-zinc-600 mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
        <span className="text-[8px] text-zinc-500 truncate w-full">{filename}</span>
      </div>
    );
  }
  return (
    <img
      src={imageUrl}
      alt={filename}
      onError={() => setError(true)}
      className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
      loading="lazy"
    />
  );
}

const STAGES = [
  { id: "Parser", label: "Query Parser" },
  { id: "Scraper", label: "Image Scraper" },
  { id: "Filter", label: "Quality Filter" },
  { id: "Labeler", label: "Auto-Labeler" },
  { id: "Exporter", label: "Exporter" },
];

export function Progress({ jobId, onComplete, onCancel }) {
  const { job, error: pollError } = useJob(jobId, onComplete);
  const [previewImages, setPreviewImages] = useState([]);

  const handleCancelClick = async () => {
    try {
      await fetch(`${BACKEND_URL}/api/jobs/${jobId}/cancel`, { method: "POST" });
    } catch (err) {
      console.error("Failed to cancel job:", err);
    }
    onCancel();
  };
  
  // Derive status once to use as a stable dependency
  const jobStatus = job?.status;
  
  // Fetch preview images every 3 seconds while job is running
  useEffect(() => {
    if (!jobId) return;
    // Stop polling previews once the job is finished
    if (jobStatus === "done" || jobStatus === "failed") return;
    
    const fetchPreview = async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/api/jobs/${jobId}/preview`);
        if (res.ok) {
          const data = await res.json();
          setPreviewImages(data.images || []);
        }
      } catch (err) {
        console.error("Error fetching live preview:", err);
      }
    };
    
    fetchPreview();
    const interval = setInterval(fetchPreview, 3000);
    return () => clearInterval(interval);
  }, [jobId, jobStatus]);

  // Set final preview once job finishes
  useEffect(() => {
    if (job && job.status === "done") {
      setPreviewImages(job.results || []);
    }
  }, [job]);

  if (!job) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] gap-3">
        <Loader2 className="w-10 h-10 text-violet-500 animate-spin" />
        <p className="text-zinc-400 text-sm">Synchronizing pipeline job...</p>
      </div>
    );
  }

  const activeStageIndex = job.stage === "Completed" || job.status === "done"
    ? STAGES.length
    : STAGES.findIndex((s) => s.id === job.stage);
  const isFailed = job.status === "failed";
  const isDone = job.status === "done";
  
  return (
    <div className="max-w-4xl mx-auto px-4 py-8 animate-fadeIn">
      {/* Header */}
      <div className="flex items-start justify-between mb-8">
        <div>
          <button
            onClick={handleCancelClick}
            className="flex items-center gap-1 text-zinc-500 hover:text-zinc-300 text-xs font-semibold uppercase tracking-wider mb-2 focus:outline-none transition-colors cursor-pointer"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            Cancel &amp; Return
          </button>
          <h2 className="text-2xl font-black text-white flex items-center gap-2">
            Dataset Generator Job
            <span className="text-xs bg-zinc-800 text-zinc-400 border border-zinc-700 px-2 py-0.5 rounded-full font-normal">
              {jobId.includes("-") && jobId.length > 20 ? jobId.slice(0, 8) : jobId}
            </span>
          </h2>
          <p className="text-zinc-400 text-xs mt-1">
            Query: <span className="text-zinc-300 italic">"{job.config?.query}"</span>
          </p>
          {job.has_sample_image && (
            <div className="mt-3 flex items-center gap-2 bg-violet-950/20 border border-violet-500/10 rounded-lg p-2 max-w-sm animate-slideDown">
              <img
                src={`${BACKEND_URL}${job.sample_image_url}`}
                alt="Reference Sample"
                className="w-10 h-10 object-cover rounded-md border border-violet-500/20"
              />
              <div>
                <span className="text-[10px] text-violet-400 font-extrabold uppercase tracking-wider block font-black">Visual Grounding Active</span>
                <span className="text-[10px] text-zinc-400 block -mt-0.5">Filtering by similarity to sample image</span>
              </div>
            </div>
          )}
        </div>
        
        {isFailed && (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full border border-red-500/30 bg-red-500/10 text-red-400 text-xs font-semibold">
            <AlertTriangle className="w-4 h-4" />
            Failed
          </span>
        )}
        {!isFailed && !isDone && (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full border border-violet-500/30 bg-violet-500/10 text-violet-400 text-xs font-semibold pulse-glow-violet">
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
            Processing
          </span>
        )}
        {isDone && (
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-full border border-emerald-500/30 bg-emerald-500/10 text-emerald-400 text-xs font-semibold">
            <CheckCircle className="w-4 h-4" />
            Completed
          </span>
        )}
      </div>

      {/* Progress Card */}
      <div className="glass-panel p-6 rounded-2xl shadow-lg mb-8 space-y-6">
        {/* Horizontal Pipeline Steps */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {STAGES.map((stage, idx) => {
            const isCompleted = idx < activeStageIndex;
            const isActive = idx === activeStageIndex;
            const isFuture = idx > activeStageIndex;
            
            return (
              <div
                key={stage.id}
                className={`flex flex-col p-3 rounded-xl border transition-all ${
                  isActive
                    ? "bg-violet-950/20 border-violet-500/50 text-white"
                    : isCompleted
                    ? "bg-zinc-900/40 border-emerald-500/20 text-zinc-300"
                    : "bg-zinc-950/10 border-zinc-900 text-zinc-600"
                }`}
              >
                <span className="text-[10px] uppercase font-bold tracking-wider mb-1 flex items-center justify-between">
                  Step 0{idx + 1}
                  {isCompleted && <CheckCircle className="w-3 h-3 text-emerald-500" />}
                  {isActive && !isFailed && <Loader2 className="w-3 h-3 text-violet-400 animate-spin" />}
                  {isActive && isFailed && <AlertTriangle className="w-3 h-3 text-red-500" />}
                </span>
                <span className="text-xs font-bold truncate">{stage.label}</span>
              </div>
            );
          })}
        </div>

        {/* Progress Bar */}
        <div className="space-y-2">
          <div className="flex justify-between items-center text-xs font-bold">
            <span className="text-zinc-400">Pipeline Progress</span>
            <span className="text-violet-400">{job.progress}%</span>
          </div>
          <div className="w-full h-3 rounded-full bg-zinc-900 overflow-hidden p-0.5 border border-zinc-800">
            <div
              className={`h-full rounded-full transition-all duration-500 ${
                isFailed
                  ? "bg-gradient-to-r from-red-600 to-rose-600 shadow-[0_0_10px_rgba(239,68,68,0.3)]"
                  : "bg-gradient-to-r from-violet-600 to-indigo-600 shadow-[0_0_10px_rgba(139,92,246,0.3)]"
              }`}
              style={{ width: `${job.progress}%` }}
            />
          </div>
        </div>

        {/* Board Stats */}
        <div className="grid grid-cols-3 gap-4 pt-4 border-t border-zinc-800/80">
          <div className="text-center p-3 rounded-xl bg-zinc-950/40 border border-zinc-900">
            <span className="block text-zinc-500 text-[10px] uppercase font-bold tracking-wider mb-1">
              Scraped Raw
            </span>
            <span className="text-xl font-extrabold text-zinc-200">
              {job.images_collected}
            </span>
          </div>
          <div className="text-center p-3 rounded-xl bg-zinc-950/40 border border-zinc-900">
            <span className="block text-zinc-500 text-[10px] uppercase font-bold tracking-wider mb-1">
              Passed Filter
            </span>
            <span className="text-xl font-extrabold text-emerald-400">
              {job.images_passed}
            </span>
          </div>
          <div className="text-center p-3 rounded-xl bg-zinc-950/40 border border-zinc-900">
            <span className="block text-zinc-500 text-[10px] uppercase font-bold tracking-wider mb-1">
              State
            </span>
            <span className={`text-xs font-extrabold uppercase tracking-wide inline-block mt-1 ${
              isFailed ? "text-red-400" : "text-violet-400"
            }`}>
              {job.status}
            </span>
          </div>
        </div>

        {/* Failed Banner */}
        {isFailed && (
          <div className="flex items-start gap-2.5 p-4 rounded-xl border border-red-500/30 bg-red-500/10 text-red-300 text-sm">
            <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5 text-red-500" />
            <div>
              <span className="font-bold text-red-200">Pipeline Error:</span>
              <p className="mt-1 text-zinc-400 text-xs font-mono">{job.error || pollError || "Unknown pipeline crash occurred."}</p>
              <button
                onClick={onCancel}
                className="mt-3 flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-zinc-900 hover:bg-zinc-800 text-white text-xs font-semibold border border-zinc-800 cursor-pointer transition-all"
              >
                Configure New Job
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Live Preview Grid */}
      <div className="space-y-4">
        <h3 className="text-sm font-bold text-zinc-300 flex items-center gap-1.5 uppercase tracking-wider">
          <ImageIcon className="w-4 h-4 text-violet-400" />
          Live Image Pipeline Feed (First 20)
        </h3>
        
        {previewImages.length === 0 ? (
          <div className="glass-panel rounded-2xl p-10 flex flex-col items-center justify-center text-center text-zinc-500 border-dashed border-zinc-800">
            <ImageIcon className="w-12 h-12 stroke-[1] mb-2 text-zinc-700" />
            <p className="text-sm font-semibold">Feed is empty</p>
            <p className="text-xs text-zinc-600 max-w-sm mt-1">
              Images will begin populating here in real-time as the scraper downloads them and the quality filter validates them.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-5 gap-3">
            {previewImages.map((img, idx) => (
              <div
                key={img.filename}
                className="relative aspect-square rounded-xl overflow-hidden border border-zinc-800 bg-zinc-950 flex items-center justify-center group animate-zoomIn"
              >
                <ProgressImage
                  imageUrl={`${BACKEND_URL}${img.url}`}
                  filename={img.filename}
                />
                
                {/* Image Index Overlay */}
                <div className="absolute top-1.5 left-1.5 text-[9px] bg-black/70 backdrop-blur-sm border border-zinc-800 text-zinc-300 font-bold px-1.5 py-0.5 rounded">
                  {idx + 1}
                </div>

                {/* Status Indicator */}
                {img.labels_data?.label && (
                  <div className="absolute bottom-1.5 left-1.5 text-[9px] max-w-[85%] truncate bg-violet-600/90 backdrop-blur-sm text-white font-semibold px-1.5 py-0.5 rounded shadow">
                    {img.labels_data.label}
                  </div>
                )}
                {img.labels_data?.bboxes && img.labels_data.bboxes.length > 0 && (
                  <div className="absolute bottom-1.5 left-1.5 text-[9px] bg-emerald-600/95 backdrop-blur-sm text-white font-semibold px-1.5 py-0.5 rounded shadow">
                    {img.labels_data.bboxes.length} Box(es)
                  </div>
                )}
                {img.labels_data?.polygons && img.labels_data.polygons.length > 0 && (
                  <div className="absolute bottom-1.5 left-1.5 text-[9px] bg-indigo-600/95 backdrop-blur-sm text-white font-semibold px-1.5 py-0.5 rounded shadow">
                    {img.labels_data.polygons.length} Mask(s)
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
export default Progress;
