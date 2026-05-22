import React, { useState, useEffect, useRef } from "react";
import {
  RefreshCw, Download, X, Folder, Tag, Layers, ScanSearch,
  PenLine, Trash2, Plus, Save, ChevronLeft, ChevronRight,
  Sparkles, CheckCircle, AlertTriangle, Box, Hexagon, Type
} from "lucide-react";
import { BACKEND_URL } from "../config";

/* ─────────────────────────────────────────────
   Reusable AnnotatedImage (same logic as Results page)
   ───────────────────────────────────────────── */
function AnnotatedImage({ imageUrl, filename, labelsData, mode }) {
  const [scaling, setScaling] = useState({ scaleX: 1, scaleY: 1, width: 0, height: 0 });
  const [imgLoaded, setImgLoaded] = useState(false);
  const [imgLoadFailed, setImgLoadFailed] = useState(false);
  const imgRef = useRef(null);

  const handleImageLoad = () => {
    if (!imgRef.current) return;
    const { clientWidth, clientHeight, naturalWidth, naturalHeight } = imgRef.current;
    if (naturalWidth && naturalHeight) {
      setScaling({ scaleX: clientWidth / naturalWidth, scaleY: clientHeight / naturalHeight, width: clientWidth, height: clientHeight, naturalWidth, naturalHeight });
      setImgLoaded(true);
    }
  };

  useEffect(() => {
    const h = () => handleImageLoad();
    window.addEventListener("resize", h);
    return () => window.removeEventListener("resize", h);
  }, [imgLoaded]);

  const renderBboxes = () => {
    if (!imgLoaded || !labelsData?.bboxes) return null;
    return labelsData.bboxes.map((box, idx) => {
      const left = box.x_min * scaling.scaleX;
      const top = box.y_min * scaling.scaleY;
      const w = (box.x_max - box.x_min) * scaling.scaleX;
      const h = (box.y_max - box.y_min) * scaling.scaleY;
      return (
        <div key={idx} className="absolute border-2 border-violet-500 bg-violet-500/10 rounded-sm pointer-events-none" style={{ left, top, width: w, height: h }}>
          <span className="absolute -top-5 left-0 bg-violet-600 text-white font-semibold text-[9px] px-1 rounded shadow-md truncate max-w-[120px]">
            {box.label} ({Math.round(box.confidence * 100)}%)
          </span>
        </div>
      );
    });
  };

  const renderPolygons = () => {
    if (!imgLoaded || !labelsData?.polygons || !scaling.naturalWidth) return null;
    return (
      <svg className="absolute top-0 left-0 w-full h-full pointer-events-none" viewBox={`0 0 ${scaling.naturalWidth} ${scaling.naturalHeight}`} width={scaling.width} height={scaling.height}>
        {labelsData.polygons.map((polyItem, idx) =>
          polyItem.polygons.map((pts, pIdx) => {
            const pointsString = pts.reduce((acc, val, i) => acc + (i % 2 === 0 ? `${val},` : `${val} `), "").trim();
            return <polygon key={`${idx}-${pIdx}`} points={pointsString} className="fill-violet-500/30 stroke-violet-500 stroke-2" />;
          })
        )}
      </svg>
    );
  };

  if (imgLoadFailed) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center bg-zinc-900 text-zinc-500 p-4 text-center">
        <AlertTriangle className="w-8 h-8 text-zinc-600 mb-2" />
        <span className="text-[10px] font-bold truncate max-w-full">{filename}</span>
      </div>
    );
  }

  return (
    <div className="relative w-full h-full select-none overflow-hidden flex items-center justify-center bg-zinc-950">
      <img ref={imgRef} src={`${BACKEND_URL}${imageUrl}`} alt={filename} onLoad={handleImageLoad} onError={() => setImgLoadFailed(true)} className="max-w-full max-h-full object-contain" />
      {mode === "detection" && renderBboxes()}
      {mode === "segmentation" && renderPolygons()}
    </div>
  );
}

/* ─────────────────────────────────────────────
   Toast Notification
   ───────────────────────────────────────────── */
function Toast({ message, type, onClose }) {
  useEffect(() => {
    const t = setTimeout(onClose, 3500);
    return () => clearTimeout(t);
  }, [onClose]);

  const bg = type === "success" ? "bg-emerald-600" : type === "error" ? "bg-rose-600" : "bg-amber-600";
  const Icon = type === "success" ? CheckCircle : type === "error" ? AlertTriangle : Save;

  return (
    <div className={`fixed bottom-6 right-6 z-[100] ${bg} text-white px-5 py-3 rounded-xl shadow-2xl flex items-center gap-2.5 animate-fadeIn text-sm font-bold`}>
      <Icon className="w-4 h-4" />
      {message}
    </div>
  );
}

/* ─────────────────────────────────────────────
   Manual Edit Drawer
   ───────────────────────────────────────────── */
function EditDrawer({ image, mode, jobId, onClose, onSaved }) {
  const [editData, setEditData] = useState(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!image) return;
    // Deep clone the current labels
    setEditData(JSON.parse(JSON.stringify(image.labels_data || {})));
  }, [image]);

  useEffect(() => {
    const handleKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [onClose]);

  if (!image || !editData) return null;

  const handleSave = async () => {
    setSaving(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/jobs/${jobId}/labels/${encodeURIComponent(image.filename)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(editData),
      });
      if (!res.ok) {
        const d = await res.json();
        throw new Error(d.detail || "Save failed");
      }
      onSaved(image.filename, editData);
    } catch (err) {
      alert("Error saving: " + err.message);
    } finally {
      setSaving(false);
    }
  };

  /* ── Classification ── */
  const renderClassificationEditor = () => (
    <div className="space-y-4">
      <div className="space-y-1.5">
        <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider">Class Label</label>
        <input
          type="text"
          value={editData.label || ""}
          onChange={(e) => setEditData({ ...editData, label: e.target.value })}
          className="w-full px-3 py-2 rounded-lg border border-zinc-800 bg-zinc-950 text-zinc-200 text-sm focus:outline-none focus:ring-1 focus:ring-violet-500/50"
          placeholder="Enter class name..."
        />
      </div>
      <div className="space-y-1.5">
        <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider">Confidence</label>
        <div className="flex items-center gap-3">
          <input
            type="range" min="0" max="1" step="0.01"
            value={editData.confidence ?? 0}
            onChange={(e) => setEditData({ ...editData, confidence: parseFloat(e.target.value) })}
            className="flex-1 h-2 rounded-lg bg-zinc-800 accent-violet-500"
          />
          <span className="text-sm font-bold text-violet-400 w-12 text-right">{Math.round((editData.confidence ?? 0) * 100)}%</span>
        </div>
      </div>
    </div>
  );

  /* ── Detection ── */
  const renderDetectionEditor = () => {
    const bboxes = editData.bboxes || [];
    const updateBox = (idx, field, val) => {
      const updated = [...bboxes];
      updated[idx] = { ...updated[idx], [field]: field === "label" ? val : parseFloat(val) || 0 };
      setEditData({ ...editData, bboxes: updated });
    };
    const deleteBox = (idx) => {
      setEditData({ ...editData, bboxes: bboxes.filter((_, i) => i !== idx) });
    };
    const addBox = () => {
      setEditData({ ...editData, bboxes: [...bboxes, { label: "object", x_min: 0, y_min: 0, x_max: 100, y_max: 100, confidence: 1.0 }] });
    };

    return (
      <div className="space-y-3">
        {bboxes.length === 0 && <p className="text-xs text-zinc-600 italic">No bounding boxes. Add one below.</p>}
        <div className="space-y-3 max-h-[45vh] overflow-y-auto pr-1">
          {bboxes.map((box, idx) => (
            <div key={idx} className="p-3 rounded-xl border border-zinc-800 bg-zinc-900/30 space-y-2">
              <div className="flex items-center justify-between gap-2">
                <input
                  type="text"
                  value={box.label}
                  onChange={(e) => updateBox(idx, "label", e.target.value)}
                  className="flex-1 px-2.5 py-1.5 rounded-lg border border-zinc-800 bg-zinc-950 text-zinc-200 text-xs font-bold focus:outline-none focus:ring-1 focus:ring-violet-500/50"
                />
                <span className="text-[10px] text-violet-400 font-bold shrink-0">{Math.round(box.confidence * 100)}%</span>
                <button onClick={() => deleteBox(idx)} className="p-1.5 rounded-lg hover:bg-rose-500/10 text-zinc-500 hover:text-rose-400 transition-colors cursor-pointer">
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
              <div className="grid grid-cols-4 gap-1.5">
                {["x_min", "y_min", "x_max", "y_max"].map((f) => (
                  <div key={f} className="space-y-0.5">
                    <label className="text-[8px] font-bold text-zinc-600 uppercase">{f}</label>
                    <input
                      type="number"
                      value={Math.round(box[f])}
                      onChange={(e) => updateBox(idx, f, e.target.value)}
                      className="w-full px-1.5 py-1 rounded border border-zinc-800 bg-zinc-950 text-zinc-300 text-[11px] font-mono text-center focus:outline-none focus:ring-1 focus:ring-violet-500/50"
                    />
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
        <button onClick={addBox} className="w-full py-2 rounded-xl border border-dashed border-zinc-700 text-zinc-400 hover:border-violet-500/50 hover:text-violet-300 text-xs font-bold flex items-center justify-center gap-1.5 transition-colors cursor-pointer">
          <Plus className="w-3.5 h-3.5" /> Add Bounding Box
        </button>
      </div>
    );
  };

  /* ── Segmentation ── */
  const renderSegmentationEditor = () => {
    const polygons = editData.polygons || [];
    const updatePoly = (idx, field, val) => {
      const updated = [...polygons];
      updated[idx] = { ...updated[idx], [field]: val };
      setEditData({ ...editData, polygons: updated });
    };
    const deletePoly = (idx) => {
      setEditData({ ...editData, polygons: polygons.filter((_, i) => i !== idx) });
    };

    return (
      <div className="space-y-3">
        {polygons.length === 0 && <p className="text-xs text-zinc-600 italic">No polygon masks found.</p>}
        <div className="space-y-3 max-h-[45vh] overflow-y-auto pr-1">
          {polygons.map((poly, idx) => (
            <div key={idx} className="p-3 rounded-xl border border-zinc-800 bg-zinc-900/30 space-y-2">
              <div className="flex items-center justify-between gap-2">
                <input
                  type="text"
                  value={poly.label}
                  onChange={(e) => updatePoly(idx, "label", e.target.value)}
                  className="flex-1 px-2.5 py-1.5 rounded-lg border border-zinc-800 bg-zinc-950 text-zinc-200 text-xs font-bold focus:outline-none focus:ring-1 focus:ring-violet-500/50"
                />
                <span className="text-[10px] text-indigo-400 font-bold shrink-0">{Math.round(poly.confidence * 100)}%</span>
                <button onClick={() => deletePoly(idx)} className="p-1.5 rounded-lg hover:bg-rose-500/10 text-zinc-500 hover:text-rose-400 transition-colors cursor-pointer">
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
              <div className="text-[9px] font-mono text-zinc-600 truncate">
                {poly.polygons?.length || 0} polygon group(s)
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="fixed inset-0 z-50 flex items-stretch justify-end bg-black/70 backdrop-blur-sm animate-fadeIn" onClick={onClose}>
      <div className="w-full max-w-lg bg-zinc-950 border-l border-zinc-800 flex flex-col shadow-2xl animate-slideIn" onClick={(e) => e.stopPropagation()}>
        {/* Drawer Header */}
        <div className="px-6 py-4 border-b border-zinc-800 flex items-center justify-between shrink-0">
          <div>
            <h3 className="text-sm font-black text-white flex items-center gap-2">
              <PenLine className="w-4 h-4 text-amber-400" />
              Edit Labels
            </h3>
            <p className="text-[10px] text-zinc-500 font-mono mt-0.5">{image.filename}</p>
          </div>
          <button onClick={onClose} className="p-2 rounded-lg hover:bg-zinc-800 text-zinc-400 hover:text-white transition-colors cursor-pointer">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Image Preview */}
        <div className="h-48 shrink-0 border-b border-zinc-800 bg-black">
          <AnnotatedImage imageUrl={image.url} filename={image.filename} labelsData={editData} mode={mode} />
        </div>

        {/* Editor Body */}
        <div className="flex-1 overflow-y-auto p-6">
          <h4 className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider mb-3">
            {mode === "classification" ? "Class Assignment" : mode === "detection" ? "Bounding Boxes" : "Polygon Masks"}
          </h4>
          {mode === "classification" && renderClassificationEditor()}
          {mode === "detection" && renderDetectionEditor()}
          {mode === "segmentation" && renderSegmentationEditor()}
        </div>

        {/* Save Footer */}
        <div className="px-6 py-4 border-t border-zinc-800 shrink-0">
          <button
            onClick={handleSave}
            disabled={saving}
            className={`w-full py-3 rounded-xl font-bold text-sm flex items-center justify-center gap-2 transition-all cursor-pointer ${
              saving
                ? "bg-zinc-800 text-zinc-500 border border-zinc-700"
                : "bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-500 hover:to-orange-500 text-white shadow-lg shadow-amber-600/20"
            }`}
          >
            {saving ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
            {saving ? "Saving..." : "Save Changes"}
          </button>
        </div>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────
   Main Relabel Page Component
   ───────────────────────────────────────────── */
export function Relabel() {
  // Job selection
  const [folders, setFolders] = useState([]);
  const [selectedJob, setSelectedJob] = useState("");

  // Config panel
  const [labelType, setLabelType] = useState("detection");
  const [targetLabels, setTargetLabels] = useState("");
  const [exportFormat, setExportFormat] = useState("yolo");

  // Image grid state
  const [images, setImages] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalImages, setTotalImages] = useState(0);
  const [loadingImages, setLoadingImages] = useState(false);

  // Relabel action state
  const [relabeling, setRelabeling] = useState(false);
  const [relabelDone, setRelabelDone] = useState(false);

  // Edit drawer
  const [editingImage, setEditingImage] = useState(null);
  const [editedFiles, setEditedFiles] = useState(new Set());

  // Toast
  const [toast, setToast] = useState(null);

  /* ── Fetch job folders ── */
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/api/jobs`);
        if (res.ok) {
          const data = await res.json();
          setFolders(data);
          if (data.length > 0 && !selectedJob) setSelectedJob(data[0]);
        }
      } catch (err) { console.error(err); }
    })();
  }, []);

  /* ── Fetch images when job or page changes ── */
  const fetchImages = async (jobId, page) => {
    if (!jobId) return;
    setLoadingImages(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/jobs/${jobId}/images?page=${page}&per_page=24`);
      if (!res.ok) throw new Error("Failed to load images");
      const data = await res.json();
      setImages(data.images || []);
      setTotalPages(data.total_pages || 1);
      setTotalImages(data.total || 0);
      // Pre-fill config from the job's current settings
      if (data.label_type) setLabelType(data.label_type);
      if (data.export_format) setExportFormat(data.export_format);
      if (data.target_labels !== undefined) setTargetLabels(data.target_labels);
    } catch (err) {
      console.error(err);
      setImages([]);
    } finally {
      setLoadingImages(false);
    }
  };

  useEffect(() => {
    if (selectedJob) {
      setCurrentPage(1);
      setRelabelDone(false);
      setEditedFiles(new Set());
      fetchImages(selectedJob, 1);
    } else {
      setImages([]);
    }
  }, [selectedJob]);

  useEffect(() => {
    if (selectedJob) fetchImages(selectedJob, currentPage);
  }, [currentPage]);

  /* ── Relabel action ── */
  const runRelabel = async () => {
    if (!selectedJob) return;
    setRelabeling(true);
    setRelabelDone(false);
    try {
      const formData = new FormData();
      formData.append("label_type", labelType);
      formData.append("target_labels", targetLabels);
      formData.append("export_format", exportFormat);

      const res = await fetch(`${BACKEND_URL}/api/jobs/${selectedJob}/relabel`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const d = await res.json();
        throw new Error(d.detail || "Relabeling failed");
      }
      const data = await res.json();
      setRelabelDone(true);
      setEditedFiles(new Set());
      setToast({ message: `Relabeled ${data.labeled_count} images successfully!`, type: "success" });
      // Refresh grid
      fetchImages(selectedJob, currentPage);
    } catch (err) {
      setToast({ message: err.message, type: "error" });
    } finally {
      setRelabeling(false);
    }
  };

  /* ── On label saved for a single image ── */
  const handleLabelSaved = (filename, newLabelData) => {
    // Update in local state immediately
    setImages((prev) =>
      prev.map((img) =>
        img.filename === filename ? { ...img, labels_data: newLabelData } : img
      )
    );
    setEditedFiles((prev) => new Set(prev).add(filename));
    setEditingImage(null);
    setToast({ message: `Labels saved for ${filename}`, type: "save" });
  };

  /* ── Label type icons ── */
  const labelTypeOptions = [
    { value: "classification", label: "Classification", icon: Type, color: "text-emerald-400" },
    { value: "detection", label: "Detection", icon: Box, color: "text-violet-400" },
    { value: "segmentation", label: "Segmentation", icon: Hexagon, color: "text-indigo-400" },
  ];

  /* ── Get annotation count for an image card ── */
  const getAnnotationBadge = (img) => {
    const ld = img.labels_data;
    if (!ld || Object.keys(ld).length === 0) return { text: "No Labels", cls: "bg-rose-500/15 text-rose-400 border-rose-500/30" };
    if (ld.failed) return { text: "Failed", cls: "bg-rose-500/15 text-rose-400 border-rose-500/30" };
    if (labelType === "classification") {
      return ld.label ? { text: ld.label, cls: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30" } : { text: "No Label", cls: "bg-rose-500/15 text-rose-400 border-rose-500/30" };
    }
    if (labelType === "detection") {
      const c = ld.bboxes?.length || 0;
      return c > 0 ? { text: `${c} Box${c > 1 ? "es" : ""}`, cls: "bg-violet-500/15 text-violet-400 border-violet-500/30" } : { text: "No Boxes", cls: "bg-rose-500/15 text-rose-400 border-rose-500/30" };
    }
    if (labelType === "segmentation") {
      const c = ld.polygons?.length || 0;
      return c > 0 ? { text: `${c} Mask${c > 1 ? "s" : ""}`, cls: "bg-indigo-500/15 text-indigo-400 border-indigo-500/30" } : { text: "No Masks", cls: "bg-rose-500/15 text-rose-400 border-rose-500/30" };
    }
    return { text: "Labeled", cls: "bg-zinc-500/15 text-zinc-400 border-zinc-500/30" };
  };

  const downloadUrl = `${BACKEND_URL}/api/jobs/${selectedJob}/download`;

  return (
    <div className="max-w-6xl mx-auto px-4 py-6 animate-fadeIn">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
        <div>
          <span className="text-[10px] w-fit bg-amber-500/10 border border-amber-500/30 text-amber-300 font-bold px-2.5 py-1 rounded-full uppercase tracking-wider">
            Precision Label Editor
          </span>
          <h1 className="text-3xl font-black text-white tracking-tight flex items-center gap-2.5 mt-2">
            <ScanSearch className="w-8 h-8 text-amber-500" />
            Relabel Studio
          </h1>
          <p className="text-zinc-400 text-sm max-w-lg mt-1">
            Re-run AI labeling with new parameters or manually fine-tune individual annotations for maximum accuracy.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* ──────────────────── LEFT SIDEBAR: Config Panel ──────────────────── */}
        <div className="lg:col-span-1 space-y-4">
          {/* Job Selector */}
          <div className="glass-panel rounded-2xl p-5 border border-zinc-800/80 space-y-4">
            <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-500 flex items-center gap-1.5">
              <Folder className="w-3.5 h-3.5 text-violet-400" />
              Output Folder
            </h3>
            <select
              value={selectedJob}
              onChange={(e) => setSelectedJob(e.target.value)}
              disabled={relabeling}
              className="w-full px-3 py-2.5 rounded-lg border border-zinc-800 bg-zinc-950 text-zinc-200 text-sm focus:outline-none focus:ring-1 focus:ring-violet-500/50 cursor-pointer"
            >
              {folders.length === 0 ? (
                <option value="">No datasets found</option>
              ) : (
                folders.map((f) => <option key={f} value={f}>{f}</option>)
              )}
            </select>
            {selectedJob && (
              <div className="flex items-center justify-between text-[10px] text-zinc-500">
                <span>{totalImages} images</span>
                <span className="capitalize font-bold text-zinc-400">{labelType}</span>
              </div>
            )}
          </div>

          {/* Label Type */}
          <div className="glass-panel rounded-2xl p-5 border border-zinc-800/80 space-y-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-500 flex items-center gap-1.5">
              <Tag className="w-3.5 h-3.5 text-violet-400" />
              Label Type
            </h3>
            <div className="space-y-2">
              {labelTypeOptions.map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => setLabelType(opt.value)}
                  disabled={relabeling}
                  className={`w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl border text-left text-xs font-bold transition-all cursor-pointer ${
                    labelType === opt.value
                      ? "bg-violet-600/10 border-violet-500/40 text-white shadow-sm shadow-violet-500/5"
                      : "bg-zinc-950/40 border-zinc-800 text-zinc-400 hover:border-zinc-700"
                  }`}
                >
                  <opt.icon className={`w-4 h-4 ${labelType === opt.value ? opt.color : "text-zinc-600"}`} />
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          {/* Target Labels */}
          <div className="glass-panel rounded-2xl p-5 border border-zinc-800/80 space-y-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-500">Target Labels</h3>
            <input
              type="text"
              value={targetLabels}
              onChange={(e) => setTargetLabels(e.target.value)}
              disabled={relabeling}
              placeholder="can, bottle, label..."
              className="w-full px-3 py-2.5 rounded-lg border border-zinc-800 bg-zinc-950 text-zinc-200 text-sm focus:outline-none focus:ring-1 focus:ring-violet-500/50 placeholder-zinc-600"
            />
            <p className="text-[10px] text-zinc-600">Comma-separated class names for the model to detect.</p>
          </div>

          {/* Export Format */}
          <div className="glass-panel rounded-2xl p-5 border border-zinc-800/80 space-y-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-500">Export Format</h3>
            <select
              value={exportFormat}
              onChange={(e) => setExportFormat(e.target.value)}
              disabled={relabeling}
              className="w-full px-3 py-2.5 rounded-lg border border-zinc-800 bg-zinc-950 text-zinc-200 text-sm focus:outline-none focus:ring-1 focus:ring-violet-500/50 cursor-pointer"
            >
              <option value="yolo">YOLO</option>
              <option value="coco">COCO</option>
              <option value="csv">CSV</option>
            </select>
          </div>

          {/* Actions */}
          <div className="space-y-3">
            <button
              onClick={runRelabel}
              disabled={relabeling || !selectedJob}
              className={`w-full py-3 px-4 rounded-xl font-bold text-sm flex items-center justify-center gap-2 transition-all cursor-pointer ${
                relabeling
                  ? "bg-zinc-800 text-zinc-500 border border-zinc-700"
                  : !selectedJob
                  ? "bg-zinc-900 text-zinc-600 border border-zinc-800 cursor-not-allowed"
                  : "bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white shadow-lg shadow-violet-600/25 hover:shadow-violet-600/40 active:scale-[0.98]"
              }`}
            >
              {relabeling ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Re-labeling {totalImages} images...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  Re-run Labeling
                </>
              )}
            </button>

            {selectedJob && (
              <a
                href={downloadUrl}
                download
                className="w-full py-2.5 px-4 rounded-xl border border-zinc-800 bg-zinc-900/60 hover:bg-zinc-800 text-zinc-300 font-bold text-xs flex items-center justify-center gap-2 transition-all cursor-pointer"
              >
                <Download className="w-4 h-4" />
                Download ZIP
              </a>
            )}
          </div>
        </div>

        {/* ──────────────────── MAIN: Image Grid ──────────────────── */}
        <div className="lg:col-span-3">
          {/* Grid Header */}
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-bold text-zinc-300 flex items-center gap-2">
              <Layers className="w-4 h-4 text-violet-400" />
              Image Review
              {totalImages > 0 && <span className="text-zinc-500 font-normal text-xs">({totalImages} total)</span>}
            </h3>
            {editedFiles.size > 0 && (
              <span className="text-[10px] bg-amber-500/10 border border-amber-500/30 text-amber-300 font-bold px-2.5 py-1 rounded-full">
                ✎ {editedFiles.size} edited
              </span>
            )}
          </div>

          {/* Loading State */}
          {loadingImages && (
            <div className="flex flex-col items-center justify-center py-20 gap-3">
              <RefreshCw className="w-8 h-8 text-violet-500 animate-spin" />
              <p className="text-zinc-500 text-xs">Loading images...</p>
            </div>
          )}

          {/* Empty State */}
          {!loadingImages && images.length === 0 && (
            <div className="flex flex-col items-center justify-center py-20 gap-3 glass-panel rounded-2xl border border-zinc-800">
              <Folder className="w-12 h-12 text-zinc-700" />
              <p className="text-zinc-500 text-sm font-bold">No images found</p>
              <p className="text-zinc-600 text-xs">Select a job folder with filtered images to begin.</p>
            </div>
          )}

          {/* Image Cards Grid */}
          {!loadingImages && images.length > 0 && (
            <>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                {images.map((img) => {
                  const badge = getAnnotationBadge(img);
                  const isEdited = editedFiles.has(img.filename);
                  return (
                    <div
                      key={img.filename}
                      onClick={() => setEditingImage(img)}
                      className={`glass-panel rounded-xl overflow-hidden aspect-square border bg-zinc-950 flex flex-col group cursor-pointer transition-all hover:shadow-lg ${
                        isEdited
                          ? "border-amber-500/40 shadow-amber-500/5 hover:border-amber-400/60 hover:shadow-amber-500/10"
                          : "border-zinc-800 hover:border-violet-500/50 hover:shadow-violet-500/5"
                      }`}
                    >
                      {/* Image Frame */}
                      <div className="w-full flex-1 relative overflow-hidden flex items-center justify-center">
                        <AnnotatedImage imageUrl={img.url} filename={img.filename} labelsData={img.labels_data} mode={labelType} />
                        {/* Hover overlay */}
                        <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                          <span className="bg-zinc-900 border border-zinc-700 text-white font-semibold text-[10px] px-3 py-1.5 rounded-lg shadow-lg flex items-center gap-1.5">
                            <PenLine className="w-3 h-3" />
                            Edit Labels
                          </span>
                        </div>
                        {/* Edited badge */}
                        {isEdited && (
                          <span className="absolute top-1.5 right-1.5 text-[8px] bg-amber-500/90 text-white font-bold px-1.5 py-0.5 rounded-md shadow">
                            ✎ Edited
                          </span>
                        )}
                      </div>

                      {/* Card Footer */}
                      <div className="px-2.5 py-2 border-t border-zinc-900/60 bg-zinc-950/80 flex items-center justify-between gap-1">
                        <span className="text-[9px] font-bold text-zinc-500 truncate max-w-[55%]">{img.filename}</span>
                        <span className={`text-[8px] font-bold px-1.5 py-0.5 rounded border ${badge.cls}`}>
                          {badge.text}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Pagination */}
              {totalPages > 1 && (
                <div className="flex items-center justify-center gap-4 mt-6 pt-4 border-t border-zinc-900">
                  <button
                    onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                    disabled={currentPage === 1}
                    className="px-3.5 py-2 rounded-xl border border-zinc-800 bg-zinc-900 text-zinc-300 font-semibold text-xs transition-colors hover:bg-zinc-800 hover:text-white disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer flex items-center gap-1"
                  >
                    <ChevronLeft className="w-3.5 h-3.5" />
                    Previous
                  </button>
                  <span className="text-zinc-400 text-xs font-bold">
                    Page {currentPage} of {totalPages}
                  </span>
                  <button
                    onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                    disabled={currentPage === totalPages}
                    className="px-3.5 py-2 rounded-xl border border-zinc-800 bg-zinc-900 text-zinc-300 font-semibold text-xs transition-colors hover:bg-zinc-800 hover:text-white disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer flex items-center gap-1"
                  >
                    Next
                    <ChevronRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Edit Drawer */}
      {editingImage && (
        <EditDrawer
          image={editingImage}
          mode={labelType}
          jobId={selectedJob}
          onClose={() => setEditingImage(null)}
          onSaved={handleLabelSaved}
        />
      )}

      {/* Toast */}
      {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}
    </div>
  );
}

export default Relabel;
