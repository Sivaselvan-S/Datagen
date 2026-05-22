import React, { useState } from "react";
import { Play, Settings, ShieldAlert, Sparkles, Folder, Layers, Image as ImageIcon, Cpu } from "lucide-react";
import { BACKEND_URL } from "../config";

export function Home({ onSubmit, initialConfig, onClearRepeat }) {
  const [query, setQuery] = useState("");
  const [count, setCount] = useState(20);
  const [label, setLabel] = useState(true);
  const [labelType, setLabelType] = useState("detection");
  const [exportFormat, setExportFormat] = useState("yolo");
  const [qualityThreshold, setQualityThreshold] = useState(0.6);
  const [targetLabels, setTargetLabels] = useState("");
  const [sampleImage, setSampleImage] = useState(null);
  const [sampleImagePreview, setSampleImagePreview] = useState(null);
  const [previousSampleImageUrl, setPreviousSampleImageUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Custom directory & duplicates states
  const [existingFolders, setExistingFolders] = useState([]);
  const [folderMode, setFolderMode] = useState("auto"); // "auto" | "manual"
  const [customFolderName, setCustomFolderName] = useState("");
  const [allowDuplicates, setAllowDuplicates] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  React.useEffect(() => {
    if (initialConfig) {
      if (initialConfig.query) setQuery(initialConfig.query);
      if (initialConfig.count) setCount(initialConfig.count);
      if (initialConfig.label !== undefined) setLabel(initialConfig.label);
      if (initialConfig.label_type) setLabelType(initialConfig.label_type);
      if (initialConfig.export_format) setExportFormat(initialConfig.export_format);
      if (initialConfig.quality_threshold) setQualityThreshold(initialConfig.quality_threshold);
      if (initialConfig.target_labels) {
        if (Array.isArray(initialConfig.target_labels)) {
          setTargetLabels(initialConfig.target_labels.join(", "));
        } else {
          setTargetLabels(String(initialConfig.target_labels));
        }
      }
      if (initialConfig.allow_duplicates !== undefined) setAllowDuplicates(initialConfig.allow_duplicates);
      if (initialConfig.folder_mode) setFolderMode(initialConfig.folder_mode);
      if (initialConfig.custom_folder_name) setCustomFolderName(initialConfig.custom_folder_name);
      
      if (initialConfig.sample_image_url) {
        const fullUrl = initialConfig.sample_image_url.startsWith("http") 
          ? initialConfig.sample_image_url 
          : `${BACKEND_URL}${initialConfig.sample_image_url}`;
        setSampleImagePreview(fullUrl);
        setPreviousSampleImageUrl(initialConfig.sample_image_url);
        setSampleImage(null);
      } else {
        setSampleImagePreview(null);
        setPreviousSampleImageUrl(null);
        setSampleImage(null);
      }
      
      if (onClearRepeat) {
        onClearRepeat();
      }
    }
  }, [initialConfig, onClearRepeat]);

  React.useEffect(() => {
    fetch(`${BACKEND_URL}/api/jobs`)
      .then((res) => res.json())
      .then((data) => setExistingFolders(data))
      .catch((err) => console.error("Error fetching folders:", err));
  }, []);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith("image/")) {
        setSampleImage(file);
        setPreviousSampleImageUrl(null);
        const reader = new FileReader();
        reader.onloadend = () => {
          setSampleImagePreview(reader.result);
        };
        reader.readAsDataURL(file);
      }
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSampleImage(file);
      setPreviousSampleImageUrl(null);
      const reader = new FileReader();
      reader.onloadend = () => {
        setSampleImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleRemoveFile = () => {
    setSampleImage(null);
    setSampleImagePreview(null);
    setPreviousSampleImageUrl(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    if (folderMode === "manual" && !customFolderName.trim()) {
      setError("Please select an existing folder to append to.");
      setLoading(false);
      return;
    }

    try {
      const formData = new FormData();
      formData.append("query", query.trim());
      formData.append("count", parseInt(count, 10));
      formData.append("label", label);
      formData.append("label_type", labelType);
      formData.append("export_format", exportFormat);
      formData.append("quality_threshold", parseFloat(qualityThreshold));
      const labelsStr = Array.isArray(targetLabels) ? targetLabels.join(", ") : String(targetLabels || "");
      formData.append("target_labels", labelsStr.trim());
      formData.append("folder_mode", folderMode);
      formData.append("custom_folder_name", customFolderName.trim());
      formData.append("allow_duplicates", allowDuplicates);
      if (sampleImage) {
        formData.append("sample_image", sampleImage);
      } else if (previousSampleImageUrl) {
        formData.append("sample_image_url", previousSampleImageUrl);
      }

      const res = await fetch(`${BACKEND_URL}/api/jobs`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Failed to start collection job.");
      }

      const data = await res.json();
      onSubmit(data.job_id);
    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-8 animate-fadeIn">
      {/* Hero Section */}
      <div className="text-center mb-10">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-violet-500/30 bg-violet-500/10 text-violet-300 text-xs font-semibold mb-4 pulse-glow-violet">
          <Sparkles className="w-3.5 h-3.5" />
          AI-Powered Data Collector
        </div>
        <h1 className="text-4xl md:text-5xl font-black tracking-tight mb-4 bg-clip-text text-transparent bg-gradient-to-r from-white via-zinc-200 to-zinc-500">
          Scrape, Filter &amp; Label Datasets
        </h1>
        <p className="text-zinc-400 text-base md:text-lg max-w-2xl mx-auto">
          Enter a search query, set your requirements, and watch CLIP and Florence-2 collect, filter, and label a production-ready vision dataset.
        </p>
      </div>

      {/* Main Configuration Card */}
      <form onSubmit={handleSubmit} className="glass-panel p-6 md:p-8 rounded-2xl shadow-xl relative overflow-hidden">
        {/* Shimmer Overlay */}
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent shimmer -translate-y-full pointer-events-none" />

        <div className="space-y-6">
          {/* Natural Language Prompt */}
          <div>
            <label className="block text-sm font-semibold text-zinc-300 mb-2 flex items-center gap-1.5">
              <Sparkles className="w-4 h-4 text-violet-400" />
              Describe your dataset
            </label>
            <input
              type="text"
              required
              disabled={loading}
              placeholder="e.g. 50 images of coca cola cans in the fridge"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="w-full px-4 py-3 rounded-xl border border-zinc-800 bg-zinc-950/70 text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-violet-500/50 transition-all text-base"
            />
            <p className="text-xs text-zinc-500 mt-1">
              You can include the quantity directly in the text prompt (e.g. "100 cats").
            </p>
          </div>

          {/* Sample Image Upload (Optional) */}
          <div className="border border-zinc-800/80 rounded-xl bg-zinc-950/20 p-4">
            <label className="block text-sm font-semibold text-zinc-300 mb-2 flex items-center gap-1.5">
              <ImageIcon className="w-4 h-4 text-violet-400" />
              Upload Sample Image (Optional)
            </label>
            <p className="text-xs text-zinc-500 mb-3">
              Provide a reference image to collect only visual/semantic equivalents (uses CLIP visual similarity).
            </p>
            
            <div className="flex items-center gap-4">
              {!sampleImagePreview ? (
                <label
                  onDragEnter={handleDrag}
                  onDragOver={handleDrag}
                  onDragLeave={handleDrag}
                  onDrop={handleDrop}
                  className={`flex-grow flex flex-col items-center justify-center border-2 border-dashed rounded-xl p-4 cursor-pointer transition-all ${
                    dragActive
                      ? "border-violet-500 bg-violet-950/20 shadow-md shadow-violet-500/10"
                      : "border-zinc-800 hover:border-violet-500/50 bg-zinc-950/50 hover:bg-zinc-950/80"
                  }`}
                >
                  <div className="flex flex-col items-center text-center">
                    <span className="text-xs font-semibold text-zinc-400">
                      {dragActive ? "Drop image here!" : "Drag & drop or click to upload"}
                    </span>
                    <span className="text-[10px] text-zinc-600 mt-0.5">Supports JPG, PNG</span>
                  </div>
                  <input
                    type="file"
                    accept="image/png, image/jpeg, image/jpg"
                    disabled={loading}
                    onChange={handleFileChange}
                    className="hidden"
                  />
                </label>
              ) : (
                <div className="flex items-center gap-4 w-full p-2 bg-zinc-950/50 border border-zinc-800 rounded-xl">
                  <img
                    src={sampleImagePreview}
                    alt="Sample preview"
                    className="w-16 h-16 object-cover rounded-lg border border-zinc-850"
                  />
                  <div className="flex-grow min-w-0">
                    <p className="text-xs font-bold text-zinc-200 truncate">
                      {sampleImage ? sampleImage.name : "Reused Sample Image"}
                    </p>
                    {sampleImage && (
                      <p className="text-[10px] text-zinc-500">{(sampleImage.size / 1024).toFixed(1)} KB</p>
                    )}
                  </div>
                  <button
                    type="button"
                    onClick={handleRemoveFile}
                    className="px-3 py-1.5 text-[10px] font-bold text-red-400 hover:text-red-300 border border-red-500/20 hover:border-red-500/40 rounded-lg hover:bg-red-500/10 transition-colors cursor-pointer"
                  >
                    Remove
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Folder & Deduplication Settings */}
          <div className="border border-zinc-800/80 rounded-xl bg-zinc-950/20 p-4 space-y-4">
            <label className="block text-sm font-semibold text-zinc-300 flex items-center gap-1.5">
              <Folder className="w-4 h-4 text-violet-400" />
              Target Folder &amp; Deduplication Settings
            </label>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Directory Option */}
              <div className="space-y-3">
                <label className="block text-xs font-semibold text-zinc-400">
                  Target Folder Option
                </label>
                <div className="space-y-1">
                  <select
                    disabled={loading}
                    value={folderMode === "auto" ? "__new__" : customFolderName}
                    onChange={(e) => {
                      const val = e.target.value;
                      if (val === "__new__") {
                        setFolderMode("auto");
                        setCustomFolderName("");
                      } else {
                        setFolderMode("manual");
                        setCustomFolderName(val);
                      }
                    }}
                    className="w-full px-3 py-2 rounded-lg border border-zinc-800 bg-zinc-950 text-zinc-200 focus:outline-none focus:ring-1 focus:ring-violet-500/50 text-sm cursor-pointer"
                  >
                    <option value="__new__">Create New Folder (Same as Label Name)</option>
                    {existingFolders.map((folder) => (
                      <option key={folder} value={folder}>
                        {folder} (Append to existing)
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Duplicate policy */}
              <div className="space-y-3">
                <label className="block text-xs font-semibold text-zinc-400">
                  Duplicate Image Filtering
                </label>
                <div className="flex items-center justify-between p-3 border border-zinc-800 bg-zinc-950/40 rounded-xl">
                  <div>
                    <span className="block text-xs font-semibold text-zinc-300">Allow Duplicate Images</span>
                    <span className="block text-[10px] text-zinc-500 mt-0.5">Skip hash-based deduplication</span>
                  </div>
                  <button
                    type="button"
                    disabled={loading}
                    onClick={() => setAllowDuplicates(!allowDuplicates)}
                    className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none cursor-pointer ${
                      allowDuplicates ? "bg-violet-600" : "bg-zinc-800"
                    }`}
                  >
                    <span
                      className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                        allowDuplicates ? "translate-x-4" : "translate-x-1"
                      }`}
                    />
                  </button>
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Image Count Slider + Number Input */}
            <div className="space-y-2">
              <div className="flex justify-between items-center text-sm font-semibold text-zinc-300">
                <span className="flex items-center gap-1.5"><ImageIcon className="w-4 h-4 text-emerald-400" /> Target Count</span>
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    min="1"
                    max="5000"
                    disabled={loading}
                    value={count}
                    onChange={(e) => {
                      const val = Math.min(5000, Math.max(1, parseInt(e.target.value, 10) || 1));
                      setCount(val);
                    }}
                    className="w-20 px-2 py-1 rounded-lg border border-zinc-800 bg-zinc-950 text-violet-400 text-center text-sm font-bold focus:outline-none focus:ring-1 focus:ring-violet-500/50"
                  />
                  <span className="text-zinc-500 text-xs">images</span>
                </div>
              </div>
              <input
                type="range"
                min="1"
                max="5000"
                step="1"
                disabled={loading}
                value={count}
                onChange={(e) => setCount(parseInt(e.target.value, 10) || 1)}
                className="w-full h-2 rounded-lg bg-zinc-800 accent-violet-500 cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-zinc-500">
                <span>1 image</span>
                <span>5000 images (Max target count)</span>
              </div>
            </div>

            {/* Quality Threshold Slider */}
            <div className="space-y-2">
              <div className="flex justify-between items-center text-sm font-semibold text-zinc-300">
                <span className="flex items-center gap-1.5"><Cpu className="w-4 h-4 text-cyan-400" /> CLIP Quality Cutoff</span>
                <span className="text-violet-400">{qualityThreshold}</span>
              </div>
              <input
                type="range"
                min="0.4"
                max="0.9"
                step="0.05"
                disabled={loading}
                value={qualityThreshold}
                onChange={(e) => setQualityThreshold(e.target.value)}
                className="w-full h-2 rounded-lg bg-zinc-800 accent-violet-500 cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-zinc-500">
                <span>0.4 (Relaxed)</span>
                <span>0.9 (Very strict semantic match)</span>
              </div>
            </div>
          </div>

          {/* Toggle Panel for Labeling */}
          <div className="border border-zinc-800/80 rounded-xl bg-zinc-950/20 p-4">
            <div className="flex justify-between items-center mb-4">
              <div className="space-y-0.5">
                <h3 className="text-sm font-semibold text-zinc-200 flex items-center gap-1.5">
                  <Layers className="w-4 h-4 text-violet-400" />
                  Auto-Labeling Pipeline
                </h3>
                <p className="text-xs text-zinc-500">Enable Florence-2 / CLIP zero-shot models</p>
              </div>
              <button
                type="button"
                disabled={loading}
                onClick={() => setLabel(!label)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none ${
                  label ? "bg-violet-600" : "bg-zinc-800"
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    label ? "translate-x-6" : "translate-x-1"
                  }`}
                />
              </button>
            </div>

            {label && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2 border-t border-zinc-800/50 animate-slideDown">
                {/* Target Label List Input */}
                <div className="md:col-span-2 space-y-1">
                  <label className="block text-xs font-semibold text-zinc-300">
                    What label(s) should be given to the collected data?
                  </label>
                  <input
                    type="text"
                    disabled={loading}
                    placeholder="e.g. cup, mug (comma-separated. If left blank, we will automatically extract the class from your query)"
                    value={targetLabels}
                    onChange={(e) => setTargetLabels(e.target.value)}
                    className="w-full px-3.5 py-2.5 rounded-lg border border-zinc-800 bg-zinc-950/60 text-zinc-100 placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-violet-500/30 text-sm transition-all focus:border-violet-500/60"
                  />
                  <p className="text-[10px] text-zinc-500">
                    Separate multiple labels with commas (e.g. "red can, blue can"). These will become the class names in your YOLO txt, COCO JSON, or CSV export.
                  </p>
                </div>

                {/* Labeling Mode */}
                <div>
                  <label className="block text-xs font-semibold text-zinc-400 mb-1.5">
                    Annotation Task Type
                  </label>
                  <select
                    disabled={loading}
                    value={labelType}
                    onChange={(e) => setLabelType(e.target.value)}
                    className="w-full px-3 py-2 rounded-lg border border-zinc-800 bg-zinc-950/80 text-zinc-200 focus:outline-none focus:ring-1 focus:ring-violet-500/50 text-sm cursor-pointer"
                  >
                    <option value="classification">Classification (Zero-Shot CLIP)</option>
                    <option value="detection">Object Detection (Florence-2 BBoxes)</option>
                    <option value="segmentation">Segmentation (Florence-2 Polygon Masks)</option>
                  </select>
                </div>

                {/* Export Format */}
                <div>
                  <label className="block text-xs font-semibold text-zinc-400 mb-1.5">
                    Export Format
                  </label>
                  <select
                    disabled={loading}
                    value={exportFormat}
                    onChange={(e) => setExportFormat(e.target.value)}
                    className="w-full px-3 py-2 rounded-lg border border-zinc-800 bg-zinc-950/80 text-zinc-200 focus:outline-none focus:ring-1 focus:ring-violet-500/50 text-sm cursor-pointer"
                  >
                    <option value="yolo">YOLO txt (Normalized coordinates)</option>
                    <option value="coco">COCO JSON (Standard 2017 annotations)</option>
                    <option value="csv">CSV Sheet (Table summary)</option>
                  </select>
                </div>
              </div>
            )}
          </div>

          {/* Error Message */}
          {error && (
            <div className="flex items-start gap-2.5 p-3 rounded-lg border border-red-500/30 bg-red-500/10 text-red-400 text-sm">
              <ShieldAlert className="w-5 h-5 shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          {/* Submit Button */}
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="w-full flex items-center justify-center gap-2 py-3.5 rounded-xl text-white font-bold bg-gradient-to-r from-violet-600 via-indigo-600 to-violet-700 hover:from-violet-500 hover:to-indigo-500 focus:outline-none focus:ring-2 focus:ring-violet-500/50 transition-all shadow-lg hover:shadow-violet-500/20 disabled:opacity-50 disabled:cursor-not-allowed text-base cursor-pointer"
          >
            {loading ? (
              <>
                <svg className="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Deploying Pipeline...
              </>
            ) : (
              <>
                <Play className="w-5 h-5 fill-current" />
                Start Collection Task
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
export default Home;
