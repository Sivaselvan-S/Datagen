import React, { useEffect, useState, useRef } from "react";
import { Download, RefreshCw, RotateCcw, Layers, ShieldCheck, Image as ImageIcon, Sparkles, LayoutGrid, X } from "lucide-react";
import { BACKEND_URL } from "../config";

// Component to dynamically scale and render Florence-2 annotations (bboxes / polygons) on top of the image
function AnnotatedImage({ imageUrl, filename, labelsData, mode }) {
  const [scaling, setScaling] = useState({ scaleX: 1, scaleY: 1, width: 0, height: 0 });
  const [imgLoaded, setImgLoaded] = useState(false);
  const [imgLoadFailed, setImgLoadFailed] = useState(false);
  const containerRef = useRef(null);
  const imgRef = useRef(null);

  const handleImageLoad = () => {
    if (!imgRef.current) return;
    const { clientWidth, clientHeight, naturalWidth, naturalHeight } = imgRef.current;
    if (naturalWidth && naturalHeight) {
      setScaling({
        scaleX: clientWidth / naturalWidth,
        scaleY: clientHeight / naturalHeight,
        width: clientWidth,
        height: clientHeight,
        naturalWidth,
        naturalHeight
      });
      setImgLoaded(true);
    }
  };

  // Re-calculate on window resize
  useEffect(() => {
    const handleResize = () => {
      handleImageLoad();
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [imgLoaded]);

  // Bounding Boxes Rendering
  const renderBboxes = () => {
    if (!imgLoaded || !labelsData || !labelsData.bboxes) return null;
    
    return labelsData.bboxes.map((box, idx) => {
      const left = box.x_min * scaling.scaleX;
      const top = box.y_min * scaling.scaleY;
      const width = (box.x_max - box.x_min) * scaling.scaleX;
      const height = (box.y_max - box.y_min) * scaling.scaleY;
      
      return (
        <div
          key={idx}
          className="absolute border-2 border-violet-500 bg-violet-500/10 rounded-sm pointer-events-none group"
          style={{ left, top, width, height }}
        >
          <span className="absolute -top-5 left-0 bg-violet-600 text-white font-semibold text-[9px] px-1 rounded shadow-md truncate max-w-[120px]">
            {box.label} ({Math.round(box.confidence * 100)}%)
          </span>
        </div>
      );
    });
  };

  // Polygon Masks Rendering (Instance Segmentation)
  const renderPolygons = () => {
    if (!imgLoaded || !labelsData || !labelsData.polygons || !scaling.naturalWidth) return null;

    return (
      <svg
        className="absolute top-0 left-0 w-full h-full pointer-events-none"
        viewBox={`0 0 ${scaling.naturalWidth} ${scaling.naturalHeight}`}
        width={scaling.width}
        height={scaling.height}
      >
        {labelsData.polygons.map((polyItem, idx) => {
          // polyItem.polygons is a List[List[float]] where each list is [x1, y1, x2, y2, ...]
          return polyItem.polygons.map((pts, pIdx) => {
            const pointsString = pts
              .map((val, i) => val)
              .reduce((acc, val, i) => acc + (i % 2 === 0 ? `${val},` : `${val} `), "")
              .trim();
            
            return (
              <polygon
                key={`${idx}-${pIdx}`}
                points={pointsString}
                className="fill-violet-500/30 stroke-violet-500 stroke-2 hover:fill-violet-500/50 transition-colors"
              />
            );
          });
        })}
      </svg>
    );
  };

  if (imgLoadFailed) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center bg-zinc-900 text-zinc-500 p-4 text-center">
        <svg className="w-10 h-10 text-zinc-600 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
        <span className="text-[10px] text-zinc-500 font-bold truncate max-w-full">{filename}</span>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="relative w-full h-full select-none overflow-hidden flex items-center justify-center bg-zinc-950">
      <img
        ref={imgRef}
        src={`${BACKEND_URL}${imageUrl}`}
        alt={filename}
        onLoad={handleImageLoad}
        onError={() => setImgLoadFailed(true)}
        className="max-w-full max-h-full object-contain"
      />
      {mode === "detection" && renderBboxes()}
      {mode === "segmentation" && renderPolygons()}
    </div>
  );
}

export function Results({ jobId, onReset, onRepeat }) {
  const [job, setJob] = useState(null);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);

  useEffect(() => {
    const fetchResults = async () => {
      try {
        // Fetch Job Details
        const jobRes = await fetch(`${BACKEND_URL}/api/jobs/${jobId}/status`);
        if (!jobRes.ok) throw new Error("Failed to load job details.");
        const jobData = await jobRes.json();
        setJob(jobData);

        // Fetch Filter Report File
        const reportRes = await fetch(`${BACKEND_URL}/outputs/${jobId}/filter_report.json`);
        if (reportRes.ok) {
          const reportData = await reportRes.json();
          setReport(reportData);
        }
      } catch (err) {
        console.error(err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchResults();
  }, [jobId]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] gap-3">
        <RefreshCw className="w-10 h-10 text-violet-500 animate-spin" />
        <p className="text-zinc-400 text-sm">Compiling dataset statistics...</p>
      </div>
    );
  }

  if (error || !job) {
    return (
      <div className="max-w-md mx-auto text-center p-6 border border-zinc-800 bg-zinc-950 rounded-2xl">
        <p className="text-red-400 font-bold mb-4">Error loading results</p>
        <p className="text-zinc-500 text-sm mb-6">{error || "Job details are unavailable."}</p>
        <button onClick={onReset} className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-white rounded-lg cursor-pointer transition-all">
          Configure New Job
        </button>
      </div>
    );
  }

  const mode = job.config?.label_type || "detection";
  const datasetZipUrl = `${BACKEND_URL}/api/jobs/${jobId}/download`;

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 animate-fadeIn">
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
        <div>
          <span className="text-[10px] bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-bold px-2.5 py-1 rounded-full uppercase tracking-wider">
            Dataset Ready
          </span>
          <h2 className="text-3xl font-black text-white mt-2">
            Collection Report &amp; Assets
          </h2>
          <p className="text-zinc-400 text-xs mt-1">
            Job reference: <span className="text-zinc-300 font-mono">{jobId}</span>
          </p>
        </div>
        
        <div className="flex gap-3">
          <button
            onClick={onReset}
            className="flex items-center gap-1.5 px-4 py-2.5 rounded-xl border border-zinc-800 bg-zinc-900/60 hover:bg-zinc-800 text-zinc-300 font-bold text-sm transition-all cursor-pointer"
          >
            <RefreshCw className="w-4 h-4" />
            Collect Another
          </button>

          <button
            onClick={() => {
              if (onRepeat && job && job.config) {
                onRepeat({
                  query: job.search_query || job.config.query || "",
                  count: job.config.count || 20,
                  label: job.config.label !== undefined ? job.config.label : true,
                  label_type: job.config.label_type || "detection",
                  export_format: job.config.export_format || "yolo",
                  quality_threshold: job.config.quality_threshold || 0.6,
                  target_labels: job.config.target_labels || "",
                  folder_mode: "manual",
                  custom_folder_name: jobId,
                  allow_duplicates: job.config.allow_duplicates !== undefined ? job.config.allow_duplicates : false,
                  sample_image_url: job.sample_image_url || null
                });
              }
            }}
            className="flex items-center gap-1.5 px-4 py-2.5 rounded-xl border border-zinc-800 bg-violet-950/20 hover:bg-violet-950/40 text-violet-400 font-bold text-sm transition-all cursor-pointer"
          >
            <RotateCcw className="w-4 h-4" />
            Repeat Config
          </button>
          
          <a
            href={datasetZipUrl}
            download
            className="flex items-center gap-1.5 px-5 py-2.5 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white font-bold text-sm transition-all shadow-lg hover:shadow-violet-500/20 cursor-pointer"
          >
            <Download className="w-4 h-4" />
            Download Dataset ZIP
          </a>
        </div>
      </div>

      {/* Statistics and Filtering Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        {/* Quality Checked Card */}
        <div className="glass-panel p-5 rounded-2xl flex items-center gap-4 border-l-4 border-l-emerald-500">
          <div className="p-3 bg-emerald-500/10 rounded-xl text-emerald-400">
            <ShieldCheck className="w-6 h-6" />
          </div>
          <div>
            <span className="block text-[10px] uppercase font-bold tracking-wider text-zinc-500">Passed Filtering</span>
            <span className="text-2xl font-black text-white">{report?.passed || job.images_passed}</span>
            <span className="block text-[10px] text-zinc-400 mt-0.5">Ready for training</span>
          </div>
        </div>

        {/* Duplicates Filtered Card */}
        <div className="glass-panel p-5 rounded-2xl flex items-center gap-4">
          <div>
            <span className="block text-[10px] uppercase font-bold tracking-wider text-zinc-500">Duplicates Removed</span>
            <span className="text-xl font-black text-zinc-300">{report?.duplicates_removed ?? 0}</span>
            <span className="block text-[10px] text-zinc-500 mt-0.5">imagehash dedup</span>
          </div>
        </div>

        {/* Blurry Filtered Card */}
        <div className="glass-panel p-5 rounded-2xl flex items-center gap-4">
          <div>
            <span className="block text-[10px] uppercase font-bold tracking-wider text-zinc-500">Blurry Discarded</span>
            <span className="text-xl font-black text-zinc-300">{report?.blurry_removed ?? 0}</span>
            <span className="block text-[10px] text-zinc-500 mt-0.5">Laplacian &lt; 10.0</span>
          </div>
        </div>

        {/* Irrelevant Filtered Card */}
        <div className="glass-panel p-5 rounded-2xl flex items-center gap-4">
          <div>
            <span className="block text-[10px] uppercase font-bold tracking-wider text-zinc-500">Irrelevant Removed</span>
            <span className="text-xl font-black text-zinc-300">{report?.irrelevant_removed ?? 0}</span>
            <span className="block text-[10px] text-zinc-500 mt-0.5">CLIP similarity cutoff</span>
          </div>
        </div>
      </div>

      {/* Dataset Grid Display */}
        {/* Pagination Calculations */}
        {(() => {
          const resultsPerPage = 20;
          const totalResults = job.results?.length || 0;
          const totalPages = Math.ceil(totalResults / resultsPerPage);
          const paginatedResults = job.results?.slice(
            (currentPage - 1) * resultsPerPage,
            currentPage * resultsPerPage
          ) || [];

          return (
            <>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                {paginatedResults.map((img) => (
                  <div
                    key={img.filename}
                    onClick={() => setSelectedImage(img)}
                    className="glass-panel rounded-2xl overflow-hidden aspect-square border border-zinc-800 bg-zinc-950 flex flex-col justify-between group cursor-pointer hover:border-violet-500/50 hover:shadow-lg hover:shadow-violet-500/5 transition-all"
                  >
                    {/* Image Frame with Annotations */}
                    <div className="w-full flex-1 relative overflow-hidden flex items-center justify-center">
                      <AnnotatedImage
                        imageUrl={img.url}
                        filename={img.filename}
                        labelsData={img.labels_data}
                        mode={mode}
                      />
                      
                      {/* Mode Specific Hover Helper */}
                      <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                        <span className="bg-zinc-900 border border-zinc-700 text-white font-semibold text-xs px-3 py-1.5 rounded-lg shadow-lg">
                          Inspect Coordinates
                        </span>
                      </div>
                    </div>

                    {/* Card Footer */}
                    <div className="px-3.5 py-2.5 border-t border-zinc-900/60 bg-zinc-950/80 flex justify-between items-center">
                      <span className="text-[10px] font-bold text-zinc-400 truncate max-w-[70%]">
                        {img.filename}
                      </span>
                      
                      {mode === "classification" && img.labels_data?.label && (
                        <span className="text-[9px] bg-violet-600 text-white font-bold px-1.5 py-0.5 rounded truncate max-w-[45%]">
                          {img.labels_data.label}
                        </span>
                      )}
                      {mode === "detection" && img.labels_data?.bboxes && (
                        <span className="text-[9px] bg-emerald-600/90 text-white font-bold px-1.5 py-0.5 rounded">
                          {img.labels_data.bboxes.length} Box(es)
                        </span>
                      )}
                      {mode === "segmentation" && img.labels_data?.polygons && (
                        <span className="text-[9px] bg-indigo-600/90 text-white font-bold px-1.5 py-0.5 rounded">
                          {img.labels_data.polygons.length} Mask(s)
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              {/* Pagination Controls */}
              {totalPages > 1 && (
                <div className="flex items-center justify-center gap-4 mt-8 pt-4 border-t border-zinc-900">
                  <button
                    onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                    disabled={currentPage === 1}
                    className="px-3.5 py-2 rounded-xl border border-zinc-800 bg-zinc-900 text-zinc-300 font-semibold text-xs transition-colors hover:bg-zinc-850 hover:text-white disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
                  >
                    Previous
                  </button>
                  <span className="text-zinc-400 text-xs font-bold">
                    Page {currentPage} of {totalPages}
                  </span>
                  <button
                    onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                    disabled={currentPage === totalPages}
                    className="px-3.5 py-2 rounded-xl border border-zinc-800 bg-zinc-900 text-zinc-300 font-semibold text-xs transition-colors hover:bg-zinc-850 hover:text-white disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
                  >
                    Next
                  </button>
                </div>
              )}
            </>
          );
        })()}

      {/* Detailed Inspection Modal */}
      {selectedImage && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/90 backdrop-blur-sm animate-fadeIn">
          <div className="glass-panel w-full max-w-4xl rounded-2xl overflow-hidden shadow-2xl relative flex flex-col md:flex-row max-h-[85vh] border border-zinc-800">
            {/* Close Button */}
            <button
              onClick={() => setSelectedImage(null)}
              className="absolute top-4 right-4 z-10 p-2 rounded-full bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-400 hover:text-white transition-colors cursor-pointer"
            >
              <X className="w-5 h-5" />
            </button>

            {/* Left: Interactive Image view */}
            <div className="flex-[3] relative bg-black flex items-center justify-center overflow-hidden min-h-[300px] md:min-h-0">
              <AnnotatedImage
                imageUrl={selectedImage.url}
                filename={selectedImage.filename}
                labelsData={selectedImage.labels_data}
                mode={mode}
              />
            </div>

            {/* Right: Annotations and Coordinates list */}
            <div className="flex-[2] p-6 bg-zinc-950 flex flex-col justify-between overflow-y-auto border-t md:border-t-0 md:border-l border-zinc-800">
              <div className="space-y-6">
                <div>
                  <h4 className="text-sm font-bold text-zinc-400 uppercase tracking-wider">Asset Inspect</h4>
                  <h3 className="text-xl font-black text-white mt-1">{selectedImage.filename}</h3>
                </div>

                {/* Metadata info */}
                <div>
                  <h5 className="text-xs font-bold text-zinc-500 uppercase tracking-wider mb-2">Structure Annotations</h5>
                  
                  {mode === "classification" && (
                    <div className="p-3.5 rounded-xl border border-zinc-800 bg-zinc-900/40 space-y-1">
                      <span className="text-zinc-500 text-xs">Zero-shot Class</span>
                      <p className="text-white font-extrabold text-sm uppercase">
                        {selectedImage.labels_data?.label || "unlabeled"}
                      </p>
                      {selectedImage.labels_data?.confidence && (
                        <div className="text-[10px] text-zinc-400 mt-1">
                          Confidence: <span className="text-violet-400 font-bold">{Math.round(selectedImage.labels_data.confidence * 100)}%</span>
                        </div>
                      )}
                    </div>
                  )}

                  {mode === "detection" && (
                    <div className="space-y-2.5 max-h-[40vh] overflow-y-auto pr-1">
                      {(!selectedImage.labels_data?.bboxes || selectedImage.labels_data.bboxes.length === 0) ? (
                        <p className="text-zinc-600 text-xs italic">No bounding boxes found.</p>
                      ) : (
                        selectedImage.labels_data.bboxes.map((box, idx) => (
                          <div key={idx} className="p-3 rounded-xl border border-zinc-800 bg-zinc-900/30 flex justify-between items-center">
                            <div>
                              <span className="text-[10px] bg-violet-600 text-white font-bold px-1.5 py-0.5 rounded">
                                {box.label}
                              </span>
                              <div className="text-[10px] font-mono text-zinc-500 mt-1.5">
                                box: [{Math.round(box.x_min)}, {Math.round(box.y_min)}, {Math.round(box.x_max)}, {Math.round(box.y_max)}]
                              </div>
                            </div>
                            <span className="text-xs font-bold text-violet-400">
                              {Math.round(box.confidence * 100)}%
                            </span>
                          </div>
                        ))
                      )}
                    </div>
                  )}

                  {mode === "segmentation" && (
                    <div className="space-y-2.5 max-h-[40vh] overflow-y-auto pr-1">
                      {(!selectedImage.labels_data?.polygons || selectedImage.labels_data.polygons.length === 0) ? (
                        <p className="text-zinc-600 text-xs italic">No polygon masks found.</p>
                      ) : (
                        selectedImage.labels_data.polygons.map((poly, idx) => (
                          <div key={idx} className="p-3 rounded-xl border border-zinc-800 bg-zinc-900/30">
                            <div className="flex justify-between items-center">
                              <span className="text-[10px] bg-indigo-600 text-white font-bold px-1.5 py-0.5 rounded">
                                {poly.label}
                              </span>
                              <span className="text-xs font-bold text-indigo-400">
                                {Math.round(poly.confidence * 100)}%
                              </span>
                            </div>
                            <div className="text-[9px] font-mono text-zinc-500 mt-2 truncate">
                              poly-points: {JSON.stringify(poly.polygons)}
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  )}
                </div>
              </div>

              <div className="pt-6 border-t border-zinc-900">
                <button
                  onClick={() => setSelectedImage(null)}
                  className="w-full py-2.5 bg-zinc-900 border border-zinc-800 hover:bg-zinc-800 text-white font-bold text-sm rounded-xl cursor-pointer transition-colors"
                >
                  Close Inspection
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
export default Results;
