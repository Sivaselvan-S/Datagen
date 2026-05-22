import React, { useState, useEffect } from "react";
import Home from "./pages/Home";
import Progress from "./pages/Progress";
import Results from "./pages/Results";
import { Augment } from "./pages/Augment";
import { Relabel } from "./pages/Relabel";
import { Sparkles, Database, X } from "lucide-react";

export function App() {
  const [screen, setScreen] = useState("home"); // "home" | "progress" | "results"
  const [jobId, setJobId] = useState(null);
  const [showDocsModal, setShowDocsModal] = useState(false);

  // Sync state with location.hash to support native history navigation
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash || "#home";
      if (hash.startsWith("#progress/")) {
        const id = hash.replace("#progress/", "");
        setJobId(id);
        setScreen("progress");
      } else if (hash.startsWith("#results/")) {
        const id = hash.replace("#results/", "");
        setJobId(id);
        setScreen("results");
      } else if (hash === "#augment") {
        setJobId(null);
        setScreen("augment");
      } else if (hash === "#relabel") {
        setJobId(null);
        setScreen("relabel");
      } else {
        setJobId(null);
        setScreen("home");
      }
    };

    // Initialize state from hash on mount
    handleHashChange();

    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, []);

  const [repeatConfig, setRepeatConfig] = useState(null);

  const handleStartJob = (id) => {
    setRepeatConfig(null);
    window.location.hash = `#progress/${id}`;
  };

  const handleJobComplete = () => {
    if (jobId) {
      window.location.hash = `#results/${jobId}`;
    }
  };

  const handleCancelOrReset = () => {
    setRepeatConfig(null);
    window.location.hash = "#home";
  };

  const handleRepeatJob = (config) => {
    setRepeatConfig(config);
    window.location.hash = "#home";
  };

  return (
    <div className="min-h-screen grid-bg relative flex flex-col justify-between">
      {/* Background radial overlays */}
      <div className="absolute inset-0 bg-zinc-950/20 pointer-events-none" />

      {/* Main Header */}
      <header className="border-b border-zinc-800/80 bg-zinc-950/40 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-6xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2.5 cursor-pointer" onClick={handleCancelOrReset}>
            <div className="p-2 rounded-xl bg-violet-600/10 border border-violet-500/20 text-violet-400">
              <Database className="w-5 h-5" />
            </div>
            <div>
              <span className="font-black text-white text-base tracking-tight block">
                DATAGEN
              </span>
              <span className="text-[10px] text-zinc-500 uppercase tracking-wider block font-bold -mt-1">
                Vision Engine v1.0
              </span>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex items-center gap-1 bg-zinc-950/60 p-1 rounded-xl border border-zinc-800/80">
            <button
              onClick={() => { window.location.hash = "#home"; }}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-black tracking-wide uppercase transition-all cursor-pointer ${
                screen === "home" || screen === "progress" || screen === "results"
                  ? "bg-violet-600/20 text-violet-300 border border-violet-500/30"
                  : "text-zinc-500 hover:text-zinc-300 border border-transparent"
              }`}
            >
              Collect
            </button>
            <button
              onClick={() => { window.location.hash = "#augment"; }}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-black tracking-wide uppercase transition-all cursor-pointer ${
                screen === "augment"
                  ? "bg-violet-600/20 text-violet-300 border border-violet-500/30"
                  : "text-zinc-500 hover:text-zinc-300 border border-transparent"
              }`}
            >
              Expand
            </button>
            <button
              onClick={() => { window.location.hash = "#relabel"; }}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-black tracking-wide uppercase transition-all cursor-pointer ${
                screen === "relabel"
                  ? "bg-amber-600/20 text-amber-300 border border-amber-500/30"
                  : "text-zinc-500 hover:text-zinc-300 border border-transparent"
              }`}
            >
              Relabel
            </button>
          </nav>

          <div className="flex items-center gap-4">
            <span className="text-zinc-500 text-xs hidden sm:inline">
              FastAPI + React Pipeline
            </span>
            <div className="h-4 w-px bg-zinc-800 hidden sm:block" />
            <button
              onClick={() => setShowDocsModal(true)}
              className="text-xs text-zinc-400 hover:text-white transition-colors bg-transparent border-0 cursor-pointer focus:outline-none"
            >
              Docs
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-grow flex items-center justify-center py-6">
        <div className="w-full">
          {screen === "home" && (
            <Home 
              onSubmit={handleStartJob} 
              initialConfig={repeatConfig} 
              onClearRepeat={() => setRepeatConfig(null)}
            />
          )}
          {screen === "augment" && (
            <Augment />
          )}
          {screen === "relabel" && (
            <Relabel />
          )}
          {screen === "progress" && (
            <Progress
              jobId={jobId}
              onComplete={handleJobComplete}
              onCancel={handleCancelOrReset}
            />
          )}
          {screen === "results" && (
            <Results 
              jobId={jobId} 
              onReset={handleCancelOrReset} 
              onRepeat={handleRepeatJob}
            />
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-900 bg-zinc-950/50 py-4 text-center text-xs text-zinc-600">
        <div className="max-w-6xl mx-auto px-4 flex flex-col sm:flex-row justify-between items-center gap-2">
          <span>&copy; 2026 Datagen. All rights reserved.</span>
          <span className="flex items-center gap-1">
            Built with <Sparkles className="w-3.5 h-3.5 text-violet-500 fill-current" /> using Florence-2 &amp; CLIP
          </span>
        </div>
      </footer>

      {/* Documentation Modal */}
      {showDocsModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/90 backdrop-blur-sm animate-fadeIn">
          <div className="glass-panel w-full max-w-3xl rounded-2xl p-6 md:p-8 shadow-2xl relative max-h-[85vh] overflow-y-auto border border-zinc-800 flex flex-col justify-between">
            {/* Close Button */}
            <button
              onClick={() => setShowDocsModal(false)}
              className="absolute top-4 right-4 z-10 p-2 rounded-full bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-400 hover:text-white transition-colors cursor-pointer"
            >
              <X className="w-4 h-4" />
            </button>

            <div className="space-y-6">
              <div>
                <span className="text-[10px] bg-violet-500/10 border border-violet-500/30 text-violet-300 font-bold px-2.5 py-1 rounded-full uppercase tracking-wider">
                  Vision Engine Guide
                </span>
                <h2 className="text-3xl font-black text-white mt-2">
                  Florence-2 &amp; CLIP Pipeline
                </h2>
                <p className="text-zinc-400 text-sm mt-1">
                  Learn how the zero-shot dataset auto-labeling and visual grounding engine works under the hood.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4 border-t border-zinc-800/80">
                {/* Florence-2 */}
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-violet-400 font-black text-base">
                    <Sparkles className="w-5 h-5" />
                    Florence-2 Auto-Labeler
                  </div>
                  <p className="text-zinc-400 text-xs leading-relaxed">
                    Microsoft's vision model is utilized to generate structural annotations based on the dataset type selected:
                  </p>
                  <ul className="space-y-2 text-xs text-zinc-500 list-disc pl-4">
                    <li>
                      <strong className="text-zinc-300">Object Detection:</strong> Prompts the model for Phrase Grounding matching target labels. Bounding boxes are localized, NMS-filtered, and visual confidence is calculated.
                    </li>
                    <li>
                      <strong className="text-zinc-300">Referring Expression Segmentation:</strong> Localizes instance-level masks for targeted descriptions, supporting nested items like green Lay's packets or red cans.
                    </li>
                  </ul>
                </div>

                {/* CLIP Filtering */}
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-emerald-400 font-black text-base">
                    <Database className="w-5 h-5" />
                    CLIP Semantic Guard
                  </div>
                  <p className="text-zinc-400 text-xs leading-relaxed">
                    OpenAI's CLIP model checks image/text alignment and visual similarity to maintain high-quality labels:
                  </p>
                  <ul className="space-y-2 text-xs text-zinc-500 list-disc pl-4">
                    <li>
                      <strong className="text-zinc-300">Relevance Cutoff:</strong> Compares scraped images to search phrases. Images below the quality threshold are automatically discarded.
                    </li>
                    <li>
                      <strong className="text-zinc-300">Visual Similarity Anchor:</strong> If you upload a reference image, CLIP filters out any crawled images that deviate semantically or visually from your sample.
                    </li>
                  </ul>
                </div>
              </div>

              {/* Usage Best Practices */}
              <div className="p-4 rounded-xl border border-zinc-800 bg-zinc-950/50 space-y-2 text-xs text-zinc-400 leading-relaxed">
                <span className="font-bold text-zinc-200 block">Pro Tips for Best Quality Results:</span>
                <p>
                  1. Keep target classes specific (e.g. use "coca cola bottle" instead of just "bottle" for high visual classification).
                </p>
                <p>
                  2. For bounding boxes/segmentation, list comma-separated classes that actually appear (e.g. "can, label").
                </p>
                <p>
                  3. When using a reference image, set the CLIP cutoff to ~0.55-0.65 to ensure strict visual matching.
                </p>
              </div>

              <div className="pt-4 flex justify-end">
                <button
                  onClick={() => setShowDocsModal(false)}
                  className="px-5 py-2 bg-violet-600 hover:bg-violet-500 text-white font-bold text-xs rounded-xl cursor-pointer transition-colors"
                >
                  Got it
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
