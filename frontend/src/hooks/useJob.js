import { useState, useEffect, useRef } from "react";

const BACKEND_URL = "http://localhost:8000";

export function useJob(jobId, onComplete = null) {
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const pollTimer = useRef(null);

  const fetchStatus = async () => {
    if (!jobId) return;
    try {
      const res = await fetch(`${BACKEND_URL}/api/jobs/${jobId}/status`);
      if (!res.ok) {
        throw new Error(`Failed to fetch job status: ${res.statusText}`);
      }
      const data = await res.json();
      setJob(data);
      
      // Stop polling if the job completed or failed
      if (data.status === "done" || data.status === "failed") {
        if (pollTimer.current) {
          clearInterval(pollTimer.current);
          pollTimer.current = null;
        }
        if (data.status === "done" && onComplete) {
          onComplete(data);
        }
      }
    } catch (err) {
      console.error("Error polling job status:", err);
      setError(err.message);
    }
  };

  useEffect(() => {
    if (!jobId) {
      setJob(null);
      setError(null);
      return;
    }

    setLoading(true);
    setError(null);

    // Initial fetch
    fetchStatus().finally(() => setLoading(false));

    // Poll every 2 seconds
    pollTimer.current = setInterval(fetchStatus, 2000);

    return () => {
      if (pollTimer.current) {
        clearInterval(pollTimer.current);
        pollTimer.current = null;
      }
    };
  }, [jobId]);

  return { job, error, loading, refetch: fetchStatus };
}
export default useJob;
