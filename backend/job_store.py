import threading
import os
import json
from typing import Dict, Any, Optional

# Thread-safe in-memory job store with disk persistence
_jobs: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()

OUTPUT_DIR = os.path.abspath(os.getenv("OUTPUT_DIR", "./outputs"))
JOBS_FILE = os.path.join(OUTPUT_DIR, "jobs.json")

def _save_jobs_to_disk():
    try:
        os.makedirs(os.path.dirname(JOBS_FILE), exist_ok=True)
        temp_file = JOBS_FILE + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(_jobs, f, indent=2)
        os.replace(temp_file, JOBS_FILE)
    except Exception as e:
        print(f"Error saving jobs to disk: {e}")

def _load_jobs_from_disk():
    global _jobs
    if not os.path.exists(JOBS_FILE):
        return
    try:
        with open(JOBS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Restore jobs. Mark in-progress/queued jobs as failed.
            for job_id, job in data.items():
                if job.get("status") not in ["done", "failed"]:
                    job["status"] = "failed"
                    job["error"] = "Server was restarted while job was running."
            _jobs = data
    except Exception as e:
        print(f"Error loading jobs from disk: {e}")

# Load jobs on module import
_load_jobs_from_disk()

def create_job(job_id: str, config: Dict[str, Any], search_query: str) -> Dict[str, Any]:
    """Creates a new job with default status and parameters."""
    job_state = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "stage": "Queue",
        "images_collected": 0,
        "images_passed": 0,
        "labeled_count": 0,
        "mock_label_count": 0,
        "error": None,
        "config": config,
        "search_query": search_query,
        "results": [],
        "has_sample_image": False,
        "sample_image_url": None,
        "cancelled": False
    }
    with _lock:
        _jobs[job_id] = job_state
        _save_jobs_to_disk()
    return job_state

def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a job's current state."""
    with _lock:
        job = _jobs.get(job_id)
        if job:
            return dict(job)
        return None

def update_job(job_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Updates a job's state with the provided key-value pairs."""
    with _lock:
        if job_id not in _jobs:
            return None
        _jobs[job_id].update(updates)
        _save_jobs_to_disk()
        return dict(_jobs[job_id])

def get_all_jobs() -> Dict[str, Dict[str, Any]]:
    """Retrieves all jobs in the store."""
    with _lock:
        return {k: dict(v) for k, v in _jobs.items()}

def cancel_job(job_id: str) -> bool:
    """Cancels a job by updating its status to failed with a cancellation message."""
    with _lock:
        if job_id in _jobs:
            if _jobs[job_id]["status"] not in ["done", "failed"]:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["error"] = "Job was cancelled by the user."
                _jobs[job_id]["cancelled"] = True
                _save_jobs_to_disk()
            return True
        return False

def is_job_cancelled(job_id: str) -> bool:
    """Checks if a job has been cancelled."""
    with _lock:
        job = _jobs.get(job_id)
        return job.get("cancelled", False) if job else False
