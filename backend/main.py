import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import os
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional, Any
from fastapi import FastAPI, BackgroundTasks, HTTPException, status, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("server.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("server")

# Import backend modules
from backend.models.clip_filter import clip_filter_instance
from backend.models.florence_label import florence_labeler_instance
import backend.job_store as job_store
from backend.input_parser import parse_input
from backend.pipeline import run_pipeline
from backend.exporter import multiply_job_dataset


# Ensure outputs directory exists
output_dir = os.path.abspath(os.getenv("OUTPUT_DIR", "./outputs"))
os.makedirs(output_dir, exist_ok=True)

# LifeSpan Context for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    logger.info("Initializing models on startup...")
    
    # Load CLIP model (used for relevance filtering & classification)
    try:
        clip_filter_instance.initialize()
    except Exception as e:
        logger.error(f"Failed to load CLIP singleton on startup: {e}. relevance checks will be skipped.")
        
    # Load Florence-2 model (used for bounding boxes & segmentation)
    try:
        florence_labeler_instance.initialize()
    except Exception as e:
        logger.error(f"Failed to load Florence-2 singleton on startup: {e}. auto-labeling (OD/Seg) will use mock fallbacks.")
        
    yield
    # Shutdown actions
    logger.info("Server shutting down.")

app = FastAPI(
    title="AI Dataset Collector API",
    description="Backend pipeline to scrape, filter, auto-label, and export datasets.",
    lifespan=lifespan
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # During development, allow all. In prod, restrict.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount outputs directory as static files
app.mount("/outputs", StaticFiles(directory=output_dir), name="outputs")

# Pydantic schemas
class JobCreateRequest(BaseModel):
    query: str = Field(..., example="100 images of lays green flavour packet")
    count: int = Field(default=20, ge=1, le=5000)
    label: bool = Field(default=True)
    label_type: str = Field(default="detection", description="classification | detection | segmentation")
    export_format: str = Field(default="yolo", description="yolo | coco | csv")
    quality_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    target_labels: Optional[str] = Field(default="", description="Comma-separated target labels")

@app.get("/api/jobs")
async def list_jobs():
    """Lists all available job directories in the outputs folder."""
    if not os.path.exists(output_dir):
        return []
    jobs = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path):
            # Check if it contains raw folder to ensure it is a valid job directory
            if os.path.exists(os.path.join(item_path, "raw")):
                jobs.append(item)
    return sorted(jobs)

def str_to_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes", "on")
    return bool(val)

@app.post("/api/jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    background_tasks: BackgroundTasks,
    query: str = Form(...),
    count: int = Form(20),
    label: Any = Form(True),
    label_type: str = Form("detection"),
    export_format: str = Form("yolo"),
    quality_threshold: float = Form(0.6),
    target_labels: Optional[str] = Form(""),
    folder_mode: str = Form("auto"),
    custom_folder_name: Optional[str] = Form(None),
    allow_duplicates: Any = Form(False),
    sample_image: Optional[UploadFile] = File(None),
    sample_image_url: Optional[str] = Form(None)
):
    """
    Submits a new dataset collection job.
    The job is queued and runs asynchronously in the background.
    """
    try:
        override_job_id = custom_folder_name if folder_mode == "manual" and custom_folder_name else None
        
        # Step 1: Parse input query and parameters (Stage 1)
        job_id, search_query, config = parse_input(
            query=query,
            count=count,
            label=str_to_bool(label),
            label_type=label_type,
            export_format=export_format,
            quality_threshold=quality_threshold,
            target_labels=target_labels,
            allow_duplicates=str_to_bool(allow_duplicates),
            override_job_id=override_job_id
        )
        
        # Check if job is already running or queued
        existing_job = job_store.get_job(job_id)
        if existing_job and existing_job.get("status") not in ["done", "failed"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"A job targeting the folder '{job_id}' is already running or queued."
            )
        
        # Step 2: Initialize in-memory job state
        job_store.create_job(job_id, config, search_query)
        
        # Save sample image if provided
        if sample_image:
            job_dir = os.path.join(output_dir, job_id)
            os.makedirs(job_dir, exist_ok=True)
            
            _, ext = os.path.splitext(sample_image.filename)
            ext_lower = ext.lower()
            if ext_lower not in [".jpg", ".jpeg", ".png"]:
                ext_lower = ".jpg"
                
            dest_path = os.path.join(job_dir, f"sample_image{ext_lower}")
            
            contents = await sample_image.read()
            with open(dest_path, "wb") as f:
                f.write(contents)
                
            logger.info(f"Saved sample image for job {job_id} to {dest_path}")
            
            # Update job state with sample image info
            job_store.update_job(job_id, {
                "has_sample_image": True,
                "sample_image_url": f"/outputs/{job_id}/sample_image{ext_lower}"
            })
        elif sample_image_url:
            # Clean and verify path to prevent directory traversal
            cleaned_url = sample_image_url.lstrip("/")
            if cleaned_url.startswith("outputs/"):
                src_path = os.path.abspath(cleaned_url)
                # Verify that it resides under our outputs folder
                if src_path.startswith(output_dir) and os.path.exists(src_path):
                    job_dir = os.path.join(output_dir, job_id)
                    os.makedirs(job_dir, exist_ok=True)
                    
                    _, ext = os.path.splitext(src_path)
                    dest_path = os.path.join(job_dir, f"sample_image{ext.lower()}")
                    
                    import shutil
                    shutil.copy2(src_path, dest_path)
                    logger.info(f"Copied existing sample image from {src_path} to {dest_path}")
                    
                    # Update job state with sample image info
                    job_store.update_job(job_id, {
                        "has_sample_image": True,
                        "sample_image_url": f"/outputs/{job_id}/sample_image{ext.lower()}"
                    })
        
        # Step 3: Trigger background task
        background_tasks.add_task(
            run_pipeline,
            job_id=job_id,
            search_query=search_query,
            config=config,
            output_dir=output_dir
        )
        
        return {"job_id": job_id, "status": "queued"}
    except Exception as e:
        logger.error(f"Error submitting job request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")

@app.get("/api/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """Retrieves current progress and metrics for a job."""
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/api/jobs/{job_id}/preview")
async def get_job_preview(job_id: str):
    """
    Dynamically loads and returns up to 20 images from the post-filter
    (and post-labeled) folder for live rendering.
    """
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    filtered_dir = os.path.join(output_dir, job_id, "filtered")
    labels_file = os.path.join(output_dir, job_id, "labels", "labels_raw.json")
    
    preview_images = []
    if os.path.exists(filtered_dir):
        # Read the files currently in the filtered folder
        images = sorted([
            f for f in os.listdir(filtered_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])[:20]
        
        # Read current labels if available
        labels_data = {}
        if os.path.exists(labels_file):
            try:
                with open(labels_file, "r", encoding="utf-8") as f:
                    labels_data = json.load(f)
            except Exception:
                pass
                
        for filename in images:
            img_labels = labels_data.get(filename, {})
            preview_images.append({
                "filename": filename,
                "url": f"/outputs/{job_id}/filtered/{filename}",
                "labels_data": img_labels
            })
            
    return {"images": preview_images}

@app.get("/api/jobs/{job_id}/download")
async def download_dataset(job_id: str):
    """Serves the generated dataset ZIP file for download."""
    zip_path = os.path.join(output_dir, job_id, f"dataset_{job_id}.zip")
    if not os.path.exists(zip_path):
        # Check if the job failed or is still running
        job = job_store.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] == "failed":
            raise HTTPException(
                status_code=400, 
                detail=f"Job failed, no dataset available. Error: {job['error']}"
            )
        raise HTTPException(
            status_code=400, 
            detail=f"Dataset is not ready for download. Current status: {job['status']}"
        )
        
    return FileResponse(
        zip_path, 
        media_type="application/zip", 
        filename=f"dataset_{job_id}.zip"
    )

@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancels a running job."""
    success = job_store.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "cancelled"}


@app.post("/api/jobs/{job_id}/multiply")
async def multiply_job(job_id: str, multiplier: int = Form(...), mode: str = Form("copy")):
    """
    Multiplies the dataset for a given job ID.
    """
    try:
        total_images, zip_path = multiply_job_dataset(job_id, multiplier, mode, output_dir)
        return {
            "job_id": job_id,
            "total_images": total_images,
            "zip_url": f"/api/jobs/{job_id}/download"
        }
    except ValueError as e:
        logger.error(f"Validation error multiplying job dataset '{job_id}': {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error multiplying job dataset '{job_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to multiply dataset: {str(e)}")


@app.get("/api/jobs/{job_id}/images")
async def get_job_images(job_id: str, page: int = 1, per_page: int = 24):
    """
    Returns a paginated list of images in the filtered/ folder for a job,
    along with their current labels from labels_raw.json.
    Works for any job folder on disk, even those not tracked in the job store.
    """
    job_dir = os.path.join(output_dir, job_id)
    filtered_dir = os.path.join(job_dir, "filtered")
    labels_file = os.path.join(job_dir, "labels", "labels_raw.json")

    if not os.path.exists(filtered_dir):
        raise HTTPException(status_code=404, detail=f"No filtered folder found for job '{job_id}'.")

    # Load labels
    labels_data = {}
    if os.path.exists(labels_file):
        try:
            with open(labels_file, "r", encoding="utf-8") as f:
                labels_data = json.load(f)
        except Exception:
            pass

    # Sorted image list
    all_images = sorted([
        f for f in os.listdir(filtered_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    total = len(all_images)

    # Pagination
    per_page = max(1, min(per_page, 100))
    page = max(1, page)
    start = (page - 1) * per_page
    end = start + per_page
    page_images = all_images[start:end]

    images = [
        {
            "filename": fname,
            "url": f"/outputs/{job_id}/filtered/{fname}",
            "labels_data": labels_data.get(fname, {})
        }
        for fname in page_images
    ]

    # Try to read config from job store for convenience
    job = job_store.get_job(job_id)
    config = job.get("config", {}) if job else {}

    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": max(1, (total + per_page - 1) // per_page),
        "images": images,
        "label_type": config.get("label_type", "detection"),
        "export_format": config.get("export_format", "yolo"),
        "target_labels": config.get("target_labels", ""),
    }


@app.post("/api/jobs/{job_id}/relabel")
async def relabel_job(
    job_id: str,
    label_type: str = Form("detection"),
    target_labels: str = Form(""),
    export_format: str = Form("yolo"),
):
    """
    Re-runs labeling on the already-filtered images of a job with new parameters,
    then re-exports the dataset ZIP. Updates labels_raw.json and job store in-place.
    """
    from backend.labeler import label_dataset
    from backend.exporter import export_dataset

    job_dir = os.path.join(output_dir, job_id)
    filtered_dir = os.path.join(job_dir, "filtered")

    if not os.path.exists(filtered_dir):
        raise HTTPException(status_code=404, detail=f"No filtered folder found for job '{job_id}'.")

    image_files = [
        f for f in os.listdir(filtered_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    if not image_files:
        raise HTTPException(status_code=400, detail="No images found in the filtered folder.")

    try:
        # Parse target labels
        parsed_labels = [t.strip() for t in target_labels.split(",") if t.strip()]

        logger.info(f"Starting relabel for job '{job_id}': type={label_type}, labels={parsed_labels}, format={export_format}")

        # Re-run labeling — writes a new labels_raw.json
        labeled_count, mock_count, raw_labels_file = label_dataset(
            job_id=job_id,
            label_type=label_type,
            target_labels=parsed_labels,
            output_dir=output_dir
        )

        # Re-export ZIP
        zip_path = export_dataset(job_id, export_format, label_type, output_dir)

        # Reload the freshly written labels
        labels_data = {}
        if os.path.exists(raw_labels_file):
            with open(raw_labels_file, "r", encoding="utf-8") as f:
                labels_data = json.load(f)

        # Rebuild results[] list for the job store
        all_images = sorted([
            f for f in os.listdir(filtered_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        updated_results = [
            {
                "filename": fname,
                "url": f"/outputs/{job_id}/filtered/{fname}",
                "labels_data": labels_data.get(fname, {})
            }
            for fname in all_images
        ]

        # Update job store config + results
        job = job_store.get_job(job_id)
        if job:
            updated_config = dict(job.get("config", {}))
            updated_config.update({
                "label_type": label_type,
                "target_labels": target_labels,
                "export_format": export_format,
            })
            job_store.update_job(job_id, {
                "config": updated_config,
                "labeled_count": labeled_count,
                "mock_label_count": mock_count,
                "results": updated_results,
            })
        else:
            # Job not in store (e.g. server was restarted) — create a minimal entry
            job_store._jobs[job_id] = {
                "job_id": job_id,
                "status": "done",
                "config": {
                    "label_type": label_type,
                    "target_labels": target_labels,
                    "export_format": export_format,
                },
                "labeled_count": labeled_count,
                "images_passed": len(all_images),
                "results": updated_results,
            }
            job_store._save_jobs_to_disk()

        logger.info(f"Relabel complete for job '{job_id}': {labeled_count} images labeled.")

        return {
            "job_id": job_id,
            "labeled_count": labeled_count,
            "mock_count": mock_count,
            "zip_url": f"/api/jobs/{job_id}/download",
        }

    except Exception as e:
        logger.error(f"Relabel failed for job '{job_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Relabeling failed: {str(e)}")


@app.patch("/api/jobs/{job_id}/labels/{filename}")
async def patch_image_label(job_id: str, filename: str, label_data: dict):
    """
    Manually updates the label entry for a single image in labels_raw.json.
    Also updates the job store results[] for that image so the UI reflects the change immediately.
    """
    job_dir = os.path.join(output_dir, job_id)
    labels_file = os.path.join(job_dir, "labels", "labels_raw.json")

    if not os.path.exists(job_dir):
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    # Sanitize filename — no path traversal
    filename = os.path.basename(filename)
    img_path = os.path.join(job_dir, "filtered", filename)
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail=f"Image '{filename}' not found in filtered folder.")

    # Load existing labels
    labels_data = {}
    if os.path.exists(labels_file):
        try:
            with open(labels_file, "r", encoding="utf-8") as f:
                labels_data = json.load(f)
        except Exception:
            pass

    # Overwrite this image's labels entry
    labels_data[filename] = label_data

    # Atomic write
    os.makedirs(os.path.dirname(labels_file), exist_ok=True)
    temp_file = labels_file + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(labels_data, f, indent=2)
    os.replace(temp_file, labels_file)

    # Update the specific image entry in job store results[]
    job = job_store.get_job(job_id)
    if job:
        results = list(job.get("results", []))
        for item in results:
            if item.get("filename") == filename:
                item["labels_data"] = label_data
                break
        job_store.update_job(job_id, {"results": results})

    logger.info(f"Manual label patch applied for job '{job_id}', image '{filename}'.")
    return {"ok": True, "filename": filename}
