import os
import json
import logging
import traceback
import asyncio
from typing import Dict, Any, List
from backend.job_store import update_job, get_job, is_job_cancelled
from backend.scraper import scrape_and_download
from backend.filter import filter_dataset
from backend.labeler import label_dataset
from backend.exporter import export_dataset

logger = logging.getLogger("pipeline")

# Concurrency semaphore
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))
pipeline_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

async def run_pipeline(
    job_id: str,
    search_query: str,
    config: Dict[str, Any],
    output_dir: str = "./outputs"
) -> str:
    """
    Main pipeline execution wrapper.
    Receives a pre-created job_id, search_query, and config from the controller.
    Runs stages sequentially, updating the job store along the way.
    Returns the job_id.
    """
    # Wait for the semaphore slot
    async with pipeline_semaphore:
        # Check cancellation before starting
        if is_job_cancelled(job_id):
            logger.info(f"[{job_id}] Job was cancelled before execution started.")
            return job_id
            
        update_job(job_id, {
            "status": "scraping",
            "progress": 5,
            "stage": "Parser",
        })
        
        def progress_callback(downloaded_count):
            # Scale progress up to 45% during download based on the expanded scraper count limit
            scraped_target = config["count"] * 2
            pct = min(45, 5 + int(40 * (downloaded_count / scraped_target)))
            update_job(job_id, {
                "progress": pct,
                "images_collected": downloaded_count
            })

        try:
            # Stage 2: Scraper Engine
            if is_job_cancelled(job_id):
                raise ValueError("Job was cancelled by the user.")
                
            update_job(job_id, {"stage": "Scraper"})
            logger.info(f"[{job_id}] Stage 2: Scraping images for query: '{search_query}'")
            
            collected_count, _ = await scrape_and_download(
                query=search_query,
                count=config["count"] * 2,
                output_dir=output_dir,
                job_id=job_id,
                on_progress_update=progress_callback
            )
            
            if is_job_cancelled(job_id):
                raise ValueError("Job was cancelled by the user.")
                
            if collected_count == 0:
                raise ValueError("No images could be scraped from the web. Scraper failed.")
                
            update_job(job_id, {
                "progress": 45,
                "images_collected": collected_count
            })

            # Stage 3: Quality Filter
            if is_job_cancelled(job_id):
                raise ValueError("Job was cancelled by the user.")
                
            update_job(job_id, {"stage": "Filter", "progress": 50})
            logger.info(f"[{job_id}] Stage 3: Filtering quality and duplicates...")
            
            # Run filter in thread
            passed_count, _ = await asyncio.to_thread(
                filter_dataset,
                job_id=job_id,
                query=search_query,
                quality_threshold=config["quality_threshold"],
                allow_duplicates=config.get("allow_duplicates", False),
                output_dir=output_dir,
                target_count=config["count"]
            )
            
            if is_job_cancelled(job_id):
                raise ValueError("Job was cancelled by the user.")
                
            update_job(job_id, {
                "progress": 65,
                "images_passed": passed_count
            })
            
            if passed_count == 0:
                raise ValueError(f"All {collected_count} scraped images were rejected by quality/relevance filtering.")

            # Stage 4: Auto-Labeler
            labeled_count = 0
            mock_label_count = 0
            if config["label"]:
                if is_job_cancelled(job_id):
                    raise ValueError("Job was cancelled by the user.")
                    
                update_job(job_id, {"stage": "Labeler", "progress": 70})
                logger.info(f"[{job_id}] Stage 4: Auto-labeling images with Florence-2/CLIP...")
                
                labeled_count, mock_label_count, _ = await asyncio.to_thread(
                    label_dataset,
                    job_id=job_id,
                    label_type=config["label_type"],
                    target_labels=config["target_labels"],
                    output_dir=output_dir
                )
                
                if is_job_cancelled(job_id):
                    raise ValueError("Job was cancelled by the user.")
                    
                update_job(job_id, {
                    "progress": 85,
                    "labeled_count": labeled_count,
                    "mock_label_count": mock_label_count
                })
            else:
                logger.info(f"[{job_id}] Stage 4: Skipping labeler as requested.")
                update_job(job_id, {
                    "progress": 85,
                    "labeled_count": 0,
                    "mock_label_count": 0
                })

            # Stage 5: Exporter
            if is_job_cancelled(job_id):
                raise ValueError("Job was cancelled by the user.")
                
            update_job(job_id, {"stage": "Exporter", "progress": 90})
            logger.info(f"[{job_id}] Stage 5: Packaging and zipping dataset...")
            
            zip_path = await asyncio.to_thread(
                export_dataset,
                job_id=job_id,
                export_format=config["export_format"],
                label_type=config["label_type"],
                output_dir=output_dir
            )
            
            if is_job_cancelled(job_id):
                raise ValueError("Job was cancelled by the user.")
                
            # Load up to 20 preview items to save in job store
            preview_list = []
            filtered_dir = os.path.join(output_dir, job_id, "filtered")
            labels_file = os.path.join(output_dir, job_id, "labels", "labels_raw.json")
            
            raw_labels = {}
            if os.path.exists(labels_file):
                with open(labels_file, "r", encoding="utf-8") as f:
                    raw_labels = json_load_resilient(f)
                    
            # List first 20 images
            if os.path.exists(filtered_dir):
                images = sorted([img for img in os.listdir(filtered_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png'))])[:20]
                for img in images:
                    label_info = raw_labels.get(img, {})
                    preview_list.append({
                        "filename": img,
                        "url": f"/outputs/{job_id}/filtered/{img}",
                        "labels_data": label_info
                    })

            # Success update
            update_job(job_id, {
                "status": "done",
                "progress": 100,
                "stage": "Completed",
                "results": preview_list
            })
            logger.info(f"[{job_id}] Pipeline completed successfully.")
            
        except Exception as e:
            err_msg = str(e)
            logger.error(f"[{job_id}] Pipeline failed: {err_msg}")
            logger.error(traceback.format_exc())
            # Do not overwrite job status if it was already marked failed/cancelled by user
            current_job = get_job(job_id)
            if current_job and current_job.get("status") != "failed":
                update_job(job_id, {
                    "status": "failed",
                    "error": err_msg
                })
            
        return job_id

def json_load_resilient(f) -> Dict[str, Any]:
    try:
        return json.load(f)
    except Exception:
        return {}
