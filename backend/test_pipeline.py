import os
import asyncio
import logging
from dotenv import load_dotenv

# Configure basic logging for test script
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_pipeline")

load_dotenv()

# Import pipeline and singletons
from backend.models.clip_filter import clip_filter_instance
from backend.models.florence_label import florence_labeler_instance
from backend.pipeline import run_pipeline
import backend.job_store as job_store

async def main():
    logger.info("Initializing models...")
    # Initialize CLIP model
    clip_filter_instance.initialize()
    
    # Initialize Florence-2 model
    florence_labeler_instance.initialize()
    
    from backend.input_parser import parse_input
    
    logger.info("Starting integration test job...")
    query = "5 images of a white cup"
    count = 5
    label = True
    label_type = "detection"
    export_format = "yolo"
    quality_threshold = 0.4
    target_labels = "cup, mug"
    
    job_id, search_query, config = parse_input(
        query=query,
        count=count,
        label=label,
        label_type=label_type,
        export_format=export_format,
        quality_threshold=quality_threshold,
        target_labels=target_labels
    )
    
    job_store.create_job(job_id, config, search_query)
    
    await run_pipeline(
        job_id=job_id,
        search_query=search_query,
        config=config,
        output_dir="./outputs"
    )
    
    logger.info(f"Pipeline running. Job ID: {job_id}")
    
    # Poll job store for status
    while True:
        status_info = job_store.get_job(job_id)
        if not status_info:
            logger.error("Job not found in store!")
            break
            
        status = status_info.get("status")
        progress = status_info.get("progress")
        stage = status_info.get("stage")
        collected = status_info.get("images_collected")
        passed = status_info.get("images_passed")
        
        logger.info(f"Job Status: {status} | Stage: {stage} | Progress: {progress}% | Collected: {collected} | Passed: {passed}")
        
        if status in ["done", "failed"]:
            if status == "done":
                logger.info("Integration test PASSED!")
                # Show results preview summary
                results = status_info.get("results", [])
                logger.info(f"Job preview items (total {len(results)}):")
                for item in results:
                    logger.info(f" - {item['filename']}: {item['labels_data']}")
                    
                # Check that ZIP file was created
                zip_path = f"./outputs/{job_id}/dataset_{job_id}.zip"
                if os.path.exists(zip_path):
                    logger.info(f"ZIP file successfully created at: {os.path.abspath(zip_path)} (Size: {os.path.getsize(zip_path)} bytes)")
                else:
                    logger.error(f"ZIP file NOT found at {zip_path}!")
            else:
                logger.error(f"Integration test FAILED with error: {status_info.get('error')}")
            break
            
        await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())
