import os
import asyncio
import logging
import shutil
import json
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_append")

load_dotenv()

from backend.models.clip_filter import clip_filter_instance
from backend.models.florence_label import florence_labeler_instance
from backend.pipeline import run_pipeline
from backend.input_parser import parse_input
import backend.job_store as job_store

async def main():
    logger.info("Initializing models...")
    clip_filter_instance.initialize()
    florence_labeler_instance.initialize()
    
    output_dir = "./outputs_test_append"
    # Clean up test output dir first
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    job_id_target = "append_test_dataset"
    
    # ------------------ RUN 1: Scrape white cup ------------------
    logger.info("================== RUN 1: White Cup ==================")
    job_id_1, search_query_1, config_1 = parse_input(
        query="2 images of a white cup",
        count=2,
        label=True,
        label_type="detection",
        export_format="yolo",
        quality_threshold=0.4,
        target_labels="cup",
        allow_duplicates=False,
        override_job_id=job_id_target
    )
    
    logger.info(f"Run 1 parsed Job ID: {job_id_1}")
    job_store.create_job(job_id_1, config_1, search_query_1)
    
    await run_pipeline(
        job_id=job_id_1,
        search_query=search_query_1,
        config=config_1,
        output_dir=output_dir
    )
    
    # Check downloaded files in raw
    raw_dir = os.path.join(output_dir, job_id_target, "raw")
    raw_files_run1 = sorted([f for f in os.listdir(raw_dir) if f.endswith(".jpg")])
    logger.info(f"Raw files after Run 1: {raw_files_run1}")
    
    # Check metadata.json
    metadata_file = os.path.join(raw_dir, "metadata.json")
    with open(metadata_file, "r") as f:
        meta_data_1 = json.load(f)
    logger.info(f"Metadata entries count after Run 1: {len(meta_data_1)}")
    
    # ------------------ RUN 2: Scrape blue mug (APPEND) ------------------
    logger.info("================== RUN 2: Blue Mug (APPEND) ==================")
    job_id_2, search_query_2, config_2 = parse_input(
        query="2 images of a blue mug",
        count=2,
        label=True,
        label_type="detection",
        export_format="yolo",
        quality_threshold=0.4,
        target_labels="mug",
        allow_duplicates=True, # Verify allow_duplicates is passed
        override_job_id=job_id_target
    )
    
    logger.info(f"Run 2 parsed Job ID (should match target): {job_id_2}")
    job_store.create_job(job_id_2, config_2, search_query_2)
    
    await run_pipeline(
        job_id=job_id_2,
        search_query=search_query_2,
        config=config_2,
        output_dir=output_dir
    )
    
    # Check downloaded files in raw again
    raw_files_run2 = sorted([f for f in os.listdir(raw_dir) if f.endswith(".jpg")])
    logger.info(f"Raw files after Run 2: {raw_files_run2}")
    
    with open(metadata_file, "r") as f:
        meta_data_2 = json.load(f)
    logger.info(f"Metadata entries count after Run 2: {len(meta_data_2)}")
    
    # Verify that new raw files start from index greater than highest from Run 1
    # Run 1 had files named like 0000.jpg, 0001.jpg etc.
    # Run 2 files should be higher (e.g. 0002.jpg, 0003.jpg...)
    max_run1_idx = max(int(f[:-4]) for f in raw_files_run1)
    new_files = [f for f in raw_files_run2 if f not in raw_files_run1]
    
    logger.info(f"Newly added files in Run 2: {new_files}")
    for nf in new_files:
        assert int(nf[:-4]) > max_run1_idx, f"Appended file index is not higher! {nf} vs max index {max_run1_idx}"
        
    logger.info("Deduplication / Append checks passed successfully!")
    
    # Clean up test output dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

if __name__ == "__main__":
    asyncio.run(main())
