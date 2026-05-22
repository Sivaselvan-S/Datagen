import os
import sys
import json
import shutil
import asyncio
from typing import Optional

# Add root folder to sys.path to resolve backend imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.main import create_job
import backend.job_store as job_store

# Mock background tasks class
class MockBackgroundTasks:
    def __init__(self):
        self.tasks = []
    def add_task(self, func, *args, **kwargs):
        self.tasks.append((func, args, kwargs))

async def test_repeat_endpoint():
    print("Testing create_job endpoint with sample_image_url...")

    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Create a dummy first job directory with a sample image
    job1_id = "job_one_source"
    job1_dir = os.path.join(output_dir, job1_id)
    os.makedirs(job1_dir, exist_ok=True)
    
    sample_image_src = os.path.join(job1_dir, "sample_image.jpg")
    with open(sample_image_src, "w") as f:
        f.write("dummy-image-bytes")

    # Clean up second job directory if exists
    job2_id = "job_two_repeated"
    job2_dir = os.path.join(output_dir, job2_id)
    if os.path.exists(job2_dir):
        shutil.rmtree(job2_dir)

    # Create Mock Background Tasks
    bg_tasks = MockBackgroundTasks()

    # 2. Invoke create_job for the repeated job, passing sample_image_url
    # We pass sample_image_url="/outputs/job_one_source/sample_image.jpg"
    print("Calling create_job with sample_image_url...")
    res = await create_job(
        background_tasks=bg_tasks,
        query="5 cats",
        count=5,
        label=True,
        label_type="detection",
        export_format="yolo",
        quality_threshold=0.6,
        target_labels="cat",
        folder_mode="manual",
        custom_folder_name=job2_id,
        allow_duplicates=False,
        sample_image=None,
        sample_image_url=f"/outputs/{job1_id}/sample_image.jpg"
    )

    print(f"API Response: {res}")
    assert res["status"] == "queued"
    assert res["job_id"] == job2_id

    # 3. Verify that the repeated job directory has the copied sample image
    copied_sample = os.path.join(job2_dir, "sample_image.jpg")
    assert os.path.exists(copied_sample), f"Expected copied sample image at {copied_sample}, but it doesn't exist"
    
    with open(copied_sample, "r") as f:
        content = f.read()
    assert content == "dummy-image-bytes", f"Content mismatch! Got: {content}"
    print("Sample image copied successfully and matches source content!")

    # 4. Verify job_store state has the correct sample image URL
    job_state = job_store.get_job(job2_id)
    print(f"Job state in store: {job_state}")
    assert job_state["has_sample_image"] is True
    assert job_state["sample_image_url"] == f"/outputs/{job2_id}/sample_image.jpg"

    # Clean up
    if os.path.exists(job1_dir):
        shutil.rmtree(job1_dir)
    if os.path.exists(job2_dir):
        shutil.rmtree(job2_dir)

    print("Repeat endpoint tests PASSED successfully!")

if __name__ == "__main__":
    asyncio.run(test_repeat_endpoint())
