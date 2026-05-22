import os
import sys
import json
import shutil
import asyncio
from PIL import Image, ImageDraw

# Add root folder to sys.path to resolve backend imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.models.clip_filter import clip_filter_instance
from backend.filter import filter_dataset

def create_dummy_image(path, color):
    img = Image.new("RGB", (200, 200), color=color)
    draw = ImageDraw.Draw(img)
    # Draw sharp high-contrast grid lines to ensure it passes the blur check
    grid_color = "white" if color != "white" else "black"
    for i in range(10, 200, 20):
        draw.line([(i, 0), (i, 200)], fill=grid_color, width=2)
        draw.line([(0, i), (200, i)], fill=grid_color, width=2)
    img.save(path)

async def test_target_count():
    print("Initializing CLIP filter model for similarity scoring...")
    clip_filter_instance.initialize()

    output_dir = "./outputs_test_target_count"
    job_id = "test_truncation_job"
    job_dir = os.path.join(output_dir, job_id)
    raw_dir = os.path.join(job_dir, "raw")
    filtered_dir = os.path.join(job_dir, "filtered")

    # Clean up test output dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(raw_dir, exist_ok=True)

    # Create 4 dummy images of different colors
    # We will search for "red square", so red image should have highest score, then orange, then green, blue.
    colors = ["red", "orange", "green", "blue"]
    metadata = {}
    for idx, color in enumerate(colors):
        filename = f"{idx:04d}.jpg"
        file_path = os.path.join(raw_dir, filename)
        create_dummy_image(file_path, color)
        
        metadata[filename] = {
            "image_url": f"http://dummy.com/{color}.jpg",
            "source_page": "http://dummy.com",
            "download_timestamp": "2026-05-21T00:00:00Z",
            "file_path": filename
        }

    # Write raw metadata.json
    with open(os.path.join(raw_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Let's run filter_dataset with target_count = 2 and query "red square"
    # It should sort by relevance and keep the top 2.
    print("Running filter_dataset with target_count=2...")
    passed_count, report_file = filter_dataset(
        job_id=job_id,
        query="red square",
        quality_threshold=0.01,  # low threshold to make sure they pass the cutoff check
        allow_duplicates=False,
        output_dir=output_dir,
        target_count=2
    )

    print(f"Passed count returned: {passed_count}")
    assert passed_count == 2, f"Expected 2 passed images, got {passed_count}"

    # Verify filtered files
    filtered_files = os.listdir(filtered_dir)
    print(f"Images in filtered folder: {filtered_files}")
    assert len(filtered_files) == 2, f"Expected 2 files in filtered directory, got {len(filtered_files)}"

    # Load report file
    with open(report_file, "r") as f:
        report = json.load(f)

    print("Filter report statistics:")
    print(json.dumps(report, indent=2))

    # Verify that exactly 2 images have "passed" status and 2 have "excess_discarded" status
    passed_in_report = [item for item in report["per_image"] if item["status"] == "passed"]
    discarded_in_report = [item for item in report["per_image"] if item["status"] == "excess_discarded"]

    assert len(passed_in_report) == 2, f"Expected 2 passed in report, got {len(passed_in_report)}"
    assert len(discarded_in_report) == 2, f"Expected 2 excess_discarded in report, got {len(discarded_in_report)}"

    # Verify that the passed images indeed have higher CLIP scores than the discarded ones
    min_passed_score = min(item["clip_score"] for item in passed_in_report)
    max_discarded_score = max(item["clip_score"] for item in discarded_in_report)
    
    print(f"Min passed score: {min_passed_score}")
    print(f"Max discarded score: {max_discarded_score}")
    assert min_passed_score >= max_discarded_score, "Passed image score is lower than discarded image score!"

    # Clean up test output dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    print("Target count truncation tests PASSED successfully!")

if __name__ == "__main__":
    asyncio.run(test_target_count())
