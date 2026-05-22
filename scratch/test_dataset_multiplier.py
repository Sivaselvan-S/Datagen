import os
import json
import shutil
import sys

# Add root folder to sys.path to resolve imports correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.exporter import multiply_job_dataset
import backend.job_store as job_store

def test_multiplier():
    job_id = "test_multiplier_job"
    output_dir = "./outputs"
    job_dir = os.path.join(output_dir, job_id)
    filtered_dir = os.path.join(job_dir, "filtered")
    labels_dir = os.path.join(job_dir, "labels")
    
    # Clean up previous runs
    if os.path.exists(job_dir):
        shutil.rmtree(job_dir)
        
    os.makedirs(filtered_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Create two dummy images
    from PIL import Image
    img1_path = os.path.join(filtered_dir, "image1.jpg")
    img2_path = os.path.join(filtered_dir, "image2.png")
    
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img1_path, "JPEG")
    img.save(img2_path, "PNG")

        
    # Create labels_raw.json
    labels_raw_data = {
        "image1.jpg": {
            "label": "cup",
            "confidence": 0.95
        },
        "image2.png": {
            "bboxes": [
                {
                    "label": "mug",
                    "x_min": 10,
                    "y_min": 10,
                    "x_max": 90,
                    "y_max": 90,
                    "confidence": 0.88
                }
            ]
        }
    }
    
    with open(os.path.join(labels_dir, "labels_raw.json"), "w", encoding="utf-8") as f:
        json.dump(labels_raw_data, f, indent=2)
        
    # Create a job in job_store to mock active status
    job_store.create_job(job_id, {"export_format": "yolo", "label_type": "detection"}, "test query")
    job_store.update_job(job_id, {"images_passed": 2, "labeled_count": 2})
    
    print("Testing 3x multiplier...")
    total_images, zip_path = multiply_job_dataset(job_id, 3, output_dir)
    print(f"Result: total_images={total_images}, zip_path={zip_path}")
    
    assert total_images == 6, f"Expected 6 images, got {total_images}"
    assert os.path.exists(zip_path), "ZIP file was not created"
    
    # Check folder contents
    files = sorted(os.listdir(filtered_dir))
    expected_files = [
        "image1.jpg", "image1_copy1.jpg", "image1_copy2.jpg",
        "image2.png", "image2_copy1.png", "image2_copy2.png"
    ]
    assert files == expected_files, f"Expected files {expected_files}, got {files}"
    
    # Check updated labels_raw.json
    with open(os.path.join(labels_dir, "labels_raw.json"), "r", encoding="utf-8") as f:
        updated_labels = json.load(f)
        
    assert "image1_copy1.jpg" in updated_labels
    assert "image1_copy2.jpg" in updated_labels
    assert "image2_copy1.png" in updated_labels
    assert "image2_copy2.png" in updated_labels
    assert updated_labels["image1_copy1.jpg"]["label"] == "cup"
    assert updated_labels["image2_copy2.png"]["bboxes"][0]["label"] == "mug"
    
    # Check job_store stats
    job = job_store.get_job(job_id)
    assert job["images_passed"] == 6
    assert job["labeled_count"] == 6
    print("3x multiplication test PASSED!")
    
    # Test idempotency (multiplying to 2x now)
    print("Testing 2x multiplier (idempotency check)...")
    total_images, zip_path = multiply_job_dataset(job_id, 2, output_dir)
    print(f"Result: total_images={total_images}, zip_path={zip_path}")
    
    assert total_images == 4, f"Expected 4 images, got {total_images}"
    
    files = sorted(os.listdir(filtered_dir))
    expected_files = [
        "image1.jpg", "image1_copy1.jpg",
        "image2.png", "image2_copy1.png"
    ]
    assert files == expected_files, f"Expected files {expected_files}, got {files}"
    
    with open(os.path.join(labels_dir, "labels_raw.json"), "r", encoding="utf-8") as f:
        updated_labels2 = json.load(f)
        
    assert "image1_copy2.jpg" not in updated_labels2
    assert "image2_copy2.png" not in updated_labels2
    assert "image1_copy1.jpg" in updated_labels2
    assert "image2_copy1.png" in updated_labels2
    
    job = job_store.get_job(job_id)
    assert job["images_passed"] == 4
    assert job["labeled_count"] == 4
    print("2x multiplication (idempotency) test PASSED!")
    
    # Clean up
    shutil.rmtree(job_dir)
    print("All tests passed and clean up completed successfully!")

if __name__ == "__main__":
    test_multiplier()
