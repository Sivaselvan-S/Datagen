import os
import json
import shutil
import sys
import zipfile

# Add root folder to sys.path to resolve imports correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.exporter import multiply_job_dataset
import backend.job_store as job_store

def test_augmentation():
    job_id = "test_augment_job"
    output_dir = "./outputs"
    job_dir = os.path.join(output_dir, job_id)
    filtered_dir = os.path.join(job_dir, "filtered")
    labels_dir = os.path.join(job_dir, "labels")
    
    # Clean up previous runs
    if os.path.exists(job_dir):
        shutil.rmtree(job_dir)
        
    os.makedirs(filtered_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Create two dummy images with different sizes
    from PIL import Image
    img1_path = os.path.join(filtered_dir, "image1.jpg") # 100x100
    img2_path = os.path.join(filtered_dir, "image2.png") # 200x150
    
    img1 = Image.new("RGB", (100, 100), color="blue")
    img1.save(img1_path, "JPEG")
    
    img2 = Image.new("RGB", (200, 150), color="green")
    img2.save(img2_path, "PNG")

    # Create labels_raw.json
    labels_raw_data = {
        "image1.jpg": {
            "bboxes": [
                {
                    "label": "cup",
                    "x_min": 10.0,
                    "y_min": 20.0,
                    "x_max": 80.0,
                    "y_max": 90.0,
                    "confidence": 0.95
                }
            ]
        },
        "image2.png": {
            "polygons": [
                {
                    "label": "mug",
                    "x_min": 20.0,
                    "y_min": 30.0,
                    "x_max": 180.0,
                    "y_max": 120.0,
                    "confidence": 0.88,
                    "polygons": [
                        [20.0, 30.0, 180.0, 30.0, 180.0, 120.0, 20.0, 120.0]
                    ]
                }
            ]
        }
    }
    
    with open(os.path.join(labels_dir, "labels_raw.json"), "w", encoding="utf-8") as f:
        json.dump(labels_raw_data, f, indent=2)
        
    # Create job in job_store to mock active status
    job_store.create_job(job_id, {"export_format": "yolo", "label_type": "detection"}, "test query")
    job_store.update_job(job_id, {"images_passed": 2, "labeled_count": 2})
    
    print("Testing 3x Multiplier in 'augment' mode...")
    total_images, zip_path = multiply_job_dataset(job_id, 3, "augment", output_dir)
    print(f"Result: total_images={total_images}, zip_path={zip_path}")
    
    assert total_images == 6, f"Expected 6 images, got {total_images}"
    assert os.path.exists(zip_path), "ZIP file was not created"
    
    # Check folder files
    files = sorted(os.listdir(filtered_dir))
    expected_files = [
        "image1.jpg", "image1_copy1.jpg", "image1_copy2.jpg",
        "image2.png", "image2_copy1.png", "image2_copy2.png"
    ]
    assert files == expected_files, f"Expected files {expected_files}, got {files}"
    
    # Read updated labels_raw.json
    with open(os.path.join(labels_dir, "labels_raw.json"), "r", encoding="utf-8") as f:
        updated_labels = json.load(f)
        
    # image1_copy1 should be "flip" (op index 0)
    # W=100. Original bbox: x_min=10, x_max=80
    # Flipped bbox: x_min = 100 - 80 = 20, x_max = 100 - 10 = 90
    assert "image1_copy1.jpg" in updated_labels
    box_copy1 = updated_labels["image1_copy1.jpg"]["bboxes"][0]
    assert box_copy1["x_min"] == 20.0, f"Expected 20.0, got {box_copy1['x_min']}"
    assert box_copy1["x_max"] == 90.0, f"Expected 90.0, got {box_copy1['x_max']}"
    assert box_copy1["y_min"] == 20.0, "y_min should not change"
    assert box_copy1["y_max"] == 90.0, "y_max should not change"
    
    # image1_copy2 should be "brightness_up" (op index 1)
    # Coordinates should remain unmodified (same as original)
    assert "image1_copy2.jpg" in updated_labels
    box_copy2 = updated_labels["image1_copy2.jpg"]["bboxes"][0]
    assert box_copy2["x_min"] == 10.0, f"Expected 10.0, got {box_copy2['x_min']}"
    assert box_copy2["x_max"] == 80.0, f"Expected 80.0, got {box_copy2['x_max']}"
    
    # image2_copy1 should be "flip" (op index 0)
    # W=200. Original polygon box: x_min=20, x_max=180
    # Flipped polygon box: x_min = 200 - 180 = 20, x_max = 200 - 20 = 180
    # Original polygon points: [20.0, 30.0, 180.0, 30.0, 180.0, 120.0, 20.0, 120.0]
    # Flipped polygon points: [180.0, 30.0, 20.0, 30.0, 20.0, 120.0, 180.0, 120.0]
    assert "image2_copy1.png" in updated_labels
    poly_copy1 = updated_labels["image2_copy1.png"]["polygons"][0]
    assert poly_copy1["x_min"] == 20.0, f"Expected 20.0, got {poly_copy1['x_min']}"
    assert poly_copy1["x_max"] == 180.0, f"Expected 180.0, got {poly_copy1['x_max']}"
    
    expected_pts = [180.0, 30.0, 20.0, 30.0, 20.0, 120.0, 180.0, 120.0]
    actual_pts = poly_copy1["polygons"][0]
    assert len(expected_pts) == len(actual_pts)
    for expected, actual in zip(expected_pts, actual_pts):
        assert abs(expected - actual) < 1e-5, f"Expected pt {expected}, got {actual}"
        
    print("Augmented coordinates verified successfully!")
    
    # Let's inspect the zip file to make sure it includes the copied annotations
    with zipfile.ZipFile(zip_path, "r") as zf:
        zip_namelist = zf.namelist()
        print("Zip files list:", zip_namelist)
        
        # Check YOLO label text files exist
        txt_files = [f for f in zip_namelist if f.endswith(".txt") and "labels/" in f]
        assert len(txt_files) == 6, f"Expected 6 label text files inside zip, found {len(txt_files)}"
        
        # Find the text file for image1_copy1
        # The path will be either labels/train/image1_copy1.txt or labels/val/image1_copy1.txt
        img1_copy1_txt_path = [p for p in txt_files if "image1_copy1" in p][0]
        content = zf.read(img1_copy1_txt_path).decode("utf-8").strip()
        print(f"Content of {img1_copy1_txt_path}:", content)
        
        # YOLO format for bboxes: class_id x_center y_center width height (all normalized)
        # Original: W=100, H=100, x_min=10, x_max=80, y_min=20, y_max=90
        # Flipped: x_min=20, x_max=90, y_min=20, y_max=90
        # Flipped center x = 55. Width = 70. Normalized x_center = 0.55, width = 0.70.
        # Center y = 55. Height = 70. Normalized y_center = 0.55, height = 0.70.
        # So we expect something like "0 0.550000 0.550000 0.700000 0.700000"
        parts = content.split()
        assert parts[0] == "0", "class_id should be 0"
        assert abs(float(parts[1]) - 0.55) < 1e-4, f"Expected center x ~0.55, got {parts[1]}"
        assert abs(float(parts[3]) - 0.70) < 1e-4, f"Expected width ~0.70, got {parts[3]}"
        
    print("ZIP file structure and YOLO formatting checks PASSED!")
    
    # Test Idempotency (reducing multiplier to 2x)
    print("Testing idempotency (reducing multiplier to 2x)...")
    total_images2, zip_path2 = multiply_job_dataset(job_id, 2, "augment", output_dir)
    assert total_images2 == 4, f"Expected 4 images, got {total_images2}"
    
    files2 = sorted(os.listdir(filtered_dir))
    expected_files2 = [
        "image1.jpg", "image1_copy1.jpg",
        "image2.png", "image2_copy1.png"
    ]
    assert files2 == expected_files2, f"Expected files {expected_files2}, got {files2}"
    
    with open(os.path.join(labels_dir, "labels_raw.json"), "r", encoding="utf-8") as f:
        updated_labels2 = json.load(f)
        
    assert "image1_copy2.jpg" not in updated_labels2
    assert "image1_copy1.jpg" in updated_labels2
    
    print("Idempotency checks PASSED!")
    
    # Clean up
    shutil.rmtree(job_dir)
    print("All augmentation test assertions PASSED!")

if __name__ == "__main__":
    test_augmentation()
