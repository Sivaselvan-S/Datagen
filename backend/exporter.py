import os
import json
import shutil
import csv
import logging
from PIL import Image
from typing import Dict, Any, List, Tuple

logger = logging.getLogger("exporter")

def export_dataset(
    job_id: str,
    export_format: str,
    label_type: str,
    output_dir: str = "./outputs"
) -> str:
    """
    Exports filtered images and labels into YOLO, COCO, or CSV format and packages them as a ZIP file.
    Returns the absolute path to the generated ZIP file.
    """
    import random
    
    job_dir = os.path.join(output_dir, job_id)
    filtered_dir = os.path.join(job_dir, "filtered")
    labels_dir = os.path.join(job_dir, "labels")
    
    # 1. Validation: check that at least 1 image exists in the filtered folder
    if not os.path.exists(filtered_dir):
        raise ValueError("Filtered image folder does not exist. Cannot export.")
        
    image_files = [f for f in os.listdir(filtered_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        raise ValueError("No images passed the quality filter. Cannot export an empty dataset.")
        
    # Read raw labels if they exist
    raw_labels = {}
    labels_file = os.path.join(labels_dir, "labels_raw.json")
    if os.path.exists(labels_file):
        with open(labels_file, "r", encoding="utf-8") as f:
            raw_labels = json.load(f)
            
    # Filter out failed or corrupt images upfront
    valid_image_files = []
    for filename in image_files:
        img_labels = raw_labels.get(filename, {})
        if img_labels.get("failed") is True:
            logger.info(f"Skipping image {filename} in export because labeling failed.")
            continue
        src_img = os.path.join(filtered_dir, filename)
        try:
            with Image.open(src_img) as im:
                im.verify()
            valid_image_files.append(filename)
        except Exception:
            logger.warning(f"Image {filename} failed verification and will be skipped in export.")
            continue
            
    if not valid_image_files:
        raise ValueError("No valid/labeled images to export. All images failed labeling or are corrupt.")
        
    # Implement 80/20 train/validation split
    num_files = len(valid_image_files)
    if num_files >= 5:
        split_idx = int(num_files * 0.8)
    else:
        split_idx = num_files - 1 if num_files >= 2 else 1
        
    shuffled_files = list(valid_image_files)
    random.Random(42).shuffle(shuffled_files)
    train_files = set(shuffled_files[:split_idx])
    val_files = set(shuffled_files[split_idx:])
    
    # Setup temp export folder
    export_temp_dir = os.path.join(job_dir, "export_temp")
    if os.path.exists(export_temp_dir):
        shutil.rmtree(export_temp_dir)
    os.makedirs(export_temp_dir, exist_ok=True)
    
    # Build unique categories/classes list (excluding "unlabeled")
    classes_set = set()
    for img_name in valid_image_files:
        label_data = raw_labels.get(img_name, {})
        if "label" in label_data:
            classes_set.add(label_data["label"])
        elif "bboxes" in label_data:
            for box in label_data["bboxes"]:
                classes_set.add(box["label"])
        elif "polygons" in label_data:
            for poly in label_data["polygons"]:
                classes_set.add(poly["label"])
                
    classes = sorted(list(classes_set - {"unlabeled"}))
    if not classes:
        classes = ["object"]
    class_to_id = {cls: idx for idx, cls in enumerate(classes)}
    
    # Run format-specific formatting
    if export_format == "yolo":
        # Create YOLO structure:
        # export_temp/images/train/*.jpg, export_temp/images/val/*.jpg
        # export_temp/labels/train/*.txt, export_temp/labels/val/*.txt
        # export_temp/classes.txt
        # export_temp/data.yaml
        img_train_dir = os.path.join(export_temp_dir, "images", "train")
        img_val_dir = os.path.join(export_temp_dir, "images", "val")
        lbl_train_dir = os.path.join(export_temp_dir, "labels", "train")
        lbl_val_dir = os.path.join(export_temp_dir, "labels", "val")
        os.makedirs(img_train_dir, exist_ok=True)
        os.makedirs(img_val_dir, exist_ok=True)
        os.makedirs(lbl_train_dir, exist_ok=True)
        os.makedirs(lbl_val_dir, exist_ok=True)
        
        # Write classes.txt
        with open(os.path.join(export_temp_dir, "classes.txt"), "w", encoding="utf-8") as f:
            for cls in classes:
                f.write(f"{cls}\n")
                
        # Write data.yaml
        with open(os.path.join(export_temp_dir, "data.yaml"), "w", encoding="utf-8") as f:
            f.write(f"train: images/train\n")
            f.write(f"val: images/val\n")
            f.write(f"nc: {len(classes)}\n")
            f.write(f"names: {json.dumps(classes)}\n")
            
        for filename in valid_image_files:
            is_train = filename in train_files
            dest_img_dir = img_train_dir if is_train else img_val_dir
            dest_lbl_dir = lbl_train_dir if is_train else lbl_val_dir
            
            src_img = os.path.join(filtered_dir, filename)
            shutil.copy2(src_img, os.path.join(dest_img_dir, filename))
            
            # Get dimensions
            try:
                with Image.open(src_img) as im:
                    img_w, img_h = im.size
            except Exception:
                img_w, img_h = 640, 480
                
            # Write txt label
            label_filename = os.path.splitext(filename)[0] + ".txt"
            label_filepath = os.path.join(dest_lbl_dir, label_filename)
            
            yolo_lines = []
            img_labels = raw_labels.get(filename, {})
            
            if "label" in img_labels:
                cls_name = img_labels["label"]
                if cls_name != "unlabeled" and cls_name in class_to_id:
                    cls_id = class_to_id[cls_name]
                    yolo_lines.append(f"{cls_id} 0.5 0.5 1.0 1.0")
            elif "bboxes" in img_labels:
                for box in img_labels["bboxes"]:
                    cls_name = box["label"]
                    if cls_name in class_to_id:
                        cls_id = class_to_id[cls_name]
                        x1, y1, x2, y2 = box["x_min"], box["y_min"], box["x_max"], box["y_max"]
                        w = x2 - x1
                        h = y2 - y1
                        x_center = x1 + (w / 2)
                        y_center = y1 + (h / 2)
                        x_center_norm = max(0.0, min(1.0, x_center / img_w))
                        y_center_norm = max(0.0, min(1.0, y_center / img_h))
                        w_norm = max(0.0, min(1.0, w / img_w))
                        h_norm = max(0.0, min(1.0, h / img_h))
                        yolo_lines.append(f"{cls_id} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
            elif "polygons" in img_labels:
                for poly_data in img_labels["polygons"]:
                    cls_name = poly_data["label"]
                    if cls_name in class_to_id:
                        cls_id = class_to_id[cls_name]
                        for poly in poly_data.get("polygons", []):
                            norm_poly = []
                            for idx, pt in enumerate(poly):
                                if idx % 2 == 0:
                                    norm_poly.append(f"{pt / img_w:.6f}")
                                else:
                                    norm_poly.append(f"{pt / img_h:.6f}")
                            yolo_lines.append(f"{cls_id} " + " ".join(norm_poly))
                            
            with open(label_filepath, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines) + "\n")
                
    elif export_format == "coco":
        # Create COCO structure:
        # export_temp/images/train/*.jpg, export_temp/images/val/*.jpg
        # export_temp/annotations_train.json, export_temp/annotations_val.json
        img_train_dir = os.path.join(export_temp_dir, "images", "train")
        img_val_dir = os.path.join(export_temp_dir, "images", "val")
        os.makedirs(img_train_dir, exist_ok=True)
        os.makedirs(img_val_dir, exist_ok=True)
        
        coco_train_images = []
        coco_val_images = []
        coco_train_annotations = []
        coco_val_annotations = []
        
        # Category IDs must be 1-indexed for COCO
        class_to_coco_id = {cls: idx + 1 for idx, cls in enumerate(classes)}
        coco_categories = [{"id": id_val, "name": name, "supercategory": "object"} for name, id_val in class_to_coco_id.items()]
        
        anno_id_counter = 1
        for img_idx, filename in enumerate(valid_image_files):
            is_train = filename in train_files
            dest_img_dir = img_train_dir if is_train else img_val_dir
            
            src_img = os.path.join(filtered_dir, filename)
            shutil.copy2(src_img, os.path.join(dest_img_dir, filename))
            
            # Dimensions
            try:
                with Image.open(src_img) as im:
                    img_w, img_h = im.size
            except Exception:
                img_w, img_h = 640, 480
                
            image_id = img_idx + 1
            image_info = {
                "id": image_id,
                "file_name": f"images/train/{filename}" if is_train else f"images/val/{filename}",
                "width": img_w,
                "height": img_h
            }
            
            if is_train:
                coco_train_images.append(image_info)
            else:
                coco_val_images.append(image_info)
                
            img_labels = raw_labels.get(filename, {})
            
            if "label" in img_labels:
                cls_name = img_labels["label"]
                if cls_name != "unlabeled" and cls_name in class_to_coco_id:
                    anno = {
                        "id": anno_id_counter,
                        "image_id": image_id,
                        "category_id": class_to_coco_id[cls_name],
                        "bbox": [0.0, 0.0, float(img_w), float(img_h)],
                        "area": float(img_w * img_h),
                        "segmentation": [],
                        "iscrowd": 0
                    }
                    if is_train:
                        coco_train_annotations.append(anno)
                    else:
                        coco_val_annotations.append(anno)
                    anno_id_counter += 1
            elif "bboxes" in img_labels:
                for box in img_labels["bboxes"]:
                    cls_name = box["label"]
                    if cls_name in class_to_coco_id:
                        x1, y1, x2, y2 = box["x_min"], box["y_min"], box["x_max"], box["y_max"]
                        w = x2 - x1
                        h = y2 - y1
                        anno = {
                            "id": anno_id_counter,
                            "image_id": image_id,
                            "category_id": class_to_coco_id[cls_name],
                            "bbox": [float(x1), float(y1), float(w), float(h)],
                            "area": float(w * h),
                            "segmentation": [],
                            "iscrowd": 0
                        }
                        if is_train:
                            coco_train_annotations.append(anno)
                        else:
                            coco_val_annotations.append(anno)
                        anno_id_counter += 1
            elif "polygons" in img_labels:
                for poly_data in img_labels["polygons"]:
                    cls_name = poly_data["label"]
                    if cls_name in class_to_coco_id:
                        x1, y1, x2, y2 = poly_data["x_min"], poly_data["y_min"], poly_data["x_max"], poly_data["y_max"]
                        w = x2 - x1
                        h = y2 - y1
                        polys = poly_data.get("polygons", [])
                        anno = {
                            "id": anno_id_counter,
                            "image_id": image_id,
                            "category_id": class_to_coco_id[cls_name],
                            "bbox": [float(x1), float(y1), float(w), float(h)],
                            "area": float(w * h),
                            "segmentation": polys,
                            "iscrowd": 0
                        }
                        if is_train:
                            coco_train_annotations.append(anno)
                        else:
                            coco_val_annotations.append(anno)
                        anno_id_counter += 1
                        
        coco_train_data = {
            "images": coco_train_images,
            "annotations": coco_train_annotations,
            "categories": coco_categories
        }
        coco_val_data = {
            "images": coco_val_images,
            "annotations": coco_val_annotations,
            "categories": coco_categories
        }
        
        with open(os.path.join(export_temp_dir, "annotations_train.json"), "w", encoding="utf-8") as f:
            json.dump(coco_train_data, f, indent=2)
        with open(os.path.join(export_temp_dir, "annotations_val.json"), "w", encoding="utf-8") as f:
            json.dump(coco_val_data, f, indent=2)
            
    elif export_format == "csv":
        # Create CSV structure:
        # export_temp/images/train/*.jpg, export_temp/images/val/*.jpg
        # export_temp/annotations_train.csv, export_temp/annotations_val.csv
        img_train_dir = os.path.join(export_temp_dir, "images", "train")
        img_val_dir = os.path.join(export_temp_dir, "images", "val")
        os.makedirs(img_train_dir, exist_ok=True)
        os.makedirs(img_val_dir, exist_ok=True)
        
        csv_train_filepath = os.path.join(export_temp_dir, "annotations_train.csv")
        csv_val_filepath = os.path.join(export_temp_dir, "annotations_val.csv")
        
        with open(csv_train_filepath, "w", newline="", encoding="utf-8") as f_train, \
             open(csv_val_filepath, "w", newline="", encoding="utf-8") as f_val:
            
            writer_train = csv.writer(f_train)
            writer_val = csv.writer(f_val)
            
            header = ["filename", "label", "x_min", "y_min", "x_max", "y_max", "confidence", "polygon"]
            writer_train.writerow(header)
            writer_val.writerow(header)
            
            for filename in valid_image_files:
                is_train = filename in train_files
                dest_img_dir = img_train_dir if is_train else img_val_dir
                writer = writer_train if is_train else writer_val
                
                shutil.copy2(os.path.join(filtered_dir, filename), os.path.join(dest_img_dir, filename))
                
                csv_filename = f"images/train/{filename}" if is_train else f"images/val/{filename}"
                img_labels = raw_labels.get(filename, {})
                
                if "label" in img_labels:
                    writer.writerow([csv_filename, img_labels["label"], "", "", "", "", img_labels["confidence"], ""])
                elif "bboxes" in img_labels:
                    if not img_labels["bboxes"]:
                        writer.writerow([csv_filename, "unlabeled", "", "", "", "", 0.0, ""])
                    else:
                        for box in img_labels["bboxes"]:
                            writer.writerow([
                                csv_filename,
                                box["label"],
                                box["x_min"],
                                box["y_min"],
                                box["x_max"],
                                box["y_max"],
                                box["confidence"],
                                ""
                            ])
                elif "polygons" in img_labels:
                    if not img_labels["polygons"]:
                        writer.writerow([csv_filename, "unlabeled", "", "", "", "", 0.0, ""])
                    else:
                        for poly in img_labels["polygons"]:
                            writer.writerow([
                                csv_filename,
                                poly["label"],
                                poly["x_min"],
                                poly["y_min"],
                                poly["x_max"],
                                poly["y_max"],
                                poly["confidence"],
                                json.dumps(poly.get("polygons", []))
                            ])
                            
    else:
        raise ValueError(f"Unsupported export format: {export_format}")
        
    # 2. Package into a ZIP file
    zip_base_name = os.path.join(job_dir, f"dataset_{job_id}")
    zip_path_raw = shutil.make_archive(zip_base_name, "zip", export_temp_dir)
    
    # Cleanup temp directory
    shutil.rmtree(export_temp_dir)
    
    logger.info(f"Dataset successfully exported and zipped to: {zip_path_raw}")
    return zip_path_raw

def apply_augmentation(img_path: str, dest_path: str, op_name: str, label_data: dict) -> dict:
    from PIL import Image, ImageEnhance, ImageFilter
    import copy
    
    # Deep copy the label data to avoid mutating original
    new_label_data = copy.deepcopy(label_data)
    
    with Image.open(img_path) as img:
        W, H = img.size
        
        # 1. Flip
        if "flip" in op_name:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Update coordinate references
            if "bboxes" in new_label_data:
                for box in new_label_data["bboxes"]:
                    x_min = box["x_min"]
                    x_max = box["x_max"]
                    box["x_min"] = max(0.0, min(float(W), float(W - x_max)))
                    box["x_max"] = max(0.0, min(float(W), float(W - x_min)))
            elif "polygons" in new_label_data:
                for poly_data in new_label_data["polygons"]:
                    # Flip enclosing bbox
                    x_min = poly_data["x_min"]
                    x_max = poly_data["x_max"]
                    poly_data["x_min"] = max(0.0, min(float(W), float(W - x_max)))
                    poly_data["x_max"] = max(0.0, min(float(W), float(W - x_min)))
                    
                    # Flip points
                    new_polys = []
                    for poly in poly_data.get("polygons", []):
                        new_poly = []
                        for idx, pt in enumerate(poly):
                            if idx % 2 == 0:
                                new_poly.append(max(0.0, min(float(W), float(W - pt))))
                            else:
                                new_poly.append(pt)
                        new_polys.append(new_poly)
                    poly_data["polygons"] = new_polys
                    
        # 2. Brightness
        if "brightness_up" in op_name:
            img = ImageEnhance.Brightness(img).enhance(1.3)
        elif "brightness_down" in op_name:
            img = ImageEnhance.Brightness(img).enhance(0.7)
            
        # 3. Contrast
        if "contrast_up" in op_name:
            img = ImageEnhance.Contrast(img).enhance(1.3)
        elif "contrast_down" in op_name:
            img = ImageEnhance.Contrast(img).enhance(0.7)
            
        # 4. Blur
        if "blur" in op_name:
            img = img.filter(ImageFilter.GaussianBlur(1.5))
            
        # Convert to RGB if saving as JPEG and image has transparency
        if dest_path.lower().endswith(('.jpg', '.jpeg')) and img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
            
        # Save image
        img.save(dest_path)
        
    return new_label_data

def multiply_job_dataset(
    job_id: str,
    multiplier: int,
    mode: str = "copy",
    output_dir: str = "./outputs"
) -> Tuple[int, str]:
    """
    Multiplies the dataset for a given job ID.
    If mode is 'augment', applies smart dataset augmentations (flips, brightness, contrast, blur).
    If mode is 'copy', applies simple image/label copies.
    """
    import re
    import copy
    import backend.job_store as job_store
    
    # Defensive handling of legacy positional arguments
    if mode not in ("copy", "augment"):
        output_dir = mode
        mode = "copy"
        
    job_dir = os.path.join(output_dir, job_id)
    filtered_dir = os.path.join(job_dir, "filtered")
    labels_dir = os.path.join(job_dir, "labels")
    
    if not os.path.exists(filtered_dir):
        raise ValueError("Filtered image folder does not exist. Cannot expand dataset.")
        
    all_files = os.listdir(filtered_dir)
    image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Identify original images vs copies
    copy_pattern = re.compile(r"_copy\d+$")
    original_images = []
    for f in image_files:
        name_without_ext, _ = os.path.splitext(f)
        if not copy_pattern.search(name_without_ext):
            original_images.append(f)
            
    if not original_images:
        raise ValueError("No original images found to multiply.")
        
    # Load labels
    labels_path = os.path.join(labels_dir, "labels_raw.json")
    labels_data = {}
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            try:
                labels_data = json.load(f)
            except Exception as e:
                logger.error(f"Error loading labels file: {e}")
                
    # Clean up previous copy files and label entries for idempotency
    for f in image_files:
        name_without_ext, _ = os.path.splitext(f)
        if copy_pattern.search(name_without_ext):
            try:
                os.remove(os.path.join(filtered_dir, f))
            except Exception as e:
                logger.warning(f"Failed to remove old copy file {f}: {e}")
                
    keys_to_delete = [k for k in labels_data.keys() if copy_pattern.search(os.path.splitext(k)[0])]
    for k in keys_to_delete:
        labels_data.pop(k, None)
        
    # If multiplier is 1 or less, we just save and re-export the original dataset
    if multiplier <= 1:
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(labels_data, f, indent=2)
        total_images = len(original_images)
    else:
        # Perform replication
        AUGMENTATION_OPS = [
            "flip",
            "brightness_up",
            "brightness_down",
            "contrast_up",
            "contrast_down",
            "blur",
            "flip_brightness_up",
            "flip_brightness_down",
            "flip_contrast_up",
            "flip_contrast_down",
            "flip_blur"
        ]
        
        for orig in original_images:
            name_without_ext, ext = os.path.splitext(orig)
            src_path = os.path.join(filtered_dir, orig)
            orig_labels = labels_data.get(orig, {})
            
            for i in range(1, multiplier):
                copy_filename = f"{name_without_ext}_copy{i}{ext}"
                dest_path = os.path.join(filtered_dir, copy_filename)
                
                if mode == "augment":
                    # Choose operation sequentially
                    op = AUGMENTATION_OPS[(i - 1) % len(AUGMENTATION_OPS)]
                    try:
                        new_labels = apply_augmentation(src_path, dest_path, op, orig_labels)
                        labels_data[copy_filename] = new_labels
                    except Exception as e:
                        logger.error(f"Augmentation failed for {orig} with op {op}: {e}")
                        # Fallback to simple copy
                        shutil.copy2(src_path, dest_path)
                        labels_data[copy_filename] = copy.deepcopy(orig_labels)
                else:
                    # Simple copy mode
                    shutil.copy2(src_path, dest_path)
                    labels_data[copy_filename] = copy.deepcopy(orig_labels)
                    
        # Write updated labels_raw.json
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(labels_data, f, indent=2)
            
        total_images = len(original_images) * multiplier
        
    # Update job store metrics if the job is tracked
    job = job_store.get_job(job_id)
    if job:
        config = job.get("config", {})
        export_format = config.get("export_format", "yolo")
        label_type = config.get("label_type", "detection")

        # Rebuild results[] so the Results page shows ALL images (originals + augmented copies)
        # with their correct labels. Previously only the original pipeline images were in results[].
        all_current_images = sorted([
            f for f in os.listdir(filtered_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        updated_results = []
        for fname in all_current_images:
            updated_results.append({
                "filename": fname,
                "url": f"/outputs/{job_id}/filtered/{fname}",
                "labels_data": labels_data.get(fname, {})
            })

        job_store.update_job(job_id, {
            "images_passed": total_images,
            "labeled_count": total_images,
            "results": updated_results,
        })
    else:
        export_format = "yolo"
        label_type = "detection"
        
    # Re-trigger dataset export
    zip_path = export_dataset(job_id, export_format, label_type, output_dir)
    return total_images, zip_path





