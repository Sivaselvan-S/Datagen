import os
import json
import logging
from PIL import Image
from typing import List, Dict, Any, Tuple
import backend.job_store as job_store
from backend.models.clip_filter import clip_filter_instance
from backend.models.florence_label import florence_labeler_instance

logger = logging.getLogger("labeler")

def apply_nms(bboxes: List[Dict[str, Any]], iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """Applies Non-Maximum Suppression to remove redundant/overlapping bounding boxes."""
    if not bboxes:
        return []
    sorted_boxes = sorted(bboxes, key=lambda x: x.get("confidence", 0.0), reverse=True)
    keep = []
    while sorted_boxes:
        best_box = sorted_boxes.pop(0)
        keep.append(best_box)
        remaining = []
        for box in sorted_boxes:
            x1 = max(best_box["x_min"], box["x_min"])
            y1 = max(best_box["y_min"], box["y_min"])
            x2 = min(best_box["x_max"], box["x_max"])
            y2 = min(best_box["y_max"], box["y_max"])
            
            inter_w = max(0.0, x2 - x1)
            inter_h = max(0.0, y2 - y1)
            intersection = inter_w * inter_h
            
            area1 = (best_box["x_max"] - best_box["x_min"]) * (best_box["y_max"] - best_box["y_min"])
            area2 = (box["x_max"] - box["x_min"]) * (box["y_max"] - box["y_min"])
            union = area1 + area2 - intersection
            
            iou = intersection / union if union > 0 else 0.0
            
            # Same label NMS
            if best_box["label"] == box["label"] and iou > iou_threshold:
                continue
            # Cross-label NMS for extreme overlap
            if best_box["label"] != box["label"] and iou > 0.8:
                continue
            remaining.append(box)
        sorted_boxes = remaining
    return keep

def label_dataset(
    job_id: str,
    label_type: str,
    target_labels: List[str],
    output_dir: str = "./outputs"
) -> Tuple[int, int, str]:
    """
    Labels filtered images based on label_type:
    - classification: CLIP zero-shot matching.
    - detection: Florence-2 object detection.
    - segmentation: Florence-2 phrase segmentation.
    
    Returns (labeled_count, mock_label_count, raw_labels_file)
    """
    job_dir = os.path.join(output_dir, job_id)
    filtered_dir = os.path.join(job_dir, "filtered")
    labels_dir = os.path.join(job_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    
    if not os.path.exists(filtered_dir):
        logger.error(f"Filtered directory does not exist: {filtered_dir}")
        return 0, 0, ""
        
    image_files = [f for f in os.listdir(filtered_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    raw_labels = {}
    labeled_count = 0
    mock_label_count = 0
    
    logger.info(f"Labeling {len(image_files)} images with mode '{label_type}'...")
    
    for filename in image_files:
        # Check cancellation
        if job_store.is_job_cancelled(job_id):
            logger.info(f"Job {job_id} cancelled during labeling. Aborting.")
            raise ValueError("Job was cancelled by the user.")
            
        img_path = os.path.join(filtered_dir, filename)
        try:
            img = Image.open(img_path)
            
            if label_type == "classification":
                candidates = target_labels if target_labels else ["object", "other"]
                label, confidence = clip_filter_instance.classify_image(img, candidates)
                
                raw_labels[filename] = {
                    "label": label,
                    "confidence": confidence
                }
                labeled_count += 1
                
            elif label_type == "detection":
                # Runs Florence-2 OD
                if florence_labeler_instance.initialized:
                    bboxes = florence_labeler_instance.detect_objects(img, target_labels)
                    bboxes = apply_nms(bboxes, iou_threshold=0.5)
                else:
                    logger.warning("Florence-2 not initialized. Generating mock detection bounding box.")
                    w, h = img.size
                    bboxes = [{
                        "label": target_labels[0] if target_labels else "object",
                        "x_min": w * 0.15,
                        "y_min": h * 0.15,
                        "x_max": w * 0.85,
                        "y_max": h * 0.85,
                        "confidence": 0.5,
                        "mock": True
                    }]
                    mock_label_count += 1
                
                raw_labels[filename] = {
                    "bboxes": bboxes
                }
                if bboxes:
                    labeled_count += 1
                    
            elif label_type == "segmentation":
                # Run Florence-2 Referring Expression Segmentation
                polygons = []
                if florence_labeler_instance.initialized:
                    phrases = target_labels if target_labels else ["object"]
                    for phrase in phrases:
                        phrase_polys = florence_labeler_instance.segment_objects(img, phrase)
                        polygons.extend(phrase_polys)
                else:
                    logger.warning("Florence-2 not initialized. Generating mock segmentation polygon.")
                    w, h = img.size
                    x1, y1, x2, y2 = w * 0.2, h * 0.2, w * 0.8, h * 0.8
                    polygons = [{
                        "label": target_labels[0] if target_labels else "object",
                        "x_min": x1,
                        "y_min": y1,
                        "x_max": x2,
                        "y_max": y2,
                        "polygons": [[x1, y1, x2, y1, x2, y2, x1, y2]],
                        "confidence": 0.5,
                        "mock": True
                    }]
                    mock_label_count += 1
                
                raw_labels[filename] = {
                    "polygons": polygons
                }
                if polygons:
                    labeled_count += 1
                    
            else:
                logger.error(f"Unknown label type: {label_type}")
                raw_labels[filename] = {
                    "label": "unlabeled",
                    "confidence": 0.0,
                    "failed": True
                }
                
        except Exception as e:
            logger.error(f"Failed to label image {filename}: {e}", exc_info=True)
            # Write a failed tag
            if label_type == "classification":
                raw_labels[filename] = {"label": "unlabeled", "confidence": 0.0, "failed": True}
            elif label_type == "detection":
                raw_labels[filename] = {"bboxes": [], "failed": True}
            elif label_type == "segmentation":
                raw_labels[filename] = {"polygons": [], "failed": True}
                
    # Save labels_raw.json
    raw_labels_file = os.path.join(labels_dir, "labels_raw.json")
    temp_file = raw_labels_file + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(raw_labels, f, indent=2)
    os.replace(temp_file, raw_labels_file)
    
    logger.info(f"Labeling finished. Labeled {labeled_count} images (Mocks: {mock_label_count}).")
    return labeled_count, mock_label_count, raw_labels_file
