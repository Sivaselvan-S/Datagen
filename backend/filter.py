import os
import json
import shutil
import logging
import cv2
import imagehash
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, List
from backend.models.clip_filter import clip_filter_instance

logger = logging.getLogger("filter")

def filter_dataset(
    job_id: str,
    query: str,
    quality_threshold: float,
    allow_duplicates: bool = False,
    output_dir: str = "./outputs",
    target_count: int = None
) -> Tuple[int, str]:
    """
    Filters raw images using:
    - Size check (>= 100x100)
    - Blur check (Laplacian variance >= 10.0)
    - Perceptual hash deduplication (Hamming distance <= 8) (skipped if allow_duplicates is True)
    - CLIP relevance check (similarity >= quality_threshold)
    - Truncates to the top `target_count` images ranked by CLIP score if target_count is specified.
    """
    job_dir = os.path.join(output_dir, job_id)
    raw_dir = os.path.join(job_dir, "raw")
    filtered_dir = os.path.join(job_dir, "filtered")
    os.makedirs(filtered_dir, exist_ok=True)
    
    metadata_file = os.path.join(raw_dir, "metadata.json")
    if not os.path.exists(metadata_file):
        logger.error(f"No metadata.json found in raw folder: {raw_dir}")
        return 0, ""
        
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    # Get CLIP text embedding once
    text_features = None
    sample_features = None
    sample_image_path = None
    
    # Check if a sample image is present in the job directory
    for ext in [".jpg", ".jpeg", ".png"]:
        p = os.path.join(job_dir, f"sample_image{ext}")
        if os.path.exists(p):
            sample_image_path = p
            break
            
    if clip_filter_instance.initialized:
        if sample_image_path:
            try:
                logger.info(f"Loading sample image for visual similarity filtering: {sample_image_path}")
                sample_img = Image.open(sample_image_path)
                sample_features = clip_filter_instance.get_image_features(sample_img)
            except Exception as e:
                logger.error(f"Failed to load sample image features: {e}", exc_info=True)
        else:
            text_features = clip_filter_instance.get_text_features(query)
    else:
        logger.warning("CLIP filter not initialized. Skipping relevance filtering.")
        
    total_downloaded = len(metadata)
    duplicates_removed = 0
    blurry_removed = 0
    irrelevant_removed = 0
    too_small_removed = 0
    passed_count = 0
    
    seen_hashes: List[imagehash.ImageHash] = []
    
    # Pre-populate seen_hashes with already filtered images if appending and duplicates not allowed
    if not allow_duplicates and os.path.exists(filtered_dir):
        logger.info(f"Pre-populating seen hashes from existing filtered images in {filtered_dir}...")
        for f in os.listdir(filtered_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    existing_img = Image.open(os.path.join(filtered_dir, f))
                    seen_hashes.append(imagehash.phash(existing_img))
                except Exception as e:
                    logger.warning(f"Failed to read existing filtered image for duplicate check {f}: {e}")
                    
    # Track already passed images (append mode safety)
    existing_passed = set()
    if os.path.exists(filtered_dir):
        existing_passed = {f for f in os.listdir(filtered_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
        
    # Read existing report to preserve scores of already passed images
    existing_scores = {}
    report_file = os.path.join(job_dir, "filter_report.json")
    if os.path.exists(report_file):
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                old_report = json.load(f)
                for item in old_report.get("per_image", []):
                    existing_scores[item["filename"]] = {
                        "clip_score": item.get("clip_score", 1.0),
                        "blur_score": item.get("blur_score", 100.0)
                    }
        except Exception as e:
            logger.warning(f"Could not load existing filter report: {e}")
            
    per_image_report = []
    new_passed_candidates = []  # List of Tuple[filename, clip_score, blur_score, src_path]
    
    for filename, info in metadata.items():
        file_path_info = info["file_path"]
        if os.path.isabs(file_path_info):
            src_path = file_path_info
        else:
            src_path = os.path.join(raw_dir, file_path_info)
            
        if not os.path.exists(src_path):
            continue
            
        # If it was already passed in a previous run, preserve its status
        if filename in existing_passed:
            scores = existing_scores.get(filename, {"clip_score": 1.0, "blur_score": 100.0})
            per_image_report.append({
                "filename": filename,
                "status": "passed",
                "clip_score": scores["clip_score"],
                "blur_score": scores["blur_score"]
            })
            passed_count += 1
            continue
            
        status = "passed"
        clip_score = 1.0
        blur_score = 100.0
        
        try:
            # 1. Load image and size check
            img = Image.open(src_path)
            width, height = img.size
            if width < 100 or height < 100:
                status = "too_small"
                too_small_removed += 1
                
            # 2. Blur check (Laplacian variance)
            if status == "passed":
                # Convert PIL to grayscale numpy array for OpenCV
                gray_img = img.convert("L")
                gray_np = np.array(gray_img)
                # Compute Laplacian variance
                blur_score = float(cv2.Laplacian(gray_np, cv2.CV_64F).var())
                if blur_score < 10.0:
                    status = "blurry"
                    blurry_removed += 1
                    
            # 3. Deduplication check (perceptual hash)
            if status == "passed" and not allow_duplicates:
                phash = imagehash.phash(img)
                is_duplicate = False
                for seen_hash in seen_hashes:
                    if phash - seen_hash <= 8:
                        is_duplicate = True
                        break
                if is_duplicate:
                    status = "duplicate"
                    duplicates_removed += 1
                else:
                    seen_hashes.append(phash)
                    
            # 4. CLIP Relevance Check
            if status == "passed" and (text_features is not None or sample_features is not None):
                if sample_features is not None:
                    clip_score = float(clip_filter_instance.get_image_to_image_similarity(img, sample_features))
                else:
                    clip_score = float(clip_filter_instance.get_similarity_to_features(img, text_features))
                    
                if clip_score < quality_threshold:
                    status = "irrelevant"
                    irrelevant_removed += 1
                    
            # 5. Handle passed new images as candidates (to allow truncation later)
            if status == "passed":
                new_passed_candidates.append((filename, clip_score, blur_score, src_path))
            else:
                per_image_report.append({
                    "filename": filename,
                    "status": status,
                    "clip_score": clip_score,
                    "blur_score": blur_score
                })
                
        except Exception as e:
            logger.error(f"Error processing image {filename}: {e}", exc_info=True)
            status = "failed_check"
            per_image_report.append({
                "filename": filename,
                "status": status,
                "clip_score": 0.0,
                "blur_score": 0.0
            })
            
    # Truncate new passed candidates if target_count is specified
    excess_discarded_removed = 0
    if target_count is not None:
        remaining_slots = max(0, target_count - len(existing_passed))
        if len(new_passed_candidates) > remaining_slots:
            # Sort by CLIP score descending
            new_passed_candidates.sort(key=lambda x: x[1], reverse=True)
            accepted = new_passed_candidates[:remaining_slots]
            discarded = new_passed_candidates[remaining_slots:]
            logger.info(f"Target count is {target_count} (existing: {len(existing_passed)}). Truncating {len(new_passed_candidates)} candidates: keeping top {len(accepted)}, discarding {len(discarded)} excess.")
        else:
            accepted = new_passed_candidates
            discarded = []
    else:
        accepted = new_passed_candidates
        discarded = []

    for filename, clip_score, blur_score, src_path in accepted:
        passed_count += 1
        dest_path = os.path.join(filtered_dir, filename)
        # Atomic copy: copy to temp and rename
        temp_dest = dest_path + ".tmp"
        shutil.copy2(src_path, temp_dest)
        os.replace(temp_dest, dest_path)
        
        per_image_report.append({
            "filename": filename,
            "status": "passed",
            "clip_score": clip_score,
            "blur_score": blur_score
        })
        
    for filename, clip_score, blur_score, src_path in discarded:
        excess_discarded_removed += 1
        per_image_report.append({
            "filename": filename,
            "status": "excess_discarded",
            "clip_score": clip_score,
            "blur_score": blur_score
        })
            
    # Save filter report
    report_data = {
        "total_downloaded": total_downloaded,
        "duplicates_removed": duplicates_removed,
        "blurry_removed": blurry_removed,
        "irrelevant_removed": irrelevant_removed,
        "too_small_removed": too_small_removed,
        "excess_discarded_removed": excess_discarded_removed,
        "passed": passed_count,
        "per_image": per_image_report
    }
    
    temp_report = report_file + ".tmp"
    with open(temp_report, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    os.replace(temp_report, report_file)
    
    logger.info(f"Filtering finished: {passed_count}/{total_downloaded} total passed (New Excess Discarded: {excess_discarded_removed}).")
    return passed_count, report_file
