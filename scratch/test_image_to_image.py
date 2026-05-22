import os
import sys
import logging
from PIL import Image
import torch

# Ensure path includes root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.models.clip_filter import clip_filter_instance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_img2img")

def main():
    logger.info("Initializing CLIP filter...")
    clip_filter_instance.initialize()
    
    # We will use some images from the outputs/cup/raw folder
    raw_dir = "outputs/cup/raw"
    if not os.path.exists(raw_dir):
        logger.error(f"Directory {raw_dir} does not exist. Run test_pipeline.py first.")
        return
        
    images = sorted([f for f in os.listdir(raw_dir) if f.endswith(".jpg")])
    if len(images) < 2:
        logger.error("Not enough images in raw dir to perform similarity test.")
        return
        
    # Let's treat the first image as our "sample image" (reference)
    ref_filename = images[0]
    ref_path = os.path.join(raw_dir, ref_filename)
    logger.info(f"Using {ref_filename} as the reference sample image.")
    
    ref_img = Image.open(ref_path)
    # Get reference features
    ref_features = clip_filter_instance.get_image_features(ref_img)
    
    if ref_features is None:
        logger.error("Failed to extract features for reference image.")
        return

    # Download an unrelated image (e.g., a random pic)
    unrelated_path = os.path.join(raw_dir, "unrelated.jpg")
    try:
        import urllib.request
        logger.info("Downloading an unrelated image from picsum.photos...")
        # Use a user agent to avoid 403
        req = urllib.request.Request(
            "https://picsum.photos/300/300", 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req) as response:
            with open(unrelated_path, "wb") as f:
                f.write(response.read())
        logger.info("Unrelated image downloaded successfully.")
        if "unrelated.jpg" not in images:
            images.append("unrelated.jpg")
    except Exception as e:
        logger.warning(f"Could not download unrelated image: {e}")
        
    logger.info("Comparing other images with the reference image:")
    for img_name in images:
        img_path = os.path.join(raw_dir, img_name)
        img = Image.open(img_path)
        
        # Calculate raw cosine similarity and normalized similarity
        inputs = clip_filter_instance.processor(images=img, return_tensors="pt").to(clip_filter_instance.device)
        with torch.no_grad():
            img_features = clip_filter_instance.model.get_image_features(pixel_values=inputs['pixel_values'])
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            raw_sim = (img_features * ref_features).sum(dim=-1).item()
            
        norm_sim = clip_filter_instance.get_image_to_image_similarity(img, ref_features)
        
        logger.info(f"Image: {img_name}")
        logger.info(f"  - Raw Cosine Similarity: {raw_sim:.4f}")
        logger.info(f"  - Normalized Similarity (0-1): {norm_sim:.4f}")
        
        # Check against different threshold values
        for thresh in [0.4, 0.6, 0.8]:
            passed = "PASSED" if norm_sim >= thresh else "REJECTED"
            logger.info(f"  - Cutoff {thresh:.2f}: {passed}")

if __name__ == "__main__":
    main()
