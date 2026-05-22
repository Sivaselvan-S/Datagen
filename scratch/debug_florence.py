import os
import logging
from PIL import Image
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_florence")

load_dotenv()

from backend.models.florence_label import florence_labeler_instance

def main():
    logger.info("Initializing Florence-2...")
    florence_labeler_instance.initialize()
    
    # Locate the downloaded images from the last job
    job_id = "f0e103f5-ee3e-4eb8-bf30-0b41fdb9a0b4"
    filtered_dir = f"./outputs/{job_id}/filtered"
    
    if not os.path.exists(filtered_dir):
        logger.error(f"Filtered directory not found at: {filtered_dir}")
        return
        
    images = [f for f in os.listdir(filtered_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    logger.info(f"Found {len(images)} images to test.")
    
    for filename in images:
        img_path = os.path.join(filtered_dir, filename)
        logger.info(f"\n--- Testing Image: {filename} ---")
        img = Image.open(img_path)
        
        # Test captioning
        caption_res = florence_labeler_instance.run_prompt(img, "<CAPTION>")
        logger.info(f"CAPTION result: {caption_res}")
        
        # Test raw OD model output
        task_prompt = "<OD>"
        inputs = florence_labeler_instance.processor(text=task_prompt, images=img, return_tensors="pt").to(florence_labeler_instance.device)
        import torch
        with torch.no_grad():
            generated_ids = florence_labeler_instance.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )
        generated_text = florence_labeler_instance.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        logger.info(f"Raw generated text for <OD>: {repr(generated_text)}")
        
        parsed_answer = florence_labeler_instance.processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(img.width, img.height)
        )
        logger.info(f"Post-processed OD: {parsed_answer}")


if __name__ == "__main__":
    main()
