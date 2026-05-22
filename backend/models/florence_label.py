import logging
import torch
import re
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoProcessor, AutoModelForCausalLM
from backend.models.clip_filter import clip_filter_instance

logger = logging.getLogger("florence_label")

def estimate_confidence(image: Image.Image, box: Tuple[float, float, float, float], label: str, default_conf: float = 0.90) -> float:
    if not clip_filter_instance.initialized:
        return default_conf
    try:
        w, h = image.size
        x_min, y_min, x_max, y_max = box
        x_min_c = max(0, min(w - 1, int(x_min)))
        y_min_c = max(0, min(h - 1, int(y_min)))
        x_max_c = max(x_min_c + 1, min(w, int(x_max)))
        y_max_c = max(y_min_c + 1, min(h, int(y_max)))
        
        cropped_img = image.crop((x_min_c, y_min_c, x_max_c, y_max_c))
        if cropped_img.width > 0 and cropped_img.height > 0:
            sim = clip_filter_instance.get_similarity(cropped_img, label)
            return float(sim)
    except Exception as e:
        logger.error(f"Error estimating confidence: {e}")
    return default_conf

class FlorenceLabeler:
    _instance: Optional['FlorenceLabeler'] = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(FlorenceLabeler, cls).__new__(cls, *args, **kwargs)
            cls._instance.initialized = False
        return cls._instance
        
    def initialize(self, model_name: str = "microsoft/Florence-2-base"):
        if self.initialized:
            return
            
        logger.info(f"Loading Florence-2 model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Florence-2 models require flash-attn on GPU if possible, but CPU is standard.
        # We will use torch_dtype=torch.float32 for CPU to avoid half precision issues.
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=self.torch_dtype
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.initialized = True
            self.model.eval()
            logger.info(f"Florence-2 model loaded successfully on device: {self.device}")
        except Exception as e:
            logger.error(f"Error loading Florence-2 model: {e}", exc_info=True)
            self.initialized = False
            raise e

    def run_prompt(self, image: Image.Image, task_prompt: str, text_input: Optional[str] = None) -> Any:
        """Helper to run Florence-2 generation with a prompt."""
        if not self.initialized:
            logger.warning("Florence-2 not initialized.")
            return None
            
        try:
            # Ensure RGB image
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            prompt = task_prompt
            if text_input:
                prompt += text_input
                
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(self.device).to(self.torch_dtype) if v.dtype == torch.float32 and self.device == "cuda" else v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False
                )
                
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self.processor.post_process_generation(
                generated_text, 
                task=task_prompt, 
                image_size=(image.width, image.height)
            )
            return parsed_answer
        except Exception as e:
            logger.error(f"Error in Florence-2 generation: {e}", exc_info=True)
            return None

    def detect_objects(self, image: Image.Image, target_labels: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Runs Florence-2 object detection and maps the result to standard list of bboxes."""
        if target_labels:
            detections = []
            for label in target_labels:
                result = self.run_prompt(image, "<CAPTION_TO_PHRASE_GROUNDING>", text_input=label)
                if not result or "<CAPTION_TO_PHRASE_GROUNDING>" not in result:
                    continue
                
                data = result["<CAPTION_TO_PHRASE_GROUNDING>"]
                bboxes = data.get("bboxes", [])
                # Grounding matches the specified query label to these coordinates
                for box in bboxes:
                    x_min, y_min, x_max, y_max = box
                    conf = estimate_confidence(image, (x_min, y_min, x_max, y_max), label, default_conf=0.90)
                    detections.append({
                        "label": label,
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                        "confidence": conf
                    })
            return detections
        else:
            result = self.run_prompt(image, "<OD>")
            if not result or "<OD>" not in result:
                return []
                
            data = result["<OD>"]
            bboxes = data.get("bboxes", [])
            labels = data.get("labels", [])
            
            detections = []
            for box, label in zip(bboxes, labels):
                x_min, y_min, x_max, y_max = box
                conf = estimate_confidence(image, (x_min, y_min, x_max, y_max), label, default_conf=0.90)
                detections.append({
                    "label": label,
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "confidence": conf
                })
            return detections

    def segment_objects(self, image: Image.Image, phrase: str) -> List[Dict[str, Any]]:
        """Runs Florence-2 referring expression segmentation for a specific target phrase."""
        # Task prompt is <REFERRING_EXPRESSION_SEGMENTATION>
        result = self.run_prompt(image, "<REFERRING_EXPRESSION_SEGMENTATION>", text_input=phrase)
        if not result or "<REFERRING_EXPRESSION_SEGMENTATION>" not in result:
            return []
            
        data = result["<REFERRING_EXPRESSION_SEGMENTATION>"]
        polygons = data.get("polygons", [])
        labels = data.get("labels", [])
        
        # In referring expression segmentation, bboxes are also usually returned or can be computed from polygons
        segmentations = []
        for polys, label in zip(polygons, labels):
            # polys is a list of lists: [[x1, y1, x2, y2, ...]]
            # Let's extract bounding box coordinates from polygons
            all_pts = []
            for poly in polys:
                # flatten list of points
                for i in range(0, len(poly), 2):
                    if i + 1 < len(poly):
                        all_pts.append((poly[i], poly[i+1]))
                        
            if not all_pts:
                continue
                
            x_coords = [p[0] for p in all_pts]
            y_coords = [p[1] for p in all_pts]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            conf = estimate_confidence(image, (x_min, y_min, x_max, y_max), phrase, default_conf=0.85)
            segmentations.append({
                "label": label or phrase,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "polygons": polys,  # List[List[float]]
                "confidence": conf
            })
        return segmentations

# Global Florence-2 labeler singleton instance
florence_labeler_instance = FlorenceLabeler()
