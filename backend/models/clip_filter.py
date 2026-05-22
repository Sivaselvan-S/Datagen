import logging
import torch
from PIL import Image
from typing import List, Tuple, Optional
from transformers import CLIPProcessor, CLIPModel

logger = logging.getLogger("clip_filter")

class CLIPFilter:
    _instance: Optional['CLIPFilter'] = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CLIPFilter, cls).__new__(cls, *args, **kwargs)
            cls._instance.initialized = False
        return cls._instance
    
    def initialize(self, model_name: str = "openai/clip-vit-base-patch32"):
        if self.initialized:
            return
        
        logger.info(f"Loading CLIP model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.initialized = True
            self.model.eval()
            logger.info(f"CLIP model loaded successfully on device: {self.device}")
        except Exception as e:
            logger.error(f"Error loading CLIP model: {e}", exc_info=True)
            self.initialized = False
            raise e

    def get_similarity(self, image: Image.Image, text: str) -> float:
        """Calculate cosine similarity between image and text."""
        if not self.initialized:
            logger.warning("CLIP model not loaded. Skipping relevance check.")
            return 0.0

        try:
            # Prepare inputs
            inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                # Extract normalized features
                image_features = self.model.get_image_features(pixel_values=inputs['pixel_values'])
                text_features = self.model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarity = (image_features * text_features).sum(dim=-1).item()
                # Normalize raw CLIP similarity from [0.15, 0.35] to [0.0, 1.0]
                normalized_similarity = (similarity - 0.15) / 0.20
                return max(0.0, min(1.0, normalized_similarity))
        except Exception as e:
            logger.error(f"Error computing CLIP similarity: {e}", exc_info=True)
            return 0.0

    def get_text_features(self, text: str) -> Optional[torch.Tensor]:
        """Encodes query text once and returns its normalized feature vector."""
        if not self.initialized:
            return None
        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features
        except Exception as e:
            logger.error(f"Error encoding text features: {e}", exc_info=True)
            return None

    def get_similarity_to_features(self, image: Image.Image, text_features: torch.Tensor) -> float:
        """Computes cosine similarity of an image against pre-computed normalized text features."""
        if not self.initialized or text_features is None:
            return 0.0
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(pixel_values=inputs['pixel_values'])
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                similarity = (image_features * text_features).sum(dim=-1).item()
                # Normalize raw CLIP similarity from [0.15, 0.35] to [0.0, 1.0]
                normalized_similarity = (similarity - 0.15) / 0.20
                return max(0.0, min(1.0, normalized_similarity))
        except Exception as e:
            logger.error(f"Error computing CLIP similarity to features: {e}", exc_info=True)
            return 0.0

    def get_image_features(self, image: Image.Image) -> Optional[torch.Tensor]:
        """Encodes an image once and returns its normalized feature vector."""
        if not self.initialized:
            return None
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(pixel_values=inputs['pixel_values'])
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features
        except Exception as e:
            logger.error(f"Error encoding image features: {e}", exc_info=True)
            return None

    def get_image_to_image_similarity(self, image: Image.Image, sample_features: torch.Tensor) -> float:
        """Computes cosine similarity between an image and pre-computed normalized sample image features."""
        if not self.initialized or sample_features is None:
            return 0.0
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(pixel_values=inputs['pixel_values'])
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                similarity = (image_features * sample_features).sum(dim=-1).item()
                # Normalize raw CLIP image-to-image similarity:
                # Two highly similar/matching images generally score >= 0.70. Unrelated score ~0.50.
                # Let's map [0.55, 0.90] to [0.0, 1.0] for a sensitive response curve.
                normalized_similarity = (similarity - 0.55) / 0.35
                return max(0.0, min(1.0, normalized_similarity))
        except Exception as e:
            logger.error(f"Error computing CLIP image-to-image similarity: {e}", exc_info=True)
            return 0.0

    def classify_image(self, image: Image.Image, labels: List[str]) -> Tuple[str, float]:
        """Perform zero-shot classification of an image against candidate labels."""
        if not self.initialized or not labels:
            return "unlabeled", 0.0

        try:
            inputs = self.processor(text=labels, images=image, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image  # image-text similarity score
                probs = logits_per_image.softmax(dim=-1)
                
            best_idx = probs[0].argmax().item()
            confidence = probs[0][best_idx].item()
            return labels[best_idx], confidence
        except Exception as e:
            logger.error(f"Error in zero-shot classification: {e}", exc_info=True)
            return "unlabeled", 0.0

# Global CLIP filter singleton instance
clip_filter_instance = CLIPFilter()
