import re
import uuid
import logging
import os
from typing import Dict, Any, Tuple

logger = logging.getLogger("input_parser")

def parse_input(
    query: str,
    count: int,
    label: bool,
    label_type: str,
    export_format: str,
    quality_threshold: float,
    target_labels: str = "",
    allow_duplicates: bool = False,
    override_job_id: str = None
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Parses natural language query and extra options to build a validated job configuration.
    Extracts the image count from the query if present.
    """
    # 1. Extract count from query using Regex
    # Matches patterns like: "50 images of", "100 pcs of", "20 photos", "10 cats"
    # Look for any number at the beginning or before nouns
    count_pattern = r'\b(\d+)\b'
    match = re.search(count_pattern, query)
    
    parsed_count = count
    if match:
        extracted = int(match.group(1))
        # Ensure it falls into a reasonable range, e.g., 1 to 5000
        if 1 <= extracted <= 5000:
            parsed_count = extracted
            logger.info(f"Extracted count {parsed_count} from query: '{query}'")
            
    # 2. Clean query string for search engines
    # Remove phrases like: "100 images of", "images of", "photos of", "pictures of", etc.
    cleaned_query = query
    # Remove the extracted count digits
    if match:
        cleaned_query = cleaned_query.replace(match.group(0), "", 1)
        
    remove_patterns = [
        r'\bimages\s+of\b',
        r'\bimage\s+of\b',
        r'\bphotos\s+of\b',
        r'\bphoto\s+of\b',
        r'\bpictures\s+of\b',
        r'\bpicture\s+of\b',
        r'\bcollected\s+of\b',
        r'\bdataset\s+of\b',
        r'\bdataset\b',
        r'\bimages\b',
        r'\bphotos\b',
        r'\bpictures\b',
        r'\bof\b'
    ]
    
    for pattern in remove_patterns:
        cleaned_query = re.sub(pattern, '', cleaned_query, flags=re.IGNORECASE)
        
    # Clean up double spaces, leading/trailing punctuation/spaces
    cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
    cleaned_query = re.sub(r'^[^\w]+|[^\w]+$', '', cleaned_query).strip()
    
    # If cleaning results in empty string, revert to original query
    if not cleaned_query:
        cleaned_query = query
        
    # 3. Clean and split target labels
    parsed_target_labels = []
    if target_labels:
        parsed_target_labels = [lbl.strip() for lbl in target_labels.split(",") if lbl.strip()]
    else:
        # Fallback to query noun if label is empty
        # We can treat the entire cleaned query as the single target label
        parsed_target_labels = [cleaned_query]

    # Create job configuration
    config = {
        "query": query,
        "count": parsed_count,
        "label": label,
        "label_type": label_type.lower(),
        "export_format": export_format.lower(),
        "quality_threshold": float(quality_threshold),
        "target_labels": parsed_target_labels,
        "allow_duplicates": allow_duplicates
    }
    
    # Generate job_id
    if override_job_id:
        sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '_', override_job_id.strip())
        job_id = re.sub(r'_+', '_', sanitized).strip('_').lower()
        if not job_id:
            job_id = "custom_dataset"
    else:
        # Generate a descriptive, path-safe job_id based on the first target label
        base_label = parsed_target_labels[0] if parsed_target_labels else cleaned_query
        sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '_', base_label.strip())
        sanitized = re.sub(r'_+', '_', sanitized).strip('_').lower()
        if not sanitized:
            sanitized = "dataset"
            
        out_dir = os.path.abspath(os.getenv("OUTPUT_DIR", "./outputs"))
        job_id = sanitized
        counter = 1
        while os.path.exists(os.path.join(out_dir, job_id)):
            job_id = f"{sanitized}_{counter}"
            counter += 1
    
    logger.info(f"Job Config created: {config}")
    return job_id, cleaned_query, config
