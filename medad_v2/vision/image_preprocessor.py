"""
MEDAD 2.0 - Image Preprocessor
=================================

Image preprocessing utilities for the vision module.
"""

from typing import Tuple, Optional
import io
import logging

logger = logging.getLogger("ImagePreprocessor")


class ImagePreprocessor:
    """
    Image preprocessing utilities for medical image analysis.
    
    Handles:
    - Format conversion
    - Resizing
    - Normalization
    - Quality assessment
    - Augmentation for inference
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        Initialize the preprocessor.
        
        Args:
            target_size: Target image size (height, width)
            normalize_mean: Normalization mean (RGB)
            normalize_std: Normalization std (RGB)
        """
        self.target_size = target_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
    
    def load_image(self, image_data: bytes):
        """Load image from bytes"""
        from PIL import Image
        return Image.open(io.BytesIO(image_data))
    
    def assess_quality(self, image) -> Tuple[str, dict]:
        """
        Assess image quality for skin analysis.
        
        Returns:
            Tuple of (quality_rating, quality_metrics)
        """
        width, height = image.size
        
        metrics = {
            "resolution": (width, height),
            "aspect_ratio": width / height if height > 0 else 0,
            "is_color": image.mode in ("RGB", "RGBA"),
        }
        
        # Calculate quality score
        score = 0
        
        # Resolution check
        if width >= 512 and height >= 512:
            score += 3
        elif width >= 256 and height >= 256:
            score += 2
        elif width >= 128 and height >= 128:
            score += 1
        
        # Aspect ratio check (prefer square-ish images for skin)
        if 0.5 <= metrics["aspect_ratio"] <= 2.0:
            score += 1
        
        # Color check
        if metrics["is_color"]:
            score += 1
        
        # Determine quality rating
        if score >= 4:
            quality = "good"
        elif score >= 2:
            quality = "fair"
        else:
            quality = "poor"
        
        metrics["quality_score"] = score
        
        return quality, metrics
    
    def preprocess(
        self,
        image,
        apply_augmentation: bool = False
    ):
        """
        Preprocess image for model input.
        
        Args:
            image: PIL Image
            apply_augmentation: Whether to apply test-time augmentation
        
        Returns:
            Preprocessed image tensor
        """
        import numpy as np
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize
        from PIL import Image as PILImage
        image = image.resize(self.target_size, PILImage.LANCZOS)
        
        # Convert to numpy
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # Normalize
        mean = np.array(self.normalize_mean)
        std = np.array(self.normalize_std)
        image_array = (image_array - mean) / std
        
        return image_array
    
    def apply_augmentations(self, image):
        """
        Apply test-time augmentations.
        
        Returns multiple versions of the image for ensemble prediction.
        """
        augmented = [image]
        
        # Horizontal flip
        augmented.append(image.transpose(0))
        
        # Slight rotations
        # (Add rotation augmentations if needed)
        
        return augmented
