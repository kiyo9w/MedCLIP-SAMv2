"""
Wavelet-based Frequency-Aware Preprocessing Module
Based on FMISeg's wavelet transform approach (utils/wave.py)

This module performs Discrete Wavelet Transform (DWT) on medical images
to extract frequency components that enhance boundary detection.

Architecture:
- Input: Medical image (RGB or Grayscale)
- Output: 
  * LL: Low-frequency component (image approximation)
  * High-freq: Merged high-frequency components (LH + HL + HH)
            representing edges and boundaries
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import pywt
from typing import Tuple, Union, Optional
import cv2


class WaveletTransform(nn.Module):
    """
    Discrete Wavelet Transform (DWT) for image preprocessing.
    
    This module computes the 2D DWT of an image to extract frequency components.
    The high-frequency components (LH, HL, HH) contain edge and boundary information
    that complements BiomedCLIP's semantic feature extraction.
    """
    
    def __init__(self, wavelet_type: str = 'haar', normalize: bool = True):
        """
        Args:
            wavelet_type: Type of wavelet ('haar', 'db2', 'bior1.5', etc.)
            normalize: Whether to normalize components to [0, 255] range
        """
        super().__init__()
        self.wavelet_type = wavelet_type
        self.normalize = normalize
        
    def forward(self, image: Union[np.ndarray, torch.Tensor]) -> dict:
        """
        Perform 2D DWT on input image.
        
        Args:
            image: Input image (numpy array or torch tensor)
                   Shape: (H, W) for grayscale or (H, W, C) for RGB
        
        Returns:
            dict with keys:
                'll': Low-frequency approximation
                'lh': Horizontal high-frequency component
                'hl': Vertical high-frequency component
                'hh': Diagonal high-frequency component
                'high_freq_merged': LH + HL + HH (combined boundary info)
        """
        # Convert to numpy if needed
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # Convert RGB to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif len(image.shape) == 3:
            image = image.squeeze()
        
        # Ensure proper data type
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Perform 2D DWT
        coeffs = pywt.dwt2(image, self.wavelet_type)
        LL, (LH, HL, HH) = coeffs
        
        # Normalize if requested
        if self.normalize:
            LL = self._normalize_component(LL)
            LH = self._normalize_component(LH)
            HL = self._normalize_component(HL)
            HH = self._normalize_component(HH)
        
        # Merge high-frequency components
        high_freq_merged = (LH + HL + HH) / 3.0
        if self.normalize:
            high_freq_merged = self._normalize_component(high_freq_merged)
        
        return {
            'll': LL,
            'lh': LH,
            'hl': HL,
            'hh': HH,
            'high_freq_merged': high_freq_merged
        }
    
    @staticmethod
    def _normalize_component(component: np.ndarray) -> np.ndarray:
        """Normalize component to [0, 255] range."""
        min_val = component.min()
        max_val = component.max()
        if max_val > min_val:
            normalized = (component - min_val) / (max_val - min_val) * 255
        else:
            normalized = np.zeros_like(component)
        return normalized.astype(np.float32)


class DualStreamPreprocessor(nn.Module):
    """
    Dual-Stream Preprocessing for Frequency-Aware Integration.
    
    This preprocessor creates two processing streams:
    1. Standard stream: Original image normalized for BiomedCLIP
    2. Frequency-aware stream: Wavelet-transformed components for boundary detection
    
    The high-frequency components from stream 2 will be injected into the Vision Encoder
    to enhance boundary detection during saliency map generation.
    """
    
    def __init__(
        self, 
        wavelet_type: str = 'haar',
        image_size: int = 224,
        normalize_std: Tuple[float, float, float] = (0.26862954, 0.26130256, 0.27577711),
        normalize_mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    ):
        """
        Args:
            wavelet_type: Type of wavelet for DWT
            image_size: Size to resize images to
            normalize_std: Standard deviation for normalization (BiomedCLIP defaults)
            normalize_mean: Mean for normalization (BiomedCLIP defaults)
        """
        super().__init__()
        self.wavelet_transform = WaveletTransform(wavelet_type=wavelet_type)
        self.image_size = image_size
        self.normalize_mean = np.array(normalize_mean)
        self.normalize_std = np.array(normalize_std)
        
    def forward(
        self, 
        image: Union[np.ndarray, torch.Tensor, Image.Image]
    ) -> dict:
        """
        Process image through both streams.
        
        Args:
            image: Input medical image
        
        Returns:
            dict with keys:
                'original_stream': Normalized image for BiomedCLIP (RGB, [0, 1] or normalized)
                'wavelet_components': Dict of wavelet coefficients
                'high_freq_enhanced': Enhanced high-frequency map for feature injection
        """
        # Convert to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # Ensure 3 channels for BiomedCLIP (RGB)
        if len(image.shape) == 2:
            image_rgb = np.stack([image, image, image], axis=-1)
        elif image.shape[2] == 1:
            image_rgb = np.repeat(image, 3, axis=2)
        else:
            image_rgb = image
        
        # Stream 1: Standard preprocessing for BiomedCLIP
        original_stream = self._preprocess_original(image_rgb)
        
        # Stream 2: Wavelet analysis for frequency-aware injection
        wavelet_components = self.wavelet_transform(image)
        high_freq_enhanced = self._enhance_high_frequency(wavelet_components)
        
        return {
            'original_stream': original_stream,
            'wavelet_components': wavelet_components,
            'high_freq_enhanced': high_freq_enhanced,
            'raw_image': image_rgb
        }
    
    def _preprocess_original(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for BiomedCLIP using standard normalization.
        
        Args:
            image: RGB image (H, W, 3)
        
        Returns:
            Normalized tensor ready for BiomedCLIP
        """
        # Resize
        if image.shape[:2] != (self.image_size, self.image_size):
            image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        
        # Apply standard normalization
        image = (image - self.normalize_mean) / self.normalize_std
        
        # Convert to tensor (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        return image
    
    def _enhance_high_frequency(self, wavelet_components: dict) -> torch.Tensor:
        """
        Enhance high-frequency components for boundary detection.
        
        Args:
            wavelet_components: Dict with 'll', 'lh', 'hl', 'hh' components
        
        Returns:
            Enhanced high-frequency tensor for feature injection
        """
        high_freq = wavelet_components['high_freq_merged']
        
        # Resize to standard size
        if high_freq.shape != (self.image_size, self.image_size):
            high_freq = cv2.resize(high_freq, (self.image_size, self.image_size))
        
        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        high_freq_enhanced = clahe.apply(high_freq.astype(np.uint8)).astype(np.float32)
        
        # Normalize to [0, 1]
        high_freq_enhanced = high_freq_enhanced / 255.0
        
        # Convert to tensor and expand to 3 channels
        high_freq_tensor = torch.from_numpy(high_freq_enhanced).float()
        high_freq_tensor = high_freq_tensor.unsqueeze(0).repeat(3, 1, 1)
        
        return high_freq_tensor


# Utility functions for batch processing
def process_batch_dual_stream(
    images: Union[list, torch.Tensor],
    preprocessor: DualStreamPreprocessor,
    device: str = 'cuda'
) -> dict:
    """
    Process a batch of images through dual-stream preprocessing.
    
    Args:
        images: List of image paths or tensor of images
        preprocessor: DualStreamPreprocessor instance
        device: Device to move tensors to
    
    Returns:
        Batched tensors for both streams
    """
    original_batch = []
    high_freq_batch = []
    
    for image in images:
        result = preprocessor(image)
        original_batch.append(result['original_stream'].unsqueeze(0))
        high_freq_batch.append(result['high_freq_enhanced'].unsqueeze(0))
    
    original_batch = torch.cat(original_batch, dim=0).to(device)
    high_freq_batch = torch.cat(high_freq_batch, dim=0).to(device)
    
    return {
        'original_stream': original_batch,
        'high_freq_stream': high_freq_batch
    }
