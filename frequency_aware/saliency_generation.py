"""
Refined Saliency Map Generation with Frequency-Aware Enhancement

This module generates high-quality saliency maps by leveraging both semantic
information from BiomedCLIP and boundary information from wavelet decomposition.

Key improvements over standard saliency map generation:
1. Gradient computation includes frequency-aware feature information
2. Edge-aware thresholding to preserve boundary accuracy
3. Morphological operations to reduce noise while maintaining precision
4. Multi-scale saliency aggregation for robust results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from PIL import Image


class FrequencyAwareSaliencyGenerator(nn.Module):
    """
    Generates refined saliency maps that account for both semantic and boundary information.
    
    Architecture:
    1. Compute standard CLIP-based gradients for semantic information
    2. Enhance with high-frequency component gradients for boundary sharpness
    3. Apply adaptive thresholding based on frequency content
    4. Perform morphological operations to clean up results
    """
    
    def __init__(
        self,
        blur_kernel: int = 5,
        morphology_kernel: int = 5,
        frequency_weight: float = 0.3,
        adaptive_threshold: bool = True
    ):
        """
        Args:
            blur_kernel: Kernel size for post-processing blurring
            morphology_kernel: Kernel size for morphological operations
            frequency_weight: Weight for frequency-aware enhancement (0.0 to 1.0)
            adaptive_threshold: Whether to use adaptive thresholding
        """
        super().__init__()
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.morphology_kernel = morphology_kernel if morphology_kernel % 2 == 1 else morphology_kernel + 1
        self.frequency_weight = frequency_weight
        self.adaptive_threshold = adaptive_threshold
        
        # Morphological kernels
        self.erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphology_kernel, morphology_kernel))
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphology_kernel, morphology_kernel))
    
    def forward(
        self,
        image_features: torch.Tensor,
        high_freq_features: torch.Tensor,
        text_embedding: torch.Tensor,
        image_tensor: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Dict:
        """
        Generate refined saliency map.
        
        Args:
            image_features: Features from vision encoder (B, num_patches+1, embedding_dim)
            high_freq_features: High-frequency wavelet features (B, num_patches, embedding_dim)
            text_embedding: Text embedding from language model (1, embedding_dim) or (embedding_dim,)
            image_tensor: Original image tensor for gradient computation (B, 3, H, W)
            target_size: Target size for output saliency map (H, W)
        
        Returns:
            Dict containing:
                'saliency_map': Raw saliency map [0, 1]
                'saliency_map_refined': Refined saliency map with edges
                'binary_mask': Binary segmentation mask
                'confidence_map': Confidence scores for each pixel
        """
        batch_size = image_features.shape[0]
        device = image_features.device
        
        # Normalize embeddings
        image_features_norm = F.normalize(image_features, dim=-1)
        text_embedding_norm = F.normalize(text_embedding.unsqueeze(0), dim=-1)
        
        # Compute similarity scores
        semantic_similarity = torch.matmul(image_features_norm, text_embedding_norm.squeeze(0).unsqueeze(-1)).squeeze(-1)
        
        # Compute frequency-aware saliency enhancement
        if high_freq_features is not None:
            high_freq_features_norm = F.normalize(high_freq_features, dim=-1)
            freq_similarity = torch.matmul(high_freq_features_norm, text_embedding_norm.squeeze(0).unsqueeze(-1)).squeeze(-1)
            
            # Combine semantic and frequency information
            # High-frequency features help pinpoint boundaries
            combined_similarity = semantic_similarity[:, 1:, :] * (1 - self.frequency_weight) + \
                                 freq_similarity * self.frequency_weight
            combined_similarity = torch.cat([semantic_similarity[:, :1, :], combined_similarity], dim=1)
        else:
            combined_similarity = semantic_similarity
        
        # Reshape to spatial dimensions
        saliency_maps = self._reshape_to_spatial(combined_similarity, batch_size, image_tensor.shape[-2:])
        
        # Apply refinement
        refined_maps = []
        for i in range(batch_size):
            saliency = saliency_maps[i].cpu().numpy()
            refined = self._refine_saliency_map(saliency, high_freq_features is not None)
            refined_maps.append(refined)
        
        refined_maps = np.stack(refined_maps, axis=0)
        
        # Resize to target size if provided
        if target_size is not None:
            refined_maps = np.array([
                cv2.resize(m, target_size) for m in refined_maps
            ])
        
        # Generate binary masks
        binary_masks = self._generate_binary_masks(refined_maps)
        
        # Compute confidence maps
        confidence_maps = self._compute_confidence(refined_maps)
        
        return {
            'saliency_map': torch.from_numpy(saliency_maps).float().to(device),
            'saliency_map_refined': torch.from_numpy(refined_maps).float().to(device),
            'binary_mask': torch.from_numpy(binary_masks).float().to(device),
            'confidence_map': torch.from_numpy(confidence_maps).float().to(device)
        }
    
    def _reshape_to_spatial(
        self,
        similarity: torch.Tensor,
        batch_size: int,
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Reshape similarity scores from patch representation to spatial map.
        
        Args:
            similarity: (B, num_patches+1, 1)
            batch_size: Batch size
            image_size: Original image size (H, W)
        
        Returns:
            Spatial saliency map (B, H, W)
        """
        # Remove class token contribution (first token)
        patch_similarity = similarity[:, 1:, 0]  # (B, num_patches)
        
        # Reshape to spatial grid
        # Assuming 14x14 patches for 224x224 image
        num_patches = patch_similarity.shape[1]
        grid_size = int(np.sqrt(num_patches))
        
        spatial_maps = patch_similarity.reshape(batch_size, grid_size, grid_size).cpu().numpy()
        
        # Upsample to original image size using interpolation
        saliency_maps = []
        for spatial_map in spatial_maps:
            upsampled = cv2.resize(spatial_map, image_size, interpolation=cv2.INTER_CUBIC)
            upsampled = (upsampled - upsampled.min()) / (upsampled.max() - upsampled.min() + 1e-5)
            saliency_maps.append(upsampled)
        
        return np.stack(saliency_maps, axis=0)
    
    def _refine_saliency_map(self, saliency: np.ndarray, has_freq_info: bool = True) -> np.ndarray:
        """
        Refine saliency map to enhance boundaries and reduce noise.
        
        Args:
            saliency: Saliency map [0, 1]
            has_freq_info: Whether frequency information was used
        
        Returns:
            Refined saliency map
        """
        # Ensure values are in [0, 1]
        saliency = np.clip(saliency, 0, 1)
        
        # Apply edge-aware smoothing if frequency information is available
        if has_freq_info:
            # Use edge-preserving filter (bilateral filter)
            saliency_8bit = (saliency * 255).astype(np.uint8)
            saliency_smooth = cv2.bilateralFilter(saliency_8bit, 9, 75, 75)
            saliency = saliency_smooth.astype(np.float32) / 255.0
        else:
            # Standard Gaussian blurring
            saliency_8bit = (saliency * 255).astype(np.uint8)
            saliency = cv2.GaussianBlur(saliency_8bit, (self.blur_kernel, self.blur_kernel), 0)
            saliency = saliency.astype(np.float32) / 255.0
        
        # Enhance contrast using CLAHE
        saliency_8bit = (saliency * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        saliency_enhanced = clahe.apply(saliency_8bit).astype(np.float32) / 255.0
        
        # Blend original and enhanced
        saliency = 0.7 * saliency + 0.3 * saliency_enhanced
        
        return np.clip(saliency, 0, 1)
    
    def _generate_binary_masks(
        self,
        saliency_maps: np.ndarray,
        threshold: float = 0.3
    ) -> np.ndarray:
        """
        Generate binary segmentation masks from saliency maps.
        
        Args:
            saliency_maps: Saliency maps (B, H, W)
            threshold: Threshold for binarization
        
        Returns:
            Binary masks (B, H, W)
        """
        binary_masks = []
        
        for saliency in saliency_maps:
            if self.adaptive_threshold:
                # Adaptive thresholding based on local neighborhood
                saliency_8bit = (saliency * 255).astype(np.uint8)
                # Use Otsu's thresholding for automatic threshold selection
                _, binary = cv2.threshold(saliency_8bit, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                # Simple global thresholding
                binary = (saliency > threshold).astype(np.float32)
            
            # Morphological operations to clean up
            # Remove small noise (opening)
            binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, self.erode_kernel)
            # Fill small holes (closing)
            binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, self.dilate_kernel)
            
            binary_masks.append(binary.astype(np.float32))
        
        return np.stack(binary_masks, axis=0)
    
    def _compute_confidence(self, saliency_maps: np.ndarray) -> np.ndarray:
        """
        Compute confidence scores for the saliency maps.
        
        Higher confidence indicates more certain segmentation boundaries.
        
        Args:
            saliency_maps: Saliency maps (B, H, W)
        
        Returns:
            Confidence maps (B, H, W)
        """
        confidence_maps = []
        
        for saliency in saliency_maps:
            # Confidence based on sharpness (high gradient = high confidence)
            # Compute Laplacian for sharpness
            laplacian = cv2.Laplacian((saliency * 255).astype(np.uint8), cv2.CV_32F)
            sharpness = np.abs(laplacian) / (np.abs(laplacian).max() + 1e-5)
            
            # Confidence also based on proximity to decision boundary
            boundary_confidence = 2 * np.abs(saliency - 0.5)
            
            # Combine
            confidence = 0.5 * sharpness + 0.5 * boundary_confidence
            confidence = np.clip(confidence, 0, 1)
            
            confidence_maps.append(confidence)
        
        return np.stack(confidence_maps, axis=0)


class MultiScaleSaliencyGenerator(nn.Module):
    """
    Generates saliency maps at multiple scales and aggregates them for robustness.
    
    This helps capture features at different levels of detail, from fine boundaries
    to coarse structures.
    """
    
    def __init__(self, scales: List[float] = [0.5, 1.0, 1.5], aggregation: str = 'mean'):
        """
        Args:
            scales: List of scale factors (e.g., [0.5, 1.0, 1.5])
            aggregation: How to aggregate ('mean', 'max', 'weighted_mean')
        """
        super().__init__()
        self.scales = scales
        self.aggregation = aggregation
        self.saliency_generator = FrequencyAwareSaliencyGenerator()
    
    def forward(
        self,
        image_features: torch.Tensor,
        high_freq_features: torch.Tensor,
        text_embedding: torch.Tensor,
        image_tensor: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Dict:
        """
        Generate multi-scale saliency maps and aggregate.
        
        Args:
            (same as FrequencyAwareSaliencyGenerator)
        
        Returns:
            Dict with aggregated saliency maps
        """
        scale_maps = []
        confidence_scores = []
        
        # Generate saliency at each scale
        for scale in self.scales:
            # Resize features if needed
            if scale != 1.0:
                scaled_h = int(image_tensor.shape[2] * scale)
                scaled_w = int(image_tensor.shape[3] * scale)
                scaled_image = F.interpolate(
                    image_tensor,
                    size=(scaled_h, scaled_w),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                scaled_image = image_tensor
            
            # Generate saliency
            result = self.saliency_generator(
                image_features, high_freq_features, text_embedding, scaled_image, target_size
            )
            
            scale_maps.append(result['saliency_map_refined'])
            confidence_scores.append(result['confidence_map'])
        
        # Aggregate
        if self.aggregation == 'mean':
            aggregated_saliency = torch.stack(scale_maps, dim=0).mean(dim=0)
            aggregated_confidence = torch.stack(confidence_scores, dim=0).mean(dim=0)
        elif self.aggregation == 'max':
            aggregated_saliency = torch.stack(scale_maps, dim=0).max(dim=0)[0]
            aggregated_confidence = torch.stack(confidence_scores, dim=0).max(dim=0)[0]
        elif self.aggregation == 'weighted_mean':
            # Weight by confidence
            weights = torch.stack(confidence_scores, dim=0)
            weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-5)
            aggregated_saliency = (torch.stack(scale_maps, dim=0) * weights).sum(dim=0)
            aggregated_confidence = weights.mean(dim=0)
        
        return {
            'saliency_map_refined': aggregated_saliency,
            'confidence_map': aggregated_confidence,
            'scale_maps': scale_maps
        }


# Utility function for saliency map post-processing
def post_process_saliency_map(
    saliency_map: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    denoise: bool = True,
    enhance_edges: bool = True
) -> np.ndarray:
    """
    Post-process saliency map for better visualization and accuracy.
    
    Args:
        saliency_map: Input saliency map
        target_size: Resize to this size if provided
        denoise: Whether to apply denoising
        enhance_edges: Whether to enhance edges
    
    Returns:
        Processed saliency map
    """
    # Resize if needed
    if target_size is not None:
        saliency_map = cv2.resize(saliency_map, target_size, interpolation=cv2.INTER_CUBIC)
    
    # Denoise
    if denoise:
        saliency_8bit = (saliency_map * 255).astype(np.uint8)
        saliency_8bit = cv2.fastNlMeansDenoising(saliency_8bit, h=10)
        saliency_map = saliency_8bit.astype(np.float32) / 255.0
    
    # Enhance edges
    if enhance_edges:
        saliency_8bit = (saliency_map * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(saliency_8bit, cv2.MORPH_GRADIENT, kernel)
        saliency_8bit = cv2.addWeighted(saliency_8bit, 0.9, edges, 0.1, 0)
        saliency_map = saliency_8bit.astype(np.float32) / 255.0
    
    return np.clip(saliency_map, 0, 1)
