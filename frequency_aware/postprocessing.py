"""
Post-Processing & SAM Integration Module

This module handles the final stage of the frequency-aware pipeline:
1. ROI (Region of Interest) extraction from refined saliency maps
2. Prompt generation (bounding boxes and points) for SAM
3. SAM inference and mask refinement
4. Evaluation metrics and uncertainty quantification
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from PIL import Image


@dataclass
class SAMPrompt:
    """Represents a prompt for SAM (Segment Anything Model)."""
    bboxes: Optional[torch.Tensor] = None  # (N, 4) in (x1, y1, x2, y2) format
    points: Optional[torch.Tensor] = None  # (N, 2) in (x, y) format
    labels: Optional[torch.Tensor] = None  # (N,) with values 1 (foreground) or 0 (background)
    masks: Optional[torch.Tensor] = None   # (N, H, W) with confidence scores


class ROIExtractor(nn.Module):
    """
    Extracts Regions of Interest (ROI) from saliency maps.
    
    Produces bounding boxes and keypoints that can be used as prompts for SAM.
    """
    
    def __init__(
        self,
        min_roi_size: int = 20,
        max_roi_count: int = 5,
        roi_padding: float = 0.1
    ):
        """
        Args:
            min_roi_size: Minimum size for a valid ROI
            max_roi_count: Maximum number of ROIs to extract
            roi_padding: Padding ratio around ROI (0.1 = 10%)
        """
        super().__init__()
        self.min_roi_size = min_roi_size
        self.max_roi_count = max_roi_count
        self.roi_padding = roi_padding
    
    def forward(
        self,
        saliency_map: np.ndarray,
        binary_mask: np.ndarray,
        confidence_map: Optional[np.ndarray] = None,
        threshold: float = 0.3
    ) -> SAMPrompt:
        """
        Extract ROIs from saliency and binary masks.
        
        Args:
            saliency_map: Saliency map [0, 1] (H, W)
            binary_mask: Binary segmentation mask (H, W)
            confidence_map: Confidence scores (H, W)
            threshold: Threshold for ROI selection
        
        Returns:
            SAMPrompt with bounding boxes and points
        """
        h, w = saliency_map.shape[:2]
        
        # Find connected components in binary mask
        saliency_8bit = (saliency_map * 255).astype(np.uint8)
        binary_uint8 = binary_mask.astype(np.uint8)
        
        contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        points = []
        roi_scores = []
        
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            
            # Filter by size
            if bw < self.min_roi_size or bh < self.min_roi_size:
                continue
            
            # Add padding
            pad_x = int(bw * self.roi_padding)
            pad_y = int(bh * self.roi_padding)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + bw + pad_x)
            y2 = min(h, y + bh + pad_y)
            
            # Calculate ROI confidence score
            roi_saliency = saliency_map[y1:y2, x1:x2]
            roi_confidence = confidence_map[y1:y2, x1:x2] if confidence_map is not None else np.ones_like(roi_saliency)
            roi_score = np.mean(roi_saliency * roi_confidence)
            
            bboxes.append([x1, y1, x2, y2])
            roi_scores.append(roi_score)
            
            # Extract centroid as point
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                points.append([cx, cy])
        
        # Sort by confidence and keep top-k
        if len(bboxes) > 0:
            indices = np.argsort(roi_scores)[::-1][:self.max_roi_count]
            bboxes = [bboxes[i] for i in indices]
            points = [points[i] for i in indices] if points else []
        
        # Convert to tensors
        bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32) if bboxes else None
        points_tensor = torch.tensor(points, dtype=torch.float32) if points else None
        labels_tensor = torch.ones(len(points), dtype=torch.int32) if points else None
        
        return SAMPrompt(
            bboxes=bboxes_tensor,
            points=points_tensor,
            labels=labels_tensor,
            masks=None
        )


class SAMPromptGenerator(nn.Module):
    """
    Generates prompts for SAM from refined saliency maps and ROI information.
    
    Supports multiple prompt types:
    1. Bounding box: Efficient for large objects
    2. Points: More flexible, can target specific features
    3. Masks: Pre-computed masks as prompts (if SAM supports it)
    """
    
    def __init__(
        self,
        prompt_type: str = 'bbox',  # 'bbox', 'points', 'combined'
        point_count: int = 5,
        use_boundary_points: bool = True
    ):
        """
        Args:
            prompt_type: Type of prompt to generate
            point_count: Number of points for point-based prompts
            use_boundary_points: Include boundary points in point prompts
        """
        super().__init__()
        self.prompt_type = prompt_type
        self.point_count = point_count
        self.use_boundary_points = use_boundary_points
        self.roi_extractor = ROIExtractor()
    
    def forward(
        self,
        saliency_map: np.ndarray,
        binary_mask: np.ndarray,
        confidence_map: Optional[np.ndarray] = None
    ) -> SAMPrompt:
        """
        Generate SAM prompts.
        
        Args:
            saliency_map: Saliency map [0, 1]
            binary_mask: Binary segmentation mask
            confidence_map: Confidence scores
        
        Returns:
            SAMPrompt with prompts for SAM
        """
        # Extract ROIs
        roi_prompt = self.roi_extractor(saliency_map, binary_mask, confidence_map)
        
        if self.prompt_type == 'bbox':
            return roi_prompt
        
        elif self.prompt_type == 'points':
            return self._generate_point_prompts(saliency_map, binary_mask, confidence_map)
        
        elif self.prompt_type == 'combined':
            return self._combine_prompts(roi_prompt, saliency_map, binary_mask, confidence_map)
        
        else:
            raise ValueError(f"Unknown prompt type: {self.prompt_type}")
    
    def _generate_point_prompts(
        self,
        saliency_map: np.ndarray,
        binary_mask: np.ndarray,
        confidence_map: Optional[np.ndarray] = None
    ) -> SAMPrompt:
        """Generate point-based prompts."""
        h, w = saliency_map.shape[:2]
        points = []
        labels = []
        
        # Find high-confidence regions
        high_conf_mask = saliency_map > 0.5
        coords = np.argwhere(high_conf_mask)
        
        if len(coords) > 0:
            # Sample points from high-confidence regions
            if len(coords) > self.point_count:
                indices = np.random.choice(len(coords), self.point_count, replace=False)
                sampled_coords = coords[indices]
            else:
                sampled_coords = coords
            
            for y, x in sampled_coords:
                points.append([x, y])
                labels.append(1)  # Foreground
        
        # Add boundary points if requested
        if self.use_boundary_points:
            contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                # Sample boundary points
                for point in contour[::max(1, len(contour) // 3)]:
                    x, y = point[0]
                    points.append([x, y])
                    labels.append(1)
        
        points_tensor = torch.tensor(points, dtype=torch.float32) if points else None
        labels_tensor = torch.tensor(labels, dtype=torch.int32) if labels else None
        
        return SAMPrompt(points=points_tensor, labels=labels_tensor)
    
    def _combine_prompts(
        self,
        roi_prompt: SAMPrompt,
        saliency_map: np.ndarray,
        binary_mask: np.ndarray,
        confidence_map: Optional[np.ndarray] = None
    ) -> SAMPrompt:
        """Combine bounding box and point prompts."""
        point_prompt = self._generate_point_prompts(saliency_map, binary_mask, confidence_map)
        
        return SAMPrompt(
            bboxes=roi_prompt.bboxes,
            points=point_prompt.points,
            labels=point_prompt.labels
        )


class MaskRefinement(nn.Module):
    """
    Refines SAM output masks using frequency-aware information.
    
    Improves mask accuracy by:
    1. Boundary refinement using high-frequency components
    2. Hole filling and small object removal
    3. Confidence-based mask blending
    """
    
    def __init__(
        self,
        use_frequency_refinement: bool = True,
        morph_kernel_size: int = 5,
        confidence_threshold: float = 0.5
    ):
        """
        Args:
            use_frequency_refinement: Whether to use frequency info for refinement
            morph_kernel_size: Kernel size for morphological operations
            confidence_threshold: Threshold for confidence-based filtering
        """
        super().__init__()
        self.use_frequency_refinement = use_frequency_refinement
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        self.confidence_threshold = confidence_threshold
    
    def forward(
        self,
        sam_mask: np.ndarray,
        saliency_map: Optional[np.ndarray] = None,
        confidence_map: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Refine SAM masks.
        
        Args:
            sam_mask: Mask from SAM (H, W) or (H, W, 1)
            saliency_map: Original saliency map for reference
            confidence_map: Confidence scores for refinement
        
        Returns:
            Dict with refined mask and metrics
        """
        # Ensure binary
        if len(sam_mask.shape) == 3:
            sam_mask = sam_mask[:, :, 0]
        
        mask = (sam_mask > 0.5).astype(np.uint8)
        h, w = mask.shape[:2]
        
        # Morphological operations
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)
        
        # Boundary refinement with frequency information
        if self.use_frequency_refinement and saliency_map is not None:
            mask = self._refine_boundaries(mask, saliency_map)
        
        # Confidence-based filtering
        if confidence_map is not None:
            mask = self._apply_confidence_filter(mask, confidence_map)
        
        # Remove small objects
        mask = self._remove_small_objects(mask, min_size=50)
        
        # Calculate metrics
        metrics = self._calculate_metrics(mask, saliency_map, confidence_map)
        
        return {
            'refined_mask': mask.astype(np.float32),
            'metrics': metrics
        }
    
    def _refine_boundaries(self, mask: np.ndarray, saliency_map: np.ndarray) -> np.ndarray:
        """Refine mask boundaries using saliency information."""
        # Find boundary pixels
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        refined_mask = mask.copy()
        
        for contour in contours:
            # Get pixels near the boundary
            for point in contour[::max(1, len(contour) // 20)]:
                x, y = point[0]
                
                # Check saliency around boundary
                neighborhood = saliency_map[
                    max(0, y-2):min(saliency_map.shape[0], y+3),
                    max(0, x-2):min(saliency_map.shape[1], x+3)
                ]
                
                if neighborhood.mean() < 0.3:
                    # Low saliency, likely background
                    refined_mask = cv2.circle(refined_mask, (x, y), 2, 0, -1)
        
        return refined_mask
    
    def _apply_confidence_filter(self, mask: np.ndarray, confidence_map: np.ndarray) -> np.ndarray:
        """Filter mask based on confidence scores."""
        # Only keep mask pixels with high confidence
        confidence_mask = (confidence_map > self.confidence_threshold).astype(np.uint8)
        filtered_mask = mask & confidence_mask
        
        return filtered_mask
    
    def _remove_small_objects(self, mask: np.ndarray, min_size: int = 50) -> np.ndarray:
        """Remove small objects from mask."""
        # Label connected components
        num_labels, labels = cv2.connectedComponents(mask)
        
        # Count pixels in each component
        for label_id in range(1, num_labels):
            component_size = np.sum(labels == label_id)
            if component_size < min_size:
                mask[labels == label_id] = 0
        
        return mask
    
    def _calculate_metrics(
        self,
        mask: np.ndarray,
        saliency_map: Optional[np.ndarray] = None,
        confidence_map: Optional[np.ndarray] = None
    ) -> Dict:
        """Calculate metrics for the refined mask."""
        metrics = {
            'mask_area': np.sum(mask),
            'mask_coverage': np.sum(mask) / mask.size
        }
        
        if saliency_map is not None:
            mask_saliency = saliency_map[mask > 0]
            if len(mask_saliency) > 0:
                metrics['mean_saliency'] = np.mean(mask_saliency)
                metrics['saliency_variance'] = np.var(mask_saliency)
        
        if confidence_map is not None:
            mask_confidence = confidence_map[mask > 0]
            if len(mask_confidence) > 0:
                metrics['mean_confidence'] = np.mean(mask_confidence)
        
        return metrics


class FrequencyAwarePipeline(nn.Module):
    """
    Complete frequency-aware segmentation pipeline:
    Saliency Map Generation -> ROI Extraction -> SAM Prompting -> Mask Refinement
    """
    
    def __init__(
        self,
        prompt_type: str = 'bbox',
        refine_masks: bool = True
    ):
        """
        Args:
            prompt_type: Type of SAM prompts ('bbox', 'points', 'combined')
            refine_masks: Whether to refine SAM masks
        """
        super().__init__()
        self.prompt_generator = SAMPromptGenerator(prompt_type=prompt_type)
        self.mask_refiner = MaskRefinement() if refine_masks else None
    
    def forward(
        self,
        saliency_map: np.ndarray,
        binary_mask: np.ndarray,
        confidence_map: Optional[np.ndarray] = None,
        sam_mask: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Run full post-processing pipeline.
        
        Args:
            saliency_map: From saliency generation stage
            binary_mask: From saliency generation stage
            confidence_map: From saliency generation stage
            sam_mask: Output from SAM (if already run)
        
        Returns:
            Dict with prompts, refined masks, and metrics
        """
        # Generate SAM prompts
        prompts = self.prompt_generator(saliency_map, binary_mask, confidence_map)
        
        # Refine masks if provided
        refined_results = None
        if sam_mask is not None and self.mask_refiner is not None:
            refined_results = self.mask_refiner(sam_mask, saliency_map, confidence_map)
        
        return {
            'prompts': prompts,
            'refined_mask': refined_results['refined_mask'] if refined_results else None,
            'metrics': refined_results['metrics'] if refined_results else {}
        }


# Utility functions
def draw_prompts(
    image: np.ndarray,
    prompts: SAMPrompt,
    color_bbox: Tuple[int, int, int] = (0, 255, 0),
    color_point: Tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:
    """
    Draw prompts on image for visualization.
    
    Args:
        image: Input image
        prompts: SAMPrompt instance
        color_bbox: Color for bounding boxes (BGR)
        color_point: Color for points (BGR)
    
    Returns:
        Image with drawn prompts
    """
    result = image.copy()
    
    if prompts.bboxes is not None:
        for bbox in prompts.bboxes:
            x1, y1, x2, y2 = bbox.int().numpy()
            cv2.rectangle(result, (x1, y1), (x2, y2), color_bbox, 2)
    
    if prompts.points is not None:
        for point in prompts.points:
            x, y = point.int().numpy()
            cv2.circle(result, (x, y), 5, color_point, -1)
    
    return result


def visualize_segmentation_results(
    image: np.ndarray,
    binary_mask: np.ndarray,
    saliency_map: Optional[np.ndarray] = None,
    confidence_map: Optional[np.ndarray] = None,
    alpha: float = 0.5
) -> List[np.ndarray]:
    """
    Create visualizations of segmentation results.
    
    Args:
        image: Original image
        binary_mask: Binary segmentation mask
        saliency_map: Optional saliency map
        confidence_map: Optional confidence map
        alpha: Transparency for overlay
    
    Returns:
        List of visualization images
    """
    visualizations = []
    
    # Mask overlay
    mask_overlay = image.copy()
    mask_overlay[binary_mask > 0] = [0, 255, 0]
    blended = cv2.addWeighted(image, 1 - alpha, mask_overlay, alpha, 0)
    visualizations.append(blended)
    
    # Saliency map if provided
    if saliency_map is not None:
        saliency_vis = cv2.applyColorMap((saliency_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        visualizations.append(saliency_vis)
    
    # Confidence map if provided
    if confidence_map is not None:
        confidence_vis = cv2.applyColorMap((confidence_map * 255).astype(np.uint8), cv2.COLORMAP_COOL)
        visualizations.append(confidence_vis)
    
    return visualizations
