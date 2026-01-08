"""
Prompt Extractor for converting saliency maps to SAM-compatible prompts.

Converts continuous saliency maps to discrete prompts:
- Bounding boxes from thresholded regions
- Point prompts (positive: inside object, negative: background)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PromptExtractor(nn.Module):
    """
    Extracts SAM-compatible prompts from saliency maps.
    
    Non-parametric module - no learnable weights.
    Operates on batched inputs during both training and inference.
    """
    
    def __init__(self, threshold=0.5, min_area=100, use_points=True, num_points=3):
        """
        Args:
            threshold: Threshold for binarizing saliency map
            min_area: Minimum area to consider a valid region
            use_points: Whether to extract point prompts
            num_points: Number of point prompts to extract per image
        """
        super().__init__()
        self.threshold = threshold
        self.min_area = min_area
        self.use_points = use_points
        self.num_points = num_points
        
    @torch.no_grad()
    def forward(self, saliency_map, image_size=None):
        """
        Extract prompts from saliency map.
        
        Args:
            saliency_map: [B, 1, H, W] - Saliency map in [0, 1]
            image_size: Target image size (H, W) for SAM. If None, uses saliency_map size.
        
        Returns:
            dict with:
                - boxes: [B, 4] - Bounding boxes (x1, y1, x2, y2) in image coordinates
                - point_coords: [B, num_points, 2] - Point coordinates (x, y)
                - point_labels: [B, num_points] - Point labels (1=foreground, 0=background)
                - valid_mask: [B] - Boolean mask for valid extractions
        """
        B, _, H, W = saliency_map.shape
        device = saliency_map.device
        
        if image_size is None:
            image_size = (H, W)
        target_h, target_w = image_size
        
        # Scale factors
        scale_h = target_h / H
        scale_w = target_w / W
        
        # Initialize outputs
        boxes = torch.zeros(B, 4, device=device)
        point_coords = torch.zeros(B, self.num_points, 2, device=device)
        point_labels = torch.zeros(B, self.num_points, device=device)
        valid_mask = torch.ones(B, dtype=torch.bool, device=device)
        
        # Process each image in batch
        for b in range(B):
            sal = saliency_map[b, 0]  # [H, W]
            
            # Threshold
            binary = (sal > self.threshold).float()
            
            # Find bounding box
            nonzero = torch.nonzero(binary, as_tuple=False)
            
            if len(nonzero) < self.min_area:
                # No valid region found - use center box as fallback
                cx, cy = W // 2, H // 2
                box_size = min(H, W) // 4
                boxes[b] = torch.tensor([
                    (cx - box_size) * scale_w,
                    (cy - box_size) * scale_h,
                    (cx + box_size) * scale_w,
                    (cy + box_size) * scale_h
                ], device=device)
                
                # Center point as foreground
                point_coords[b, 0] = torch.tensor([cx * scale_w, cy * scale_h], device=device)
                point_labels[b, 0] = 1
                
                # Random background points
                for i in range(1, self.num_points):
                    point_coords[b, i] = torch.tensor([
                        torch.randint(0, int(target_w), (1,)).item(),
                        torch.randint(0, int(target_h), (1,)).item()
                    ], device=device)
                    point_labels[b, i] = 0
                    
                valid_mask[b] = False
            else:
                # Valid region found
                y_coords = nonzero[:, 0]
                x_coords = nonzero[:, 1]
                
                y1 = y_coords.min().item()
                y2 = y_coords.max().item()
                x1 = x_coords.min().item()
                x2 = x_coords.max().item()
                
                # Add small margin
                margin = 2
                y1 = max(0, y1 - margin)
                y2 = min(H - 1, y2 + margin)
                x1 = max(0, x1 - margin)
                x2 = min(W - 1, x2 + margin)
                
                # Scale to image coordinates
                boxes[b] = torch.tensor([
                    x1 * scale_w,
                    y1 * scale_h,
                    x2 * scale_w,
                    y2 * scale_h
                ], device=device)
                
                if self.use_points:
                    # Extract point prompts
                    # 1. Centroid as positive point
                    cx = x_coords.float().mean().item()
                    cy = y_coords.float().mean().item()
                    point_coords[b, 0] = torch.tensor([cx * scale_w, cy * scale_h], device=device)
                    point_labels[b, 0] = 1
                    
                    # 2. Additional positive points from high-saliency areas
                    if self.num_points > 1:
                        # Find top saliency points
                        high_sal_mask = sal > (self.threshold + 0.2)
                        high_sal_nonzero = torch.nonzero(high_sal_mask, as_tuple=False)
                        
                        if len(high_sal_nonzero) > 0:
                            # Random sample from high saliency
                            idx = torch.randint(0, len(high_sal_nonzero), (min(self.num_points - 1, len(high_sal_nonzero)),))
                            for i, sample_idx in enumerate(idx):
                                if i + 1 < self.num_points:
                                    py, px = high_sal_nonzero[sample_idx]
                                    point_coords[b, i + 1] = torch.tensor([px.item() * scale_w, py.item() * scale_h], device=device)
                                    point_labels[b, i + 1] = 1
                                    
                    # 3. Background points (outside bounding box)
                    remaining = self.num_points - (point_labels[b] == 1).sum().item()
                    if remaining > 0:
                        for i in range(self.num_points - int(remaining), self.num_points):
                            # Sample from corners (likely background)
                            corner_options = [
                                (0, 0), (0, W-1), (H-1, 0), (H-1, W-1),
                                (0, W//2), (H-1, W//2), (H//2, 0), (H//2, W-1)
                            ]
                            cy, cx = corner_options[i % len(corner_options)]
                            point_coords[b, i] = torch.tensor([cx * scale_w, cy * scale_h], device=device)
                            point_labels[b, i] = 0
        
        return {
            'boxes': boxes,
            'point_coords': point_coords,
            'point_labels': point_labels,
            'valid_mask': valid_mask
        }


class DifferentiablePromptExtractor(nn.Module):
    """
    Differentiable version of prompt extractor for end-to-end training.
    
    Uses soft operations instead of hard thresholding to maintain gradients.
    """
    
    def __init__(self, temperature=0.1):
        """
        Args:
            temperature: Temperature for soft operations
        """
        super().__init__()
        self.temperature = temperature
        
    def forward(self, saliency_map, image_size=None):
        """
        Extract soft prompts from saliency map.
        
        Args:
            saliency_map: [B, 1, H, W] - Saliency map
            image_size: Target size for coordinates
        
        Returns:
            dict with soft prompts
        """
        B, _, H, W = saliency_map.shape
        device = saliency_map.device
        
        if image_size is None:
            target_h, target_w = H, W
        else:
            target_h, target_w = image_size
            
        scale_h = target_h / H
        scale_w = target_w / W
        
        # Create coordinate grids
        y_coords = torch.arange(H, device=device).float()
        x_coords = torch.arange(W, device=device).float()
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Soft weighted coordinates (center of mass)
        sal_flat = saliency_map.view(B, -1)  # [B, H*W]
        sal_weights = F.softmax(sal_flat / self.temperature, dim=-1)  # [B, H*W]
        
        xx_flat = xx.view(-1)  # [H*W]
        yy_flat = yy.view(-1)  # [H*W]
        
        # Weighted center
        cx = (sal_weights * xx_flat.unsqueeze(0)).sum(dim=1)  # [B]
        cy = (sal_weights * yy_flat.unsqueeze(0)).sum(dim=1)  # [B]
        
        # Soft bounding box (weighted extent)
        # Use weighted std as box size estimate
        dx = ((sal_weights * ((xx_flat.unsqueeze(0) - cx.unsqueeze(1)) ** 2)).sum(dim=1)).sqrt()
        dy = ((sal_weights * ((yy_flat.unsqueeze(0) - cy.unsqueeze(1)) ** 2)).sum(dim=1)).sqrt()
        
        # Box coordinates (center +/- 2*std)
        x1 = (cx - 2 * dx).clamp(min=0) * scale_w
        y1 = (cy - 2 * dy).clamp(min=0) * scale_h
        x2 = (cx + 2 * dx).clamp(max=W-1) * scale_w
        y2 = (cy + 2 * dy).clamp(max=H-1) * scale_h
        
        boxes = torch.stack([x1, y1, x2, y2], dim=1)  # [B, 4]
        
        # Center point as prompt
        point_coords = torch.stack([cx * scale_w, cy * scale_h], dim=1).unsqueeze(1)  # [B, 1, 2]
        point_labels = torch.ones(B, 1, device=device)  # [B, 1]
        
        return {
            'boxes': boxes,
            'point_coords': point_coords,
            'point_labels': point_labels,
            'center': torch.stack([cx, cy], dim=1),
            'std': torch.stack([dx, dy], dim=1)
        }
