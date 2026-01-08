"""
SAM (Segment Anything Model) Wrapper for integration with M2IB Saliency pipeline.

Provides a unified interface for SAM, supporting both:
1. HuggingFace transformers SamModel
2. Original segment-anything library

SAM is kept frozen during training - only the saliency generation is trained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple, List


class SAMWrapper(nn.Module):
    """
    Wrapper for SAM model that handles prompt-based mask generation.
    
    Supports both HuggingFace and original segment-anything implementations.
    The model is always frozen during training.
    """
    
    def __init__(
        self, 
        model_type: str = "vit_h",
        checkpoint_path: Optional[str] = None,
        use_hf: bool = True,
        device: str = "cuda"
    ):
        """
        Args:
            model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
            checkpoint_path: Path to SAM checkpoint (for original segment-anything)
            use_hf: Whether to use HuggingFace transformers implementation
            device: Device to load model on
        """
        super().__init__()
        
        self.device = device
        self.use_hf = use_hf
        self.sam_model = None
        self.sam_processor = None
        
        if use_hf:
            self._load_hf_sam()
        else:
            self._load_original_sam(model_type, checkpoint_path)
            
        # Freeze all SAM parameters
        self._freeze()
        
    def _load_hf_sam(self):
        """Load SAM from HuggingFace transformers."""
        try:
            from transformers import SamModel, SamProcessor
            
            # Use the base SAM model from HuggingFace
            model_id = "facebook/sam-vit-huge"
            
            print(f"Loading SAM from HuggingFace: {model_id}")
            self.sam_model = SamModel.from_pretrained(model_id).to(self.device)
            self.sam_processor = SamProcessor.from_pretrained(model_id)
            
            print("SAM loaded successfully from HuggingFace")
            
        except ImportError:
            print("HuggingFace transformers SAM not available, falling back to original")
            self.use_hf = False
            self._load_original_sam("vit_h", None)
            
    def _load_original_sam(self, model_type: str, checkpoint_path: Optional[str]):
        """Load SAM from original segment-anything library."""
        try:
            from segment_anything import sam_model_registry, SamPredictor
            
            if checkpoint_path is None:
                # Try default paths
                import os
                possible_paths = [
                    "segment-anything/sam_checkpoints/sam_vit_h_4b8939.pth",
                    "../segment-anything/sam_checkpoints/sam_vit_h_4b8939.pth",
                    "sam_vit_h_4b8939.pth",
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        checkpoint_path = path
                        break
                        
            if checkpoint_path is None:
                raise FileNotFoundError("SAM checkpoint not found")
                
            print(f"Loading SAM from checkpoint: {checkpoint_path}")
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            self.sam_model = sam.to(self.device)
            self.sam_predictor = SamPredictor(self.sam_model)
            
            print("SAM loaded successfully from original library")
            
        except ImportError:
            raise ImportError(
                "Neither HuggingFace transformers nor segment-anything library available. "
                "Install with: pip install transformers or pip install segment-anything"
            )
            
    def _freeze(self):
        """Freeze all SAM parameters."""
        if self.sam_model is not None:
            for param in self.sam_model.parameters():
                param.requires_grad = False
            self.sam_model.eval()
            
    def forward(
        self, 
        images: torch.Tensor,
        boxes: Optional[torch.Tensor] = None,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        multimask_output: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Generate masks using SAM with provided prompts.
        
        Args:
            images: [B, 3, H, W] - Input images (normalized)
            boxes: [B, 4] - Bounding boxes (x1, y1, x2, y2)
            point_coords: [B, N, 2] - Point coordinates (x, y)
            point_labels: [B, N] - Point labels (1=foreground, 0=background)
            multimask_output: Whether to return multiple masks per prompt
        
        Returns:
            Dict with:
                - masks: [B, 1, H, W] - Predicted masks (or [B, 3, H, W] if multimask)
                - iou_scores: [B, 1] - IoU predictions (or [B, 3] if multimask)
        """
        if self.use_hf:
            return self._forward_hf(images, boxes, point_coords, point_labels, multimask_output)
        else:
            return self._forward_original(images, boxes, point_coords, point_labels, multimask_output)
            
    def _forward_hf(
        self, 
        images: torch.Tensor,
        boxes: Optional[torch.Tensor],
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        multimask_output: bool
    ) -> Dict[str, torch.Tensor]:
        """Forward pass using HuggingFace SAM."""
        B, C, H, W = images.shape
        
        # Prepare inputs for HuggingFace SAM
        # HF SAM expects raw images, we need to denormalize if necessary
        
        # Process each image in batch (HF SAM processes one at a time)
        all_masks = []
        all_scores = []
        
        with torch.no_grad():
            for i in range(B):
                img = images[i:i+1]  # [1, 3, H, W]
                
                # Prepare prompts
                input_boxes = None
                input_points = None
                input_labels = None
                
                if boxes is not None:
                    # Reshape boxes for HF: [1, 1, 4]
                    input_boxes = boxes[i:i+1].unsqueeze(1)
                    
                if point_coords is not None and point_labels is not None:
                    # Reshape points for HF: [1, N, 2]
                    input_points = point_coords[i:i+1]
                    input_labels = point_labels[i:i+1]
                
                # Forward through SAM
                outputs = self.sam_model(
                    pixel_values=img,
                    input_boxes=input_boxes,
                    input_points=input_points,
                    input_labels=input_labels,
                    multimask_output=multimask_output
                )
                
                # Get masks and scores
                pred_masks = outputs.pred_masks  # [1, num_masks, H', W']
                iou_scores = outputs.iou_scores  # [1, num_masks]
                
                # Resize masks to input size
                pred_masks = F.interpolate(
                    pred_masks.float(),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )
                
                all_masks.append(pred_masks)
                all_scores.append(iou_scores)
                
        # Stack batch
        masks = torch.cat(all_masks, dim=0)  # [B, num_masks, H, W]
        scores = torch.cat(all_scores, dim=0)  # [B, num_masks]
        
        # Take best mask if multimask
        if multimask_output:
            best_idx = scores.argmax(dim=1)  # [B]
            masks = torch.stack([masks[i, best_idx[i]] for i in range(B)]).unsqueeze(1)
            scores = torch.stack([scores[i, best_idx[i]] for i in range(B)]).unsqueeze(1)
        else:
            masks = masks[:, 0:1]  # [B, 1, H, W]
            scores = scores[:, 0:1]  # [B, 1]
            
        return {
            'masks': masks,
            'iou_scores': scores
        }
        
    def _forward_original(
        self, 
        images: torch.Tensor,
        boxes: Optional[torch.Tensor],
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        multimask_output: bool
    ) -> Dict[str, torch.Tensor]:
        """Forward pass using original segment-anything."""
        B, C, H, W = images.shape
        
        all_masks = []
        all_scores = []
        
        with torch.no_grad():
            for i in range(B):
                # Convert to numpy for original SAM
                img_np = images[i].permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                
                # Set image
                self.sam_predictor.set_image(img_np)
                
                # Prepare prompts
                box_np = boxes[i].cpu().numpy() if boxes is not None else None
                point_np = point_coords[i].cpu().numpy() if point_coords is not None else None
                label_np = point_labels[i].cpu().numpy() if point_labels is not None else None
                
                # Predict
                masks_np, scores_np, _ = self.sam_predictor.predict(
                    point_coords=point_np,
                    point_labels=label_np,
                    box=box_np,
                    multimask_output=multimask_output
                )
                
                # Convert back to tensor
                masks_t = torch.from_numpy(masks_np).float().to(self.device)
                scores_t = torch.from_numpy(scores_np).float().to(self.device)
                
                all_masks.append(masks_t)
                all_scores.append(scores_t)
                
        # Stack batch
        masks = torch.stack(all_masks)  # [B, num_masks, H, W]
        scores = torch.stack(all_scores)  # [B, num_masks]
        
        # Take best mask
        if multimask_output:
            best_idx = scores.argmax(dim=1)
            masks = torch.stack([masks[i, best_idx[i]] for i in range(B)]).unsqueeze(1)
            scores = torch.stack([scores[i, best_idx[i]] for i in range(B)]).unsqueeze(1)
        else:
            masks = masks[:, 0:1]
            scores = scores[:, 0:1]
            
        return {
            'masks': masks,
            'iou_scores': scores
        }


class DummySAM(nn.Module):
    """
    Dummy SAM for testing without actual SAM model.
    Simply upsamples the saliency map and applies threshold.
    """
    
    def __init__(self):
        super().__init__()
        # Simple refinement conv
        self.refine = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        images: torch.Tensor,
        saliency_map: torch.Tensor,
        boxes: Optional[torch.Tensor] = None,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: [B, 3, H, W]
            saliency_map: [B, 1, H', W'] - Saliency map to refine
        """
        B, C, H, W = images.shape
        
        # Upsample saliency to image size
        saliency_up = F.interpolate(
            saliency_map, 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Refine
        masks = self.refine(saliency_up)
        
        return {
            'masks': masks,
            'iou_scores': torch.ones(B, 1, device=images.device)
        }
