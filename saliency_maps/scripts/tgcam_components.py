import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GatedITEM(nn.Module):
    """
    Gated Iterative Text Enhancement Module (G-ITEM).
    Prevents semantic drift by gating the visual influence on text.
    """
    def __init__(self, text_dim: int, visual_dim: int, mid_channels: int = 512, num_heads: int = 4):
        super().__init__()
        
        self.v_proj = nn.Linear(visual_dim, mid_channels)
        self.t_proj = nn.Linear(text_dim, mid_channels)
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=mid_channels, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        self.norm_t = nn.LayerNorm(mid_channels)
        self.norm_v = nn.LayerNorm(mid_channels)
        
        self.ffn = nn.Sequential(
            nn.Linear(mid_channels, mid_channels * 4),
            nn.ReLU(),
            nn.Linear(mid_channels * 4, mid_channels)
        )
        
        # CRITICAL: Initialize gate to small value to prioritize original text initially
        self.gate = nn.Parameter(torch.tensor([0.1]))
        self.out_proj = nn.Linear(mid_channels, text_dim)


    def forward(self, text_embed: torch.Tensor, visual_features: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_proj(text_embed)
        v_emb = self.v_proj(visual_features)
        
        t_norm = self.norm_t(t_emb)
        v_norm = self.norm_v(v_emb)
        
        # Text queries Image: "What in the image looks like this text?"
        attn_out, _ = self.multihead_attn(query=t_norm, key=v_norm, value=v_norm)
        
        # Gated Update: t_new = t + gate * attention
        t_refined = t_emb + torch.tanh(self.gate) * attn_out
        
        t_refined = t_refined + self.ffn(self.norm_t(t_refined))
        return self.out_proj(t_refined) + text_embed


class SharpenedTGCAM(nn.Module):
    """
    Sharpened Symmetric Attention.
    Uses temperature scaling to produce sparse saliency maps.
    """
    def __init__(self, visual_dim: int, text_dim: int, mid_channels: int = 512):
        super().__init__()
        self.mid_channels = mid_channels
        self.v_proj = nn.Linear(visual_dim, mid_channels)
        self.t_proj = nn.Linear(text_dim, mid_channels)
        
        # Fusion conv for feature output
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(mid_channels * 2, visual_dim, kernel_size=1),
            nn.GroupNorm(8, visual_dim),
            nn.ReLU(inplace=True)
        )


    def forward(self, visual_features: torch.Tensor, text_features: torch.Tensor, spatial_size: int, temperature: float = 0.07) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C_v = visual_features.shape
        
        v_common = self.v_proj(visual_features) # [B, N, mid]
        t_common = self.t_proj(text_features)   # [B, L, mid]
        
        # Symmetric Affinity [B, N, L]
        affinity = torch.bmm(v_common, t_common.transpose(1, 2))
        affinity = affinity * (self.mid_channels ** -0.5)
        
        # 1. Saliency Map Generation (Max Pooling Strategy)
        # Find patches that strongly activate for *any* text token
        patch_activations = affinity.max(dim=2)[0] 
        saliency_map = patch_activations.view(B, 1, spatial_size, spatial_size)
        
        # Instance Normalization for Saliency (Critical for thresholding)
        saliency_map = (saliency_map - saliency_map.amin(dim=(2,3), keepdim=True)) / \
                       (saliency_map.amax(dim=(2,3), keepdim=True) - saliency_map.amin(dim=(2,3), keepdim=True) + 1e-6)


        # 2. Context Retrieval (Sharpened)
        attn_scores = F.softmax(affinity / temperature, dim=-1)
        text_context = torch.bmm(attn_scores, t_common)
        
        # 3. Fusion
        v_spatial = v_common.permute(0, 2, 1).view(B, -1, spatial_size, spatial_size)
        c_spatial = text_context.permute(0, 2, 1).view(B, -1, spatial_size, spatial_size)
        
        fused_out = self.fusion_conv(torch.cat([v_spatial, c_spatial], dim=1))
        
        # Residual connection: Project original visual features to match fused_out's channels
        # fused_out: [B, visual_dim, H, H]
        # visual_features: [B, N, C_v] -> Need to match visual_dim
        if C_v != fused_out.shape[1]:
            # Add a learnable projection if dimensions mismatch
            if not hasattr(self, 'residual_proj'):
                self.residual_proj = nn.Conv2d(C_v, fused_out.shape[1], kernel_size=1).to(visual_features.device)
            original_v = visual_features.permute(0, 2, 1).view(B, C_v, spatial_size, spatial_size)
            original_v = self.residual_proj(original_v)
        else:
            original_v = visual_features.permute(0, 2, 1).view(B, C_v, spatial_size, spatial_size)
        
        return fused_out + original_v, saliency_map


class TGCAMPipeline(nn.Module):
    """
    Orchestrator class. 
    Renamed from UltimateTGCAM to match existing pipeline naming convention.
    """
    def __init__(self, visual_dim=768, text_dim=768, mid_channels=512, num_item_iterations=2):
        super().__init__()
        self.num_iterations = num_item_iterations
        self.item = GatedITEM(text_dim, visual_dim, mid_channels)
        self.cam = SharpenedTGCAM(visual_dim, text_dim, mid_channels)
        
    def forward(self, visual_features, text_features):
        """
        Args:
            visual_features: [B, N, C] where N = num_patches (NO CLS TOKEN)
            text_features: [B, L, D]
        Returns:
            saliency_map: [B, 1, H, H]
            fused_patches: [B, C, H, H]
        """
        B, N, C = visual_features.shape
        
        # Validate that N is a perfect square
        spatial_size = int(N ** 0.5)
        if spatial_size * spatial_size != N:
            raise ValueError(
                f"Expected perfect square number of patches, got {N}. "
                f"Ensure CLS token is removed before calling TGCAMPipeline. "
                f"Use: visual_features[:, 1:, :] if your model outputs [B, 197, C]."
            )
        
        # Iterative Text Refinement
        refined_text = text_features
        for _ in range(self.num_iterations):
            refined_text = self.item(refined_text, visual_features)
        
        # CAM & Saliency
        fused, saliency = self.cam(visual_features, refined_text, spatial_size)
        
        return saliency, fused
