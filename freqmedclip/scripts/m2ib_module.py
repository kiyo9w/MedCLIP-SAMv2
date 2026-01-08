"""
Feed-Forward M2IB (Multimodal Information Bottleneck) Module.

Provides a differentiable approximation of M2IB for end-to-end training.
The original M2IB in iba.py uses iterative optimization per-sample at inference,
which is not suitable for training. This module replaces it with a learned
cross-modal attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class M2IBModule(nn.Module):
    """
    Feed-forward M2IB for training.
    
    Takes image embedding and text embedding, produces a coarse semantic map
    that highlights regions relevant to the text description.
    
    Architecture:
    1. Cross-attention between image and text embeddings
    2. Spatial projection to create attention map
    3. Learned bottleneck that compresses information
    """
    
    def __init__(self, embed_dim=512, hidden_dim=256, spatial_size=32):
        """
        Args:
            embed_dim: Dimension of input embeddings (512 for BiomedCLIP pooled)
            hidden_dim: Hidden dimension for processing
            spatial_size: Output spatial size (H, W) of the coarse map
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.spatial_size = spatial_size
        
        # Project embeddings to common space
        self.image_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Cross-attention: image queries text
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Spatial generator: creates 2D attention map from fused features
        # Output: [B, 1, spatial_size, spatial_size]
        self.spatial_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, spatial_size * spatial_size),
        )
        
        # Information bottleneck: learned compression
        self.bottleneck = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, image_embed, text_embed, return_intermediate=False):
        """
        Args:
            image_embed: [B, embed_dim] - Pooled image features from BiomedCLIP
            text_embed: [B, embed_dim] - Pooled text features from BiomedCLIP
            return_intermediate: If True, also return pre-bottleneck attention
        
        Returns:
            coarse_map: [B, 1, spatial_size, spatial_size] - Semantic attention map
        """
        B = image_embed.shape[0]
        
        # Project to common space
        img_feat = self.image_proj(image_embed)  # [B, hidden_dim]
        txt_feat = self.text_proj(text_embed)    # [B, hidden_dim]
        
        # Reshape for attention: [B, 1, hidden_dim]
        img_feat = img_feat.unsqueeze(1)
        txt_feat = txt_feat.unsqueeze(1)
        
        # Cross-attention: image attends to text
        fused_feat, attn_weights = self.cross_attn(
            query=img_feat,
            key=txt_feat,
            value=txt_feat
        )  # [B, 1, hidden_dim]
        
        # Combine with original image features (residual)
        fused_feat = fused_feat + img_feat
        
        # Generate spatial attention map
        fused_feat = fused_feat.squeeze(1)  # [B, hidden_dim]
        spatial_flat = self.spatial_generator(fused_feat)  # [B, H*W]
        
        # Reshape to 2D
        attention_map = spatial_flat.view(B, 1, self.spatial_size, self.spatial_size)
        
        # Apply sigmoid before bottleneck for stability
        attention_map = torch.sigmoid(attention_map)
        
        # Information bottleneck: compress and refine
        coarse_map = self.bottleneck(attention_map)
        
        if return_intermediate:
            return coarse_map, attention_map
        return coarse_map


class M2IBModuleWithSpatialFeatures(nn.Module):
    """
    Enhanced M2IB that also takes spatial features from ViT.
    
    This version uses the full spatial features (not just pooled) for
    more detailed attention map generation.
    """
    
    def __init__(self, embed_dim=512, feat_dim=768, hidden_dim=256, spatial_size=32):
        """
        Args:
            embed_dim: Dimension of pooled embeddings (512)
            feat_dim: Dimension of spatial features (768 for ViT)
            hidden_dim: Hidden dimension
            spatial_size: Output spatial size
        """
        super().__init__()
        
        self.spatial_size = spatial_size
        
        # Project pooled text embedding
        self.text_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Project spatial image features: [B, 768, H, W] -> [B, hidden_dim, H, W]
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(feat_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Text-to-spatial attention
        # Text as query, spatial features as key/value
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Generate final attention map
        self.attention_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, spatial_features, text_embed):
        """
        Args:
            spatial_features: [B, 768, H, W] - Spatial features from ViT
            text_embed: [B, 512] - Pooled text features
        
        Returns:
            coarse_map: [B, 1, H, W] - Semantic attention map
        """
        B, C, H, W = spatial_features.shape
        
        # Project spatial features
        spatial_proj = self.spatial_proj(spatial_features)  # [B, hidden, H, W]
        
        # Flatten spatial for attention: [B, H*W, hidden]
        spatial_flat = spatial_proj.flatten(2).permute(0, 2, 1)
        
        # Project and expand text for attention: [B, 1, hidden]
        text_feat = self.text_proj(text_embed).unsqueeze(1)
        
        # Cross-attention: text queries spatial features
        # Output: [B, 1, hidden] - text feature enhanced with spatial context
        attn_out, attn_weights = self.cross_attn(
            query=text_feat,
            key=spatial_flat,
            value=spatial_flat
        )
        
        # Compute similarity between text and each spatial location
        # [B, 1, hidden] x [B, hidden, H*W] -> [B, 1, H*W]
        similarity = torch.bmm(attn_out, spatial_flat.permute(0, 2, 1))
        similarity = similarity.view(B, 1, H, W)
        
        # Normalize and pass through attention head
        similarity = F.softmax(similarity.view(B, 1, -1), dim=-1).view(B, 1, H, W)
        
        # Combine similarity with projected features for final map
        combined = spatial_proj * similarity  # [B, hidden, H, W]
        coarse_map = self.attention_head(combined)  # [B, 1, H, W]
        
        return coarse_map
