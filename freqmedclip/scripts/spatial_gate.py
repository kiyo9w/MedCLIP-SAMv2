"""
Spatial Gate Attention Module for M2IB Saliency Architecture.

Combines semantic attention from M2IB with high-frequency boundary features
to produce a refined saliency map that highlights both "where" (semantics)
and "boundaries" (edges/textures).

Based on the architecture diagram:
- M2IB output -> MLP ReLU -> Spatial Logits
- High Frequency -> Conv1x1 -> Boundary Logits  
- ADD -> ReLU -> Conv3x3 -> Sigmoid -> Spatial Gating
- Features x Gating = Gated Spatial Features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGateAttention(nn.Module):
    """
    Spatial Gate Attention that fuses M2IB semantic attention with 
    high-frequency boundary information.
    
    The gate learns to:
    1. Use M2IB to identify "where" the target object is (semantic)
    2. Use high-freq features to identify precise boundaries
    3. Combine both for accurate saliency
    """
    
    def __init__(self, m2ib_channels=1, freq_channels=768, hidden_channels=256):
        """
        Args:
            m2ib_channels: Channels from M2IB output (typically 1)
            freq_channels: Channels from frequency projector (768)
            hidden_channels: Hidden channels for processing
        """
        super().__init__()
        
        # M2IB branch: process semantic attention map
        # m2ib_map [B, 1, H, W] -> spatial_logits [B, hidden, H, W]
        self.m2ib_mlp = nn.Sequential(
            nn.Conv2d(m2ib_channels, hidden_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 4, hidden_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
        )
        
        # High-frequency branch: process boundary features
        # freq_features [B, 768, H, W] -> boundary_logits [B, hidden, H, W]
        self.freq_conv = nn.Sequential(
            nn.Conv2d(freq_channels, hidden_channels // 2, kernel_size=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
        )
        
        # Combined processing after ADD
        # [B, hidden, H, W] -> [B, 1, H, W]
        self.gate_conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, hidden_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 4, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, m2ib_map, freq_features, return_gate=False):
        """
        Args:
            m2ib_map: [B, 1, H, W] - Semantic attention from M2IB
            freq_features: [B, 768, H, W] - High-frequency features from DWT projector
            return_gate: If True, also return the raw gate
        
        Returns:
            saliency_map: [B, 1, H, W] - Final saliency map
            (optional) gate: [B, 1, H, W] - Raw gating weights
        """
        # Ensure spatial dimensions match
        if m2ib_map.shape[-2:] != freq_features.shape[-2:]:
            m2ib_map = F.interpolate(
                m2ib_map, 
                size=freq_features.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Process M2IB semantic map
        spatial_logits = self.m2ib_mlp(m2ib_map)  # [B, hidden/2, H, W]
        
        # Process high-frequency features
        boundary_logits = self.freq_conv(freq_features)  # [B, hidden/2, H, W]
        
        # ADD and generate gate
        combined = spatial_logits + boundary_logits  # [B, hidden/2, H, W]
        gate = self.gate_conv(combined)  # [B, 1, H, W]
        
        # Apply gate to create saliency map
        # For saliency, we want the gate itself as the output
        saliency_map = gate
        
        if return_gate:
            return saliency_map, gate
        return saliency_map


class SpatialGateAttentionV2(nn.Module):
    """
    Enhanced version that also gates the frequency features.
    
    Output: gated_features that can be used for further processing
    """
    
    def __init__(self, m2ib_channels=1, freq_channels=768, out_channels=768):
        super().__init__()
        
        # M2IB branch
        self.m2ib_mlp = nn.Sequential(
            nn.Conv2d(m2ib_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Frequency branch for gate computation
        self.freq_gate_conv = nn.Conv2d(freq_channels, 128, kernel_size=1)
        
        # Gate generator
        self.gate_gen = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Feature processor (for gating freq features)
        self.feature_proj = nn.Sequential(
            nn.Conv2d(freq_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        
        # Learnable residual weight
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, m2ib_map, freq_features):
        """
        Args:
            m2ib_map: [B, 1, H, W] - Semantic attention from M2IB
            freq_features: [B, 768, H, W] - High-frequency features
        
        Returns:
            saliency_map: [B, 1, H, W] - Final saliency map
            gated_features: [B, out_channels, H, W] - Gated frequency features
        """
        # Ensure spatial dimensions match
        if m2ib_map.shape[-2:] != freq_features.shape[-2:]:
            m2ib_map = F.interpolate(
                m2ib_map, 
                size=freq_features.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Process branches
        m2ib_feat = self.m2ib_mlp(m2ib_map)  # [B, 128, H, W]
        freq_gate_feat = self.freq_gate_conv(freq_features)  # [B, 128, H, W]
        
        # Generate gate
        combined = m2ib_feat + freq_gate_feat
        gate = self.gate_gen(combined)  # [B, 1, H, W]
        
        # Saliency map is the gate
        saliency_map = gate
        
        # Apply gate to frequency features with residual
        feat_proj = self.feature_proj(freq_features)
        gated_features = feat_proj * gate + feat_proj * self.residual_weight
        
        return saliency_map, gated_features


class SimpleSpatialGate(nn.Module):
    """
    Simplified spatial gate matching the diagram exactly.
    
    M2IB -> MLP ReLU -> Spatial Logits
    HighFreq -> Conv1x1
    ADD -> ReLU -> Conv3x3 -> Sigmoid -> Gate
    """
    
    def __init__(self, freq_channels=768):
        super().__init__()
        
        # M2IB path: [B, 1, H, W] -> [B, 1, H, W]
        self.m2ib_mlp = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )
        
        # High-freq path: [B, 768, H, W] -> [B, 1, H, W]
        self.freq_conv = nn.Conv2d(freq_channels, 1, kernel_size=1)
        
        # Gate generation
        self.gate_conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, m2ib_map, freq_features):
        """
        Args:
            m2ib_map: [B, 1, H, W]
            freq_features: [B, 768, H, W]
        
        Returns:
            saliency_map: [B, 1, H, W]
        """
        # Ensure dimensions match
        if m2ib_map.shape[-2:] != freq_features.shape[-2:]:
            m2ib_map = F.interpolate(
                m2ib_map, 
                size=freq_features.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Process paths
        spatial_logits = self.m2ib_mlp(m2ib_map)  # [B, 1, H, W]
        boundary_logits = self.freq_conv(freq_features)  # [B, 1, H, W]
        
        # Combine and generate gate
        combined = spatial_logits + boundary_logits  # [B, 1, H, W]
        gate = self.gate_conv(combined)  # [B, 1, H, W]
        
        return gate
