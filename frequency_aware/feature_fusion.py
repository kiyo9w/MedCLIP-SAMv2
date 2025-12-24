"""
Frequency-Aware Feature Fusion Module
Injects high-frequency wavelet information into BiomedCLIP's Vision Encoder

This module implements the Feature Fusion layer (Giai đoạn 2 from brainstorm.md)
that combines semantic features from BiomedCLIP with boundary-detection features
from wavelet decomposition.

Key components:
1. HighFreqProjection: Projects wavelet components to encoder dimension
2. FeatureFusionGate: Learns how to combine original and frequency-aware features
3. FrequencyAwareVisionEncoder: Wrapped encoder with fusion capability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from functools import wraps


class HighFreqProjection(nn.Module):
    """
    Projects high-frequency wavelet components to match the embedding dimension
    of BiomedCLIP's vision encoder.
    
    Input shape: (batch_size, 3, height, width) - high-freq enhanced image
    Output shape: (batch_size, num_patches + 1, embedding_dim) - compatible with ViT
    """
    
    def __init__(self, embedding_dim: int = 768, num_patches: int = 196):
        """
        Args:
            embedding_dim: Dimension of BiomedCLIP embedding (default 768 for ViT-B)
            num_patches: Number of patches in ViT (default 196 for 224x224 image with 16x16 patches)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_patches = num_patches
        
        # Convolutional layers to extract spatial features from high-freq
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, embedding_dim, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        
        # Project to patch dimension
        self.patch_projection = nn.Linear(embedding_dim, embedding_dim)
        self.patch_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, high_freq_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            high_freq_image: Enhanced high-frequency image (B, 3, H, W)
        
        Returns:
            Projected high-frequency features (B, num_patches, embedding_dim)
        """
        batch_size = high_freq_image.shape[0]
        
        # Extract spatial features via convolutions
        features = self.feature_extractor(high_freq_image)  # (B, embedding_dim, H', W')
        
        # Flatten spatial dimensions to get patch-like representations
        b, c, h, w = features.shape
        features = features.reshape(b, c, -1).permute(0, 2, 1)  # (B, num_patches, embedding_dim)
        
        # Project and normalize
        features = self.patch_projection(features)
        features = self.patch_norm(features)
        
        return features  # (B, num_patches, embedding_dim)


class FeatureFusionGate(nn.Module):
    """
    Learns how to combine original semantic features with frequency-aware features.
    
    Uses a gating mechanism to adaptively weight the contribution of high-frequency
    features based on the original features' characteristics.
    
    Implements: Feature_fused = Original_feature + alpha * HighFreq_feature
    where alpha is learned adaptively.
    """
    
    def __init__(self, embedding_dim: int = 768, fusion_ratio: float = 0.1):
        """
        Args:
            embedding_dim: Dimension of features
            fusion_ratio: Initial weight for high-frequency injection (0.0 to 1.0)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Learnable parameter for fusion weight
        self.fusion_alpha = nn.Parameter(
            torch.tensor(fusion_ratio, dtype=torch.float32),
            requires_grad=True
        )
        
        # Optional: Learnable linear transformation for frequency features
        self.freq_transform = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )
        
        # Gating mechanism to modulate fusion strength
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        original_features: torch.Tensor, 
        high_freq_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse original and high-frequency features.
        
        Args:
            original_features: Features from BiomedCLIP (B, num_patches+1, embedding_dim)
            high_freq_features: Projected wavelet features (B, num_patches, embedding_dim)
        
        Returns:
            Fused features (B, num_patches+1, embedding_dim)
        """
        # Transform high-freq features
        freq_transformed = self.freq_transform(high_freq_features)  # (B, num_patches, embedding_dim)
        
        # Compute gating weights based on original features
        # Use mean of features as representative
        gate_input = original_features[:, 1:, :].mean(dim=1)  # Exclude class token
        gate_weights = self.gate(gate_input)  # (B, 1)
        
        # Adjust fusion alpha with gating
        adaptive_alpha = self.fusion_alpha * gate_weights  # (B, 1) -> (B, 1, 1)
        adaptive_alpha = adaptive_alpha.unsqueeze(-1)
        
        # Fuse features: keep class token, apply fusion to patch features
        class_token = original_features[:, 0:1, :]  # (B, 1, embedding_dim)
        patch_features = original_features[:, 1:, :]  # (B, num_patches, embedding_dim)
        
        # Ensure high_freq_features has same number of patches
        if high_freq_features.shape[1] != patch_features.shape[1]:
            # Resize if necessary
            if high_freq_features.shape[1] > patch_features.shape[1]:
                high_freq_features = high_freq_features[:, :patch_features.shape[1], :]
            else:
                # Pad with zeros
                padding = torch.zeros(
                    high_freq_features.shape[0],
                    patch_features.shape[1] - high_freq_features.shape[1],
                    high_freq_features.shape[2],
                    device=high_freq_features.device
                )
                high_freq_features = torch.cat([high_freq_features, padding], dim=1)
        
        # Apply fusion
        fused_patches = patch_features + adaptive_alpha * freq_transformed
        
        # Combine class token and patches
        fused_features = torch.cat([class_token, fused_patches], dim=1)
        
        return fused_features


class FrequencyAwareVisionEncoder(nn.Module):
    """
    Wrapper around BiomedCLIP's vision encoder that injects frequency-aware features.
    
    This module intercepts the vision encoder and:
    1. Extracts high-frequency features from the input
    2. Projects them to match encoder dimension
    3. Fuses them with the original patch embeddings
    4. Passes the fused features through the transformer
    """
    
    def __init__(
        self,
        original_encoder: nn.Module,
        embedding_dim: int = 768,
        num_patches: int = 196,
        fusion_ratio: float = 0.1,
        inject_at_layer: int = 0
    ):
        """
        Args:
            original_encoder: The original BiomedCLIP vision encoder
            embedding_dim: Embedding dimension of the encoder
            num_patches: Number of patches in ViT
            fusion_ratio: Initial weight for high-frequency injection
            inject_at_layer: Which transformer layer to inject features (0 = after embedding)
        """
        super().__init__()
        self.original_encoder = original_encoder
        self.embedding_dim = embedding_dim
        self.inject_at_layer = inject_at_layer
        
        # Frequency-aware components
        self.high_freq_projection = HighFreqProjection(
            embedding_dim=embedding_dim,
            num_patches=num_patches
        )
        self.fusion_gate = FeatureFusionGate(
            embedding_dim=embedding_dim,
            fusion_ratio=fusion_ratio
        )
        
        # Flag to indicate whether to use frequency injection
        self.use_freq_injection = True
        
    def forward(
        self,
        x: torch.Tensor,
        high_freq_features: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False
    ) -> Dict:
        """
        Forward pass with optional frequency-aware feature injection.
        
        Args:
            x: Input image tensor (B, 3, H, W)
            high_freq_features: Pre-computed high-frequency features (B, 3, H, W)
                               If None, use standard encoding without frequency injection
            output_hidden_states: Whether to return hidden states
        
        Returns:
            Dict with 'pooler_output' and optionally 'hidden_states'
        """
        if high_freq_features is None or not self.use_freq_injection:
            # Standard forward pass without frequency injection
            return self.original_encoder(x, output_hidden_states=output_hidden_states)
        
        # Get embeddings from original encoder
        # First, we need to extract embedding layer output
        # This depends on the specific architecture of BiomedCLIP
        
        # Project high-frequency features
        high_freq_proj = self.high_freq_projection(high_freq_features)  # (B, num_patches, embedding_dim)
        
        # We need to hook into the encoder's embedding layer
        # For now, we'll call the original encoder and modify its behavior
        # In a real implementation, we'd need to modify the encoder's forward method
        
        # As a simpler approach, we can use the embeddings directly
        # by injecting at the first transformer layer
        
        # Forward through original encoder but with frequency injection
        with torch.no_grad():
            # Get original embeddings
            embeddings = self._get_embeddings_from_encoder(x)  # (B, num_patches+1, embedding_dim)
        
        # Fuse with frequency features
        fused_embeddings = self.fusion_gate(embeddings, high_freq_proj)
        
        # Forward through transformer with fused embeddings
        output = self._forward_transformer_with_embeddings(
            fused_embeddings,
            output_hidden_states=output_hidden_states
        )
        
        return output
    
    def _get_embeddings_from_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from the original encoder without gradient.
        
        This method depends on the specific architecture of BiomedCLIP.
        """
        # This is a placeholder - actual implementation depends on BiomedCLIP's structure
        # For BiomedCLIP with ViT structure:
        if hasattr(self.original_encoder, 'embeddings'):
            return self.original_encoder.embeddings(x)
        elif hasattr(self.original_encoder, 'trunk') and hasattr(self.original_encoder.trunk, 'patch_embed'):
            # For timm-based ViT
            x_patch = self.original_encoder.trunk.patch_embed(x)
            x_patch = x_patch.flatten(2).transpose(1, 2)
            cls_tokens = self.original_encoder.trunk.cls_token.expand(x.size(0), -1, -1)
            x_emb = torch.cat((cls_tokens, x_patch), dim=1)
            x_emb = x_emb + self.original_encoder.trunk.pos_embed
            return x_emb
        else:
            raise NotImplementedError("Cannot extract embeddings from this encoder type")
    
    def _forward_transformer_with_embeddings(
        self,
        embeddings: torch.Tensor,
        output_hidden_states: bool = False
    ) -> Dict:
        """
        Forward embeddings through the transformer blocks.
        """
        # This is a placeholder - actual implementation depends on BiomedCLIP's structure
        return self.original_encoder(embeddings, output_hidden_states=output_hidden_states)


# Wrapper function to convert existing encoder to frequency-aware
def make_frequency_aware(
    encoder: nn.Module,
    embedding_dim: int = 768,
    num_patches: int = 196,
    fusion_ratio: float = 0.1
) -> FrequencyAwareVisionEncoder:
    """
    Convert an existing BiomedCLIP vision encoder to be frequency-aware.
    
    Args:
        encoder: Original BiomedCLIP vision encoder
        embedding_dim: Embedding dimension
        num_patches: Number of patches
        fusion_ratio: Initial fusion ratio
    
    Returns:
        Wrapped encoder with frequency-aware capability
    """
    return FrequencyAwareVisionEncoder(
        encoder,
        embedding_dim=embedding_dim,
        num_patches=num_patches,
        fusion_ratio=fusion_ratio
    )
