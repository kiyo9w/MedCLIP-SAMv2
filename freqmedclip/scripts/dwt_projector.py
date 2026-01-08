"""
DWT Frequency Projector for M2IB Saliency Architecture.

Takes raw image, applies Haar DWT, downsamples to match ViT spatial dimensions,
and projects to 768 channels for injection into Block 3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from freqmedclip.scripts.freq_components import haar_dwt


class DWTFrequencyProjector(nn.Module):
    """
    DWT on raw image -> downsample to match ViT spatial -> project to 768 channels.
    
    For 512x512 input with ViT patch_size=16:
    - DWT output: [B, 12, 256, 256] (4 bands * 3 RGB)
    - Downsample to: [B, 12, 32, 32] (matching 512/16 = 32 ViT patches)
    - Project to: [B, 768, 32, 32] (matching ViT hidden dim)
    
    For 224x224 input with ViT patch_size=16:
    - DWT output: [B, 12, 112, 112]
    - Downsample to: [B, 12, 14, 14] (matching 224/16 = 14 ViT patches)
    - Project to: [B, 768, 14, 14]
    """
    
    def __init__(self, in_channels=12, out_channels=768, intermediate_channels=256):
        super().__init__()
        
        # Multi-layer projection for better feature extraction
        self.projector = nn.Sequential(
            # First conv: 12 -> 64
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Second conv: 64 -> 256
            nn.Conv2d(64, intermediate_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
            
            # Final conv: 256 -> 768
            nn.Conv2d(intermediate_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        
    def forward(self, x, target_size=None):
        """
        Args:
            x: Raw image [B, 3, H, W]
            target_size: Target spatial size (H', W') to match ViT. 
                         If None, uses H//16, W//16 (assuming patch_size=16)
        
        Returns:
            freq_features: [B, 768, H', W'] - frequency features matching ViT spatial
        """
        B, C, H, W = x.shape
        
        # Apply Haar DWT: [B, 3, H, W] -> [B, 12, H/2, W/2]
        dwt_out = haar_dwt(x)
        
        # Determine target size
        if target_size is None:
            target_h = H // 16  # ViT patch size
            target_w = W // 16
        else:
            target_h, target_w = target_size
            
        # Downsample DWT output to match ViT spatial dimensions
        # DWT is at H/2, W/2, we need H/16, W/16 (so downsample by 8x)
        dwt_downsampled = F.interpolate(
            dwt_out, 
            size=(target_h, target_w), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Project to 768 channels
        freq_features = self.projector(dwt_downsampled)
        
        return freq_features


class DWTHighFreqOnly(nn.Module):
    """
    Variant that only uses high-frequency bands (LH, HL, HH), excluding LL.
    This focuses purely on edge/texture information.
    
    For 512x512 input:
    - High-freq DWT: [B, 9, 256, 256] (3 high-freq bands * 3 RGB)
    - Output: [B, 768, 32, 32]
    """
    
    def __init__(self, in_channels=9, out_channels=768, intermediate_channels=256):
        super().__init__()
        
        self.projector = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, intermediate_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(intermediate_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        
    def forward(self, x, target_size=None):
        """
        Args:
            x: Raw image [B, 3, H, W]
            target_size: Target spatial size
        
        Returns:
            high_freq_features: [B, 768, H', W']
        """
        B, C, H, W = x.shape
        
        # Apply Haar DWT: [B, 3, H, W] -> [B, 12, H/2, W/2]
        dwt_out = haar_dwt(x)
        
        # Extract only high-frequency bands (LH, HL, HH), skip LL
        # haar_dwt returns [LL, LH, HL, HH] concatenated, each has C channels
        # So for 3-channel input: [0:3]=LL, [3:6]=LH, [6:9]=HL, [9:12]=HH
        high_freq = dwt_out[:, 3:, :, :]  # [B, 9, H/2, W/2]
        
        # Determine target size
        if target_size is None:
            target_h = H // 16
            target_w = W // 16
        else:
            target_h, target_w = target_size
            
        # Downsample
        high_freq_downsampled = F.interpolate(
            high_freq, 
            size=(target_h, target_w), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Project
        high_freq_features = self.projector(high_freq_downsampled)
        
        return high_freq_features
