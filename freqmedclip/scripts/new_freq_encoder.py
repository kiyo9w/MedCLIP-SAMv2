import torch
import torch.nn as nn
import timm

class ConvNeXtTiny12Ch(nn.Module):
    """
    ConvNeXt-Tiny Encoder adapted for 12-channel input (Frequency Domain).
    Returns multi-scale features for FPNAdapter.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        # Load ConvNeXt-Tiny
        # feature_only=True returns a list of features from different stages
        # out_indices=(0, 1, 2, 3) corresponds to strides 4, 8, 16, 32
        self.model = timm.create_model(
            'convnext_tiny', 
            pretrained=pretrained, 
            features_only=True,
            out_indices=(0, 1, 2, 3) 
        )
        
        # --- Dilated Convolutions for Resolution Preservation ---
        # Hack strides in Stage 2 and 3 to prevent downsampling below 32x32.
        # Input (DWT) 256x256 -> s1(64) -> s2(32).
        # We want s3 and s4 to stay at 32x32 (instead of 16 and 8).
        if hasattr(self.model, 'stages'):
            # Stage 2 (Index 2) normally stride 2 -> 16x16. Force stride 1.
            self.model.stages[2].downsample[1].stride = (1, 1) 
            # Stage 3 (Index 3) normally stride 2 -> 8x8. Force stride 1.
            self.model.stages[3].downsample[1].stride = (1, 1) 
        
        # Modify Stem (First Conv Layer)
        # Original: Conv2d(3, 96, kernel_size=(4, 4), stride=(4, 4))
        original_stem = self.model.stem[0]
        
        new_stem = nn.Conv2d(
            in_channels=12, 
            out_channels=original_stem.out_channels, 
            kernel_size=original_stem.kernel_size, 
            stride=original_stem.stride,
            padding=original_stem.padding,
            bias=original_stem.bias is not None
        )
        
        # Weights Initialization Strategy:
        # We need to adapt the 3-channel weights to 12 channels.
        # Strategy: Repeat the weights 4 times. 
        # While not perfect, it's better than random init.
        
        with torch.no_grad():
            if original_stem.weight is not None:
                # (Out, In, kH, kW)
                orig_w = original_stem.weight.data
                # Repeat along In-channel dimension (dim 1) 4 times (3->12)
                new_w = orig_w.repeat(1, 4, 1, 1)
                # Scale by 1/4 to maintain variance
                new_w = new_w / 4.0 
                new_stem.weight.copy_(new_w)
            
            if original_stem.bias is not None:
                new_stem.bias.copy_(original_stem.bias)
                
        self.model.stem[0] = new_stem
        
        # Determine channel dimensions
        # ConvNeXt-Tiny dims: [96, 192, 384, 768]
        self.feature_info = self.model.feature_info.channels()

    def forward(self, x, text_embeds=None):
        # x: (B, 12, H, W)
        # text_embeds: Ignored for pure visual backbone
        
        features = self.model(x)
        # Features are: [s1(4x), s2(8x), s3(16x), s4(32x)]
        # For ConvNeXt-Tiny: [96, 192, 384, 768]
        
        # We need to reverse this to match the FPN usage in train_freq_fusion.py which expects:
        # [f3, f2, f1, f0] where f0 is the highest res/lowest level
        # Wait, usually FPN expects High Level -> Low Level (Small HxW -> Big HxW)?
        # Let's check train_freq_fusion.py:
        # freq_feats = self.freq_encoder(...) -> [f3, f2, f1, f0]
        # image_features2 = [freq_feats[0], freq_feats[1], freq_feats[2], f0]
        # os32_2 = image_features2[0] -> Used for Bottleneck (Smallest spatial)
        
        # So freq_feats[0] should be Deepest/Smallest (s4)
        # freq_feats[3] should be Shallowest/Largest (s1)
        
        # Return: [s4(768), s3(384), s2(192), s1(96)]
        return features[::-1] 