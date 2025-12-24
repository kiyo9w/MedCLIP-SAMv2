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
        
        # Modify Stem (First Conv Layer) to accept 12 channels.
        # Original common stem: Conv2d(3, 96, kernel_size=(4,4), stride=(4,4))
        original_stem = None
        replaced = False

        # Preferred: model has `.stem` attribute (older timm versions)
        if hasattr(self.model, 'stem'):
            try:
                original_stem = self.model.stem[0]
            except Exception:
                original_stem = None

        # If not found, search for first Conv2d with in_channels==3
        if original_stem is None:
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d) and getattr(m, 'in_channels', None) == 3:
                    original_stem = m
                    break

        if original_stem is None:
            # As a last resort, pick the first Conv2d
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    original_stem = m
                    break

        if original_stem is None:
            raise RuntimeError('Could not find a Conv2d stem in the ConvNeXt model to adapt.')

        # Create a new stem conv matching original out/in/out kernel but with in_channels=12
        new_stem = nn.Conv2d(
            in_channels=12,
            out_channels=original_stem.out_channels,
            kernel_size=original_stem.kernel_size,
            stride=original_stem.stride if hasattr(original_stem, 'stride') else (1, 1),
            padding=original_stem.padding if hasattr(original_stem, 'padding') else 0,
            bias=original_stem.bias is not None
        )

        # Weights Initialization Strategy: repeat existing RGB weights 4x along channel dim when possible
        with torch.no_grad():
            if hasattr(original_stem, 'weight') and original_stem.weight is not None:
                orig_w = original_stem.weight.data
                if orig_w.shape[1] == 3:
                    new_w = orig_w.repeat(1, 4, 1, 1) / 4.0
                else:
                    # If original stem had different in_channels, initialize by repeating mean across channels
                    mean_w = orig_w.mean(dim=1, keepdim=True)
                    repeat_factor = 12 // mean_w.shape[1] if mean_w.shape[1] != 0 else 12
                    new_w = mean_w.repeat(1, repeat_factor, 1, 1)
                # If target in_channels != new_w.shape[1], pad or trim
                if new_w.shape[1] != new_stem.weight.shape[1]:
                    if new_w.shape[1] > new_stem.weight.shape[1]:
                        new_w = new_w[:, :new_stem.weight.shape[1], ...]
                    else:
                        pad = new_stem.weight.shape[1] - new_w.shape[1]
                        pad_tensor = new_w[:, :1, ...].repeat(1, pad, 1, 1)
                        new_w = torch.cat([new_w, pad_tensor], dim=1)
                new_stem.weight.copy_(new_w)
            if hasattr(original_stem, 'bias') and original_stem.bias is not None:
                new_stem.bias.copy_(original_stem.bias)

        # Try to replace the module in-place where it lives
        for parent_name, parent_mod in self.model.named_modules():
            for child_name, child_mod in parent_mod.named_children():
                if child_mod is original_stem:
                    setattr(parent_mod, child_name, new_stem)
                    replaced = True
                    break
            if replaced:
                break

        # If replacement didn't work (structure mismatch), keep a pre_stem that maps 12->original_in and use it in forward
        self.pre_stem = None
        if not replaced:
            orig_in = getattr(original_stem, 'in_channels', None) or 3
            if orig_in != 12:
                self.pre_stem = nn.Conv2d(12, orig_in, kernel_size=1, stride=1, padding=0, bias=False)
                nn.init.kaiming_normal_(self.pre_stem.weight, mode='fan_out', nonlinearity='relu')
            else:
                # As fallback assign to model attribute if possible
                try:
                    # Try to set at top-level if 'stem' exists
                    if hasattr(self.model, 'stem'):
                        self.model.stem[0] = new_stem
                        replaced = True
                except Exception:
                    pass

        # If we replaced in-place, ensure pre_stem is None
        if replaced:
            self.pre_stem = None
        
        # Determine channel dimensions
        # ConvNeXt-Tiny dims: [96, 192, 384, 768]
        self.feature_info = self.model.feature_info.channels()

    def forward(self, x, text_embeds=None):
        # x: (B, 12, H, W)
        # text_embeds: Ignored for pure visual backbone
        # If we have a pre_stem mapping 12->orig_in, use it first
        if hasattr(self, 'pre_stem') and self.pre_stem is not None:
            x = self.pre_stem(x)

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