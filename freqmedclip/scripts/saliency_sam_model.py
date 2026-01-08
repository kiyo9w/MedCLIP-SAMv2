"""
Saliency Model - Matches FMISeg-original architecture closely.

Key components from FMISeg:
1. LFFI (Language-guided Feature Fusion Integration) Decoder with correct text_len
2. FFBI (Feature Fusion Bidirectional Integration) 
3. Dual decoder branches with deep supervision
4. Correct spatial dimensions [7, 14, 28, 56] matching FMISeg
5. Correct text_len [24, 12, 9] matching FMISeg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from typing import Optional, Dict
from monai.networks.blocks.unetr_block import UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample

from freqmedclip.scripts.dwt_projector import DWTFrequencyProjector


# ============== FMISeg Components (Exact Match) ==============

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding - exact FMISeg implementation."""
    def __init__(self, d_model: int, dropout=0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SelfAugment(nn.Module):
    """Self-attention augmentation - FMISeg uses num_heads=1!"""
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.vis_pos = PositionalEncoding(in_channels)
        # FMISeg uses num_heads=1, not 4!
        self.self_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=1, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        vis = self.norm(x)
        q = k = self.vis_pos(vis)
        vis = self.self_attn(q, k, value=vis)[0]
        vis = self.self_attn_norm(vis)
        vis = x + vis
        return vis


class FeedLinear(nn.Module):
    """Feed-forward linear block."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class LFFI(nn.Module):
    """
    Language-guided Feature Fusion Integration - FMISeg exact implementation.
    
    CRITICAL: FMISeg uses input_text_len=24 (short prompts), not 77!
    """
    def __init__(self, in_channels: int, output_text_len: int, input_text_len: int = 24, embed_dim: int = 768):
        super().__init__()
        self.in_channels = in_channels
        self.augment = SelfAugment(in_channels)
        self.cross_attn_norm = nn.LayerNorm(in_channels)
        self.cross_attn1 = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, batch_first=True)
        self.cross_attn2 = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, batch_first=True)
        
        self.text_project = nn.Sequential(
            nn.Conv1d(input_text_len, output_text_len, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Linear(embed_dim, in_channels),
            nn.LeakyReLU(),
        )
        
        self.vis_pos = PositionalEncoding(in_channels)
        self.txt_pos = PositionalEncoding(in_channels, max_len=output_text_len)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.norm3 = nn.LayerNorm(in_channels)
        self.norm4 = nn.LayerNorm(in_channels)
        self.norm5 = nn.LayerNorm(in_channels)
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.fl1 = FeedLinear(in_channels, in_channels * 2)
        self.fl2 = FeedLinear(in_channels, in_channels * 2)
        self.line = nn.Linear(output_text_len, in_channels)

    def forward(self, x, txt):
        """
        Args:
            x: [B, N, C] - Visual features
            txt: [B, L, embed_dim] - Text features (L=24 in FMISeg)
        """
        txt = self.text_project(txt)
        vis = self.augment(x)
        vis2 = self.norm1(vis)
        
        # Bidirectional cross-attention
        vis2_v, _ = self.cross_attn1(
            query=self.vis_pos(vis2),
            key=self.txt_pos(txt),
            value=txt
        )
        vis2_l, _ = self.cross_attn2(
            query=self.txt_pos(txt),
            key=self.vis_pos(vis2),
            value=vis2
        )
        
        vis2_v = self.norm2(vis2_v + vis2)
        vis2_l = self.norm3(vis2_l + txt)
        vis2_v = self.norm4(self.fl1(vis2_v) + vis2_v)
        vis2_l = self.norm5(self.fl2(vis2_l) + vis2_l)
        vis2 = vis2_v + self.line(torch.matmul(vis2_v, vis2_l.transpose(1, 2)))
        vis2 = self.cross_attn_norm(vis2)
        vis = vis * self.scale * vis2
        return vis


class FFBI(nn.Module):
    """Feature Fusion Bidirectional Integration - FMISeg exact implementation."""
    def __init__(self, dim, num_heads=4, batch_first=True):
        super().__init__()
        self.cross_attn_h = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=batch_first)
        self.cross_attn_l = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=batch_first)

    def forward(self, x, y):
        x1, _ = self.cross_attn_l(query=x, key=y, value=y)
        x2 = x1 + x
        y1, _ = self.cross_attn_h(query=y, key=x, value=x)
        y2 = y1 + y
        return x2, y2


class LFFIDecoder(nn.Module):
    """Decoder block with LFFI text guidance - FMISeg exact implementation."""
    def __init__(self, in_channels, out_channels, spatial_size, text_len, input_text_len=24, embed_dim=768):
        super().__init__()
        self.lffi_layer = LFFI(in_channels, text_len, input_text_len=input_text_len, embed_dim=embed_dim)
        self.spatial_size = spatial_size
        self.decoder = UnetrUpBlock(2, in_channels, out_channels, 3, 2, norm_name='BATCH')

    def forward(self, vis, skip_vis, txt):
        if txt is not None:
            vis = self.lffi_layer(vis, txt)
        vis = rearrange(vis, 'B (H W) C -> B C H W', H=self.spatial_size, W=self.spatial_size)
        skip_vis = rearrange(skip_vis, 'B (H W) C -> B C H W', H=self.spatial_size * 2, W=self.spatial_size * 2)
        output = self.decoder(vis, skip_vis)
        output = rearrange(output, 'B C H W -> B (H W) C')
        return output


# ============== Main Model ==============

class SaliencyModel(nn.Module):
    """
    Saliency Model - Closely matches FMISeg-original architecture.
    
    Key differences from original FMISeg:
    - Uses BiomedCLIP ViT instead of separate ConvNeXt encoders
    - Uses on-the-fly DWT instead of preprocessed H/L images
    - Single encoder with DWT injection instead of dual encoders
    
    Architecture matches FMISeg:
    - spatial_dim = [7, 14, 28, 56] (NOT [14, 28, 56, 112])
    - text_len = [24, 12, 9] (NOT [77, 38, 19])
    - SubpixelUpsample scale=4 (56→224)
    """
    
    def __init__(
        self,
        biomedclip_model,
        input_size: int = 224,
        freeze_biomedclip: bool = True,
        unfreeze_layers: list = [3, 6, 9, 11],
        max_text_len: int = 24  # FMISeg uses 24, not 77!
    ):
        super().__init__()
        
        self.biomedclip = biomedclip_model
        self.input_size = input_size
        self.vit_input_size = 224
        self.vit_spatial_size = 14  # 224/16 = 14
        self.max_text_len = max_text_len
        
        # Freeze BiomedCLIP
        if freeze_biomedclip:
            for param in self.biomedclip.parameters():
                param.requires_grad = False
                
            if hasattr(self.biomedclip, 'vision_model'):
                encoder = self.biomedclip.vision_model.encoder
                if hasattr(encoder, 'layers'):
                    for i in unfreeze_layers:
                        if i < len(encoder.layers):
                            for param in encoder.layers[i].parameters():
                                param.requires_grad = True
        
        # === DWT Frequency Projector ===
        self.dwt_projector = DWTFrequencyProjector(12, 768, 256)
        
        # === Pool from 14x14 to 7x7 to match FMISeg ===
        self.spatial_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # === FFBI for bidirectional fusion ===
        self.ffbi = FFBI(dim=768, num_heads=4)
        
        # === FMISeg exact spatial dimensions ===
        # spatial_dim[0]=7 is DEEPEST (os32 in FMISeg terminology)
        self.spatial_dim = [7, 14, 28, 56]  # FIXED to match FMISeg!
        feature_dim = [768, 384, 192, 96]
        
        # === FMISeg exact text_len values ===
        # text_len decreases as spatial resolution increases
        text_len = [24, 12, 9]  # FIXED to match FMISeg!
        
        # === Branch 1: Semantic-enhanced decoder (matches FMISeg exactly) ===
        self.decoder16 = LFFIDecoder(feature_dim[0], feature_dim[1], self.spatial_dim[0], text_len[0], max_text_len)
        self.decoder8 = LFFIDecoder(feature_dim[1], feature_dim[2], self.spatial_dim[1], text_len[1], max_text_len)
        self.decoder4 = LFFIDecoder(feature_dim[2], feature_dim[3], self.spatial_dim[2], text_len[2], max_text_len)
        self.decoder1 = SubpixelUpsample(2, feature_dim[3], 24, 4)  # scale=4 to go 56→224
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)
        
        # === Branch 2: Frequency-enhanced decoder (matches FMISeg exactly) ===
        self.decoder16_2 = LFFIDecoder(feature_dim[0], feature_dim[1], self.spatial_dim[0], text_len[0], max_text_len)
        self.decoder8_2 = LFFIDecoder(feature_dim[1], feature_dim[2], self.spatial_dim[1], text_len[1], max_text_len)
        self.decoder4_2 = LFFIDecoder(feature_dim[2], feature_dim[3], self.spatial_dim[2], text_len[2], max_text_len)
        self.decoder1_2 = SubpixelUpsample(2, feature_dim[3], 24, 4)  # scale=4 to go 56→224
        self.out_2 = UnetOutBlock(2, in_channels=24, out_channels=1)
        
        # === Projection layers for contrastive loss ===
        self.img_proj = nn.Linear(768, 512)
        self.txt_proj = nn.Linear(768, 512)
        
        # === Skip connection projectors (ViT produces 768-dim at all layers) ===
        # Project from 768 to decoder dimensions
        self.skip_proj_14 = nn.Linear(768, feature_dim[1])  # 768 -> 384 for 14x14
        self.skip_proj_28 = nn.Linear(768, feature_dim[2])  # 768 -> 192 for 28x28
        self.skip_proj_56 = nn.Linear(768, feature_dim[3])  # 768 -> 96 for 56x56

    def forward(self, images, input_ids, raw_images=None):
        B = images.shape[0]
        device = images.device
        
        if raw_images is None:
            raw_images = images
            
        # === 1. Resize for ViT ===
        if images.shape[-1] != self.vit_input_size:
            vit_images = F.interpolate(images, size=(self.vit_input_size, self.vit_input_size), 
                                       mode='bilinear', align_corners=False)
        else:
            vit_images = images
            
        # === 2. Vision encoding ===
        vision_out = self.biomedclip.vision_model(vit_images, output_hidden_states=True, return_dict=True)
        
        if hasattr(vision_out, 'hidden_states'):
            hidden_states = vision_out.hidden_states
        elif isinstance(vision_out, tuple) and len(vision_out) > 2:
            hidden_states = vision_out[2]
        else:
            hidden_states = [vision_out[0]] * 13
        
        # Get pooled embedding for contrastive loss
        if hasattr(vision_out, 'pooler_output'):
            image_embed = vision_out.pooler_output
        else:
            image_embed = vision_out[0][:, 0, :]
            
        if image_embed.shape[-1] != 512:
            image_embed = self.img_proj(image_embed)
            
        # === 3. Text encoding ===
        text_out = self.biomedclip.text_model(input_ids, output_hidden_states=True, return_dict=True)
        
        if hasattr(text_out, 'hidden_states'):
            text_hidden = text_out.hidden_states[-1]  # [B, L, 768]
        elif isinstance(text_out, tuple) and len(text_out) > 2:
            text_hidden = text_out[2][-1]
        else:
            text_hidden = text_out[0]
            
        # Truncate/pad text to max_text_len (FMISeg uses 24)
        if text_hidden.shape[1] > self.max_text_len:
            text_hidden = text_hidden[:, :self.max_text_len, :]
        elif text_hidden.shape[1] < self.max_text_len:
            padding = torch.zeros(B, self.max_text_len - text_hidden.shape[1], 768, device=device)
            text_hidden = torch.cat([text_hidden, padding], dim=1)
            
        if hasattr(text_out, 'pooler_output'):
            text_embed = text_out.pooler_output
        else:
            text_embed = text_hidden[:, 0, :]
            
        if text_embed.shape[-1] != 512:
            text_embed = self.txt_proj(text_embed)
            
        # === 4. DWT Frequency features at 7x7 (to match pooled ViT) ===
        freq_features_7 = self.dwt_projector(raw_images, target_size=(7, 7))
        freq_features_flat = rearrange(freq_features_7, 'b c h w -> b (h w) c')  # [B, 49, 768]
        
        # === 5. Get ViT features and pool to 7x7 for deepest ===
        # Use last layer for deepest features
        feat_deep = hidden_states[-1][:, 1:, :]  # Remove CLS, [B, 196, 768]
        feat_deep_spatial = rearrange(feat_deep, 'b (h w) c -> b c h w', h=14)  # [B, 768, 14, 14]
        feat_7x7 = self.spatial_pool(feat_deep_spatial)  # [B, 768, 7, 7]
        feat_7x7_flat = rearrange(feat_7x7, 'b c h w -> b (h w) c')  # [B, 49, 768]
        
        # Get features from different layers for skip connections
        # Layer indices: early=2, mid=5, late=8
        feat_early = hidden_states[2][:, 1:, :] if len(hidden_states) > 2 else feat_deep
        feat_mid = hidden_states[5][:, 1:, :] if len(hidden_states) > 5 else feat_deep
        feat_late = hidden_states[8][:, 1:, :] if len(hidden_states) > 8 else feat_deep
        
        # === 6. Early fusion: Add DWT to pooled visual features ===
        enhanced_semantic = feat_7x7_flat + freq_features_flat  # Semantic branch [B, 49, 768]
        enhanced_freq = freq_features_flat + feat_7x7_flat * 0.5  # Freq branch (emphasize DWT) [B, 49, 768]
        
        # === 7. FFBI bidirectional fusion ===
        fu32, fu32_2 = self.ffbi(enhanced_semantic, enhanced_freq)  # Both [B, 49, 768]
        
        # === 8. Prepare skip connections at correct spatial sizes ===
        # Skip for 14x14 (from ViT's 14x14)
        skip_14 = self.skip_proj_14(feat_late)  # [B, 196, 384]
        
        # Skip for 28x28 (upsample from ViT's 14x14)
        skip_28_spatial = F.interpolate(
            rearrange(feat_mid, 'b (h w) c -> b c h w', h=14),
            size=(28, 28), mode='bilinear', align_corners=False
        )
        skip_28 = rearrange(skip_28_spatial, 'b c h w -> b (h w) c')
        skip_28 = self.skip_proj_28(skip_28)  # [B, 784, 192]
        
        # Skip for 56x56 (upsample from ViT's 14x14)
        skip_56_spatial = F.interpolate(
            rearrange(feat_early, 'b (h w) c -> b c h w', h=14),
            size=(56, 56), mode='bilinear', align_corners=False
        )
        skip_56 = rearrange(skip_56_spatial, 'b c h w -> b (h w) c')
        skip_56 = self.skip_proj_56(skip_56)  # [B, 3136, 96]
        
        # === 9. Branch 1 Decoder (Semantic-enhanced) ===
        # 7x7 → 14x14
        os16 = self.decoder16(fu32, skip_14, text_hidden)  # [B, 196, 384]
        # 14x14 → 28x28
        os8 = self.decoder8(os16, skip_28, text_hidden)  # [B, 784, 192]
        # 28x28 → 56x56
        os4 = self.decoder4(os8, skip_56, text_hidden)  # [B, 3136, 96]
        os4_spatial = rearrange(os4, 'B (H W) C -> B C H W', H=56, W=56)
        # 56x56 → 224x224
        os1 = self.decoder1(os4_spatial)
        out1 = self.out(os1).sigmoid()
        
        # === 10. Branch 2 Decoder (Frequency-enhanced) ===
        os16_2 = self.decoder16_2(fu32_2, skip_14, text_hidden)
        os8_2 = self.decoder8_2(os16_2, skip_28, text_hidden)
        os4_2 = self.decoder4_2(os8_2, skip_56, text_hidden)
        os4_2_spatial = rearrange(os4_2, 'B (H W) C -> B C H W', H=56, W=56)
        os1_2 = self.decoder1_2(os4_2_spatial)
        out2 = self.out_2(os1_2).sigmoid()
        
        # === 11. Resize outputs to input size if needed ===
        target_h, target_w = images.shape[-2:]
        if out1.shape[-1] != target_h:
            out1 = F.interpolate(out1, size=(target_h, target_w), mode='bilinear', align_corners=False)
            out2 = F.interpolate(out2, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        return {
            'pred1': out1,  # Main branch
            'pred2': out2,  # Auxiliary branch
            'image_embed': image_embed,
            'text_embed': text_embed
        }
