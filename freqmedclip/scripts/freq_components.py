import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 0. DWT/IDWT Components ---

def haar_dwt(x):
    """
    Perform Haar Wavelet Transform on a batch of images.
    x: (B, C, H, W)
    Returns: (B, 4*C, H/2, W/2) -> Stacked [LL, LH, HL, HH]
    """
    B, C, H, W = x.shape
    # Ensure dimensions are even
    if H % 2 != 0 or W % 2 != 0:
        x = F.pad(x, (0, W % 2, 0, H % 2), mode='reflect')
        B, C, H, W = x.shape
        
    x = x.view(B, C, H // 2, 2, W // 2, 2)
    
    # 4 Sub-bands
    # x: (B, C, H/2, 2, W/2, 2)
    # Slice for vectorization
    x00 = x[:, :, :, 0, :, 0] # Top-Left
    x01 = x[:, :, :, 0, :, 1] # Top-Right
    x10 = x[:, :, :, 1, :, 0] # Bottom-Left
    x11 = x[:, :, :, 1, :, 1] # Bottom-Right
    
    # Haar Formulas (Normalized averages)
    # LL: Low Frequency (Approximation)
    LL = (x00 + x01 + x10 + x11) / 4.0
    # LH: Vertical Detail
    LH = (x00 - x01 + x10 - x11) / 4.0 
    # HL: Horizontal Detail
    HL = (x00 + x01 - x10 - x11) / 4.0 
    # HH: Diagonal Detail
    HH = (x00 - x01 - x10 + x11) / 4.0 
    
    # Stack channels: (B, 4C, H/2, W/2)
    # Order: LL, LH, HL, HH
    out = torch.cat([LL, LH, HL, HH], dim=1)
    return out

# --- 1. LFFI Components (FMISeg Original Logic) ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, dropout=0, max_len:int=5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0)  # size=(1, L, d_model)
        self.register_buffer('pe', pe)  

    def forward(self, x):
        # x: [B, L, D]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class SelfAugment(nn.Module):
    def __init__(self, in_channels):
        super(SelfAugment, self).__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.vis_pos = PositionalEncoding(in_channels)
        self.self_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=1, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        # x: [B, L, C]
        vis = self.norm(x)
        q = k = self.vis_pos(vis)
        vis = self.self_attn(q, k, value=vis)[0]
        vis = self.self_attn_norm(vis)
        vis = x + vis
        return vis

class FeedLinear(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedLinear, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class CrossAttentionLFFI(nn.Module):
    """
    Implements Language-Frequency Feature Interaction using Cross-Attention.
    Matches the logic from FMISeg-original/net/decoder.py
    """
    def __init__(self, in_channels:int, output_text_len:int=77, input_text_len:int=77, embed_dim:int=768):
        super(CrossAttentionLFFI, self).__init__()
        self.in_channels = in_channels
        
        # Self-Augment for Visual Features
        self.augment = SelfAugment(in_channels)
        
        self.cross_attn_norm = nn.LayerNorm(in_channels)
        
        # Dual Cross Attention
        self.cross_attn1 = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, batch_first=True)
        self.cross_attn2 = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, batch_first=True)
        
        # Text Projection
        self.text_project = nn.Sequential(
            nn.Linear(embed_dim, in_channels), # Project embedding dim
            nn.LayerNorm(in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(),
        )
        
        # Positional Encodings
        self.vis_pos = PositionalEncoding(in_channels)
        self.txt_pos = PositionalEncoding(in_channels, max_len=output_text_len)
        
        # Normalizations
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.norm3 = nn.LayerNorm(in_channels)
        self.norm4 = nn.LayerNorm(in_channels)
        self.norm5 = nn.LayerNorm(in_channels)
        
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        
        # Feed Forward Networks
        self.fl1 = FeedLinear(in_channels, in_channels*2)
        self.fl2 = FeedLinear(in_channels, in_channels*2)
        
        # Final Projection
        self.line = nn.Linear(output_text_len, in_channels)

    def forward(self, x, txt):
        '''
        x: [B, C, H, W] -> Will be flattened to [B, HW, C]
        txt: [B, L, D] (e.g., [B, 77, 768])
        '''
        B, C, H, W = x.shape
        
        # Flatten visual features: [B, C, H, W] -> [B, HW, C]
        vis_flat = x.flatten(2).transpose(1, 2)
        
        # Project Text: [B, L, D] -> [B, L, C]
        txt_proj = self.text_project(txt) # [B, L, C]
        
        # Self Augment Visual
        vis = self.augment(vis_flat) # [B, HW, C]
        vis2 = self.norm1(vis)
        
        # Cross Attn 1: Query=Visual, Key=Text, Value=Text
        vis2_v, _ = self.cross_attn1(query=self.vis_pos(vis2),
                                     key=self.txt_pos(txt_proj),
                                     value=txt_proj)
        
        # Cross Attn 2: Query=Text, Key=Visual, Value=Visual
        vis2_l, _ = self.cross_attn2(query=self.txt_pos(txt_proj),
                                     key=self.vis_pos(vis2),
                                     value=vis2)
                                     
        vis2_v = self.norm2(vis2_v + vis2)
        vis2_l = self.norm3(vis2_l + txt_proj)
        
        vis2_v = self.norm4(self.fl1(vis2_v) + vis2_v)
        vis2_l = self.norm5(self.fl2(vis2_l) + vis2_l)
        
        # Interaction: vis2_v + Linear(vis2_v @ vis2_l.T)
        interaction = torch.matmul(vis2_v, vis2_l.transpose(1, 2))
        vis2 = vis2_v + self.line(interaction)
        
        vis2 = self.cross_attn_norm(vis2)
        
        # Scale and Residual
        vis = vis * self.scale * vis2
        
        # Reshape back: [B, HW, C] -> [B, C, H, W]
        vis = vis.transpose(1, 2).view(B, C, H, W)
        
        return vis

# --- 2. FPN Adapter (NEW) ---

class FPNAdapter(nn.Module):
    """
    Adapts the Multi-Scale ViT outputs to a Feature Pyramid.
    Inputs: [feat_last(14), feat_s3(14), feat_s2(14), feat_s1(14)] -> All 14x14
    We want outputs:
    s1: 14x14 (from layer 12)
    s2: 28x28 (from layer 9)
    s3: 56x56 (from layer 6)
    s4: 112x112 (from layer 3)
    """
    def __init__(self, in_channels=768, out_channels=[768, 384, 192, 96]):
        super().__init__()
        
        # Scale 1 (14x14) - Bottleneck (from Layer 12)
        self.scale1_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[0], kernel_size=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # Scale 2 (28x28) - Skip 1 (from Layer 9)
        self.scale2_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels[1]),
            nn.ReLU(inplace=True)
        )
        
        # Scale 3 (56x56) - Skip 2 (from Layer 6)
        self.scale3_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels[2]),
            nn.ReLU(inplace=True)
        )
        
        # Scale 4 (112x112) - RESTORED: Resolution Hallucination Fix
        self.scale4_up = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels[3]),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        # features: list of [feat_layer12, feat_layer9, feat_layer6, feat_layer3]
        x12, x9, x6, x3 = features
        
        s1 = self.scale1_conv(x12) # 14x14
        s2 = self.scale2_up(x9)    # 28x28
        s3 = self.scale3_up(x6)    # 56x56
        s4 = self.scale4_up(x3)    # 112x112
        
        return [s1, s2, s3, s4]

# --- 3. Frequency Components ---

class TextGuidedSEBlock(nn.Module):
    """
    Squeeze-and-Excitation block conditioned on text embeddings.
    """
    def __init__(self, channels, text_dim=768, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.text_proj = nn.Linear(text_dim, channels)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, text_embeds):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        
        if text_embeds.dim() == 3:
            t = text_embeds.mean(dim=1)
        else:
            t = text_embeds
            
        t_proj = self.text_proj(t)
        y = y + t_proj
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class FrequencyEncoder(nn.Module):
    """
    Parallel Encoder for Frequency Features.
    Processes DWT output (12 channels: 3 RGB * 4 bands, 112x112) and returns multi-scale features.
    """
    def __init__(self, in_channels=12, base_channels=64, text_dim=768):
        super().__init__()
        
        # Input: (B, 12, 112, 112) (DWT output)
        
        # Layer 0: 112x112 Processing (No downsample)
        # 12 channels -> 64 channels
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Layer 1: 112 -> 56 (Output Scale 3)
        self.layer1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        self.se1 = TextGuidedSEBlock(base_channels*2, text_dim)
        
        # Layer 2: 56 -> 28 (Output Scale 2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True)
        )
        self.se2 = TextGuidedSEBlock(base_channels*4, text_dim)
        
        # Layer 3: 28 -> 14 (Output Scale 1)
        self.layer3 = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(inplace=True)
        )
        self.se3 = TextGuidedSEBlock(base_channels*8, text_dim)
        
    def forward(self, x, text_embeds=None):
        # x: 112x112 (from DWT)
        f0 = self.layer0(x) # 112x112
        
        f1 = self.layer1(f0) # 56x56
        if text_embeds is not None: f1 = self.se1(f1, text_embeds)
        
        f2 = self.layer2(f1) # 28x28
        if text_embeds is not None: f2 = self.se2(f2, text_embeds)
        
        f3 = self.layer3(f2) # 14x14
        if text_embeds is not None: f3 = self.se3(f3, text_embeds)
        
        # Return [f3(14), f2(28), f1(56), f0(112)]
        return [f3, f2, f1, f0]

class BottleneckFusion(nn.Module):
    """
    Fuses ViT features and Frequency Encoder features at the bottleneck.
    """
    def __init__(self, dim=768, freq_dim=256):
        super().__init__()
        
        self.freq_proj = nn.Conv2d(freq_dim, dim, kernel_size=1)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x_vit, x_freq):
        # x_vit: (B, C, 14, 14)
        # x_freq: (B, C_freq, 14, 14)
        
        x_freq_proj = self.freq_proj(x_freq)
        x_cat = torch.cat([x_vit, x_freq_proj], dim=1)
        x_fused = self.fusion(x_cat)
        
        return x_fused + x_vit

class SmartDecoderBlock(nn.Module):
    """
    Upsamples features and fuses them with skip connections AND frequency maps.
    """
    def __init__(self, in_channels, out_channels, skip_channels=0, freq_channels=0, use_lffi=False):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Input channels = Previous + Skip + Frequency
        conv_in_channels = in_channels + skip_channels + freq_channels
        
        self.conv = nn.Sequential(
            nn.Conv2d(conv_in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.use_lffi = use_lffi
        if use_lffi:
            self.lffi = CrossAttentionLFFI(out_channels)

    def forward(self, x, skip=None, freq_skip=None, text_embeds=None):
        # x: Low Res (B, C_in, H, W)
        x = self.up(x) # (B, C_in, 2H, 2W)
        
        to_cat = [x]
        
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
            to_cat.append(skip)
            
        if freq_skip is not None:
            if x.shape[-2:] != freq_skip.shape[-2:]:
                freq_skip = F.interpolate(freq_skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
            to_cat.append(freq_skip)
            
        x = torch.cat(to_cat, dim=1) 
        x = self.conv(x) 
        
        if self.use_lffi and text_embeds is not None:
            x = self.lffi(x, text_embeds)
            
        return x
