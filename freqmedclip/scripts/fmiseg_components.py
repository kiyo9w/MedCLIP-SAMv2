import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
import torch.nn.functional as F
from monai.networks.blocks.unetr_block import UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample

# --- From FMISeg-original/net/decoder.py ---

class SelfAugment(nn.Module):
    def __init__(self, in_channels):
        super(SelfAugment, self).__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.vis_pos = PositionalEncoding(in_channels)
        self.self_attn = nn.MultiheadAttention(embed_dim=in_channels,num_heads=1,batch_first=True)
        self.self_attn_norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        vis = self.norm(x)
        q = k = self.vis_pos(vis)
        vis = self.self_attn(q, k, value=vis)[0]
        vis = self.self_attn_norm(vis)
        vis = x + vis
        return vis

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
        x = x + nn.Parameter(self.pe[:, :x.size(1)],requires_grad=False) #size = [batch, L, d_model]
        return self.dropout(x) # size = [batch, L, d_model]

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

class LFFI(nn.Module):
    def __init__(self, in_channels:int, output_text_len:int, input_text_len:int=77, embed_dim:int=768):
        super(LFFI, self).__init__()
        self.in_channels = in_channels
        self.augment = SelfAugment(in_channels)
        self.cross_attn_norm = nn.LayerNorm(in_channels)
        
        # Cross Attention layers
        self.cross_attn1 = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, batch_first=True)
        self.cross_attn2 = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, batch_first=True)
        
        # Projects CLIP embedding (768) to channel dim (e.g. 384/192/96)
        self.text_project = nn.Sequential(
            nn.Linear(embed_dim, in_channels),
            nn.LeakyReLU(),
        )
        
        self.vis_pos = PositionalEncoding(in_channels)
        self.txt_pos = PositionalEncoding(in_channels, max_len=output_text_len)
        
        # Norms
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.norm3 = nn.LayerNorm(in_channels)
        self.norm4 = nn.LayerNorm(in_channels)
        self.norm5 = nn.LayerNorm(in_channels)
        
        # Feed Forward
        self.fl1 = FeedLinear(in_channels, in_channels*2)
        self.fl2 = FeedLinear(in_channels, in_channels*2)
        
        self.line = nn.Linear(output_text_len, in_channels)
        
        # GATING MECHANISM
        # Eq 6: Conv(F + F' * Sigmoid(Linear(F_M)))
        # We use a Linear gate instead of Conv for token compatibility
        self.gate_layer = nn.Linear(in_channels, 1) 
        self.final_conv = nn.Linear(in_channels, in_channels) # Equivalent to "Conv" in Eq 6

    def forward(self, x, txt):
        '''
        x: [B, (HW), C] - Visual tokens
        txt: [B, L, D] - Text embeddings (768 dim)
        '''
        # 1. Project Text to match Visual Channels
        # txt is [B, 77, 768], we need [B, 77, C]
        txt_proj = self.text_project(txt) 
        
        # 2. Self Augment Visual
        vis = self.augment(x)
        vis2 = self.norm1(vis)
        
        # 3. Cross Attention 1: Visual queries Text
        vis2_v, _ = self.cross_attn1(query=self.vis_pos(vis2),
                                   key=self.txt_pos(txt_proj),
                                   value=txt_proj)  
        
        # 4. Cross Attention 2: Text queries Visual
        vis2_l, _ = self.cross_attn2(query=self.txt_pos(txt_proj),
                                   key=self.vis_pos(vis2),
                                   value=vis2)
                                   
        vis2_v = self.norm2(vis2_v + vis2)
        vis2_l = self.norm3(vis2_l + txt_proj)
        
        vis2_v = self.norm4(self.fl1(vis2_v) + vis2_v)
        vis2_l = self.norm5(self.fl2(vis2_l) + vis2_l)
        
        # 5. Interaction (Matrix Multiplication)
        # [B, HW, C] x [B, C, L] -> [B, HW, L]
        interaction = torch.matmul(vis2_v, vis2_l.transpose(1, 2))
        
        # Project back to Channel dim: [B, HW, L] -> [B, HW, C]
        F_prime = self.line(interaction)
        
        # 6. GATING
        # Calculate Gate: Sigmoid(Linear(F_prime))
        gate = torch.sigmoid(self.gate_layer(F_prime))
        
        # Apply Gate: F' * Gate
        gated_features = F_prime * gate
        
        # 7. Additive Residual (Eq 6 structure)
        # F_out = Conv(F + Gated_F')
        out = self.final_conv(vis + gated_features)
        
        out = self.cross_attn_norm(out)
        
        # Return tuple as expected by Decoder
        return out, txt

class Decoder(nn.Module):
    def __init__(self,in_channels, out_channels, spatial_size, text_len, embed_dim=768) -> None:
        super().__init__()
        self.lffi_layer = LFFI(in_channels,text_len, embed_dim=embed_dim)  
        self.spatial_size = spatial_size
        self.decoder = UnetrUpBlock(2,in_channels,out_channels,3,2,norm_name='BATCH')

    def forward(self, vis, skip_vis, txt):
        if txt is not None:
            vis, txt =  self.lffi_layer(vis, txt)
        # Infer spatial dimensions from token length to avoid mismatches
        B, L, C = vis.shape
        H = int(L ** 0.5)
        W = H
        if H * W != L:
            raise ValueError(f"Cannot reshape tokens of length {L} into square spatial map (H*W).")
        vis = rearrange(vis, 'B (H W) C -> B C H W', H=H, W=W)
        # skip_vis expected to be 2x spatial resolution
        skip_L = skip_vis.shape[1]
        skip_H = H * 2
        skip_W = skip_H
        if skip_H * skip_W != skip_L:
            # Fallback: try to infer skip spatial size directly
            skip_H = int(skip_L ** 0.5)
            skip_W = skip_H
            if skip_H * skip_W != skip_L:
                raise ValueError(f"Cannot reshape skip tokens of length {skip_L} into square spatial map (H*W).")
        skip_vis = rearrange(skip_vis, 'B (H W) C -> B C H W', H=skip_H, W=skip_W)
        output = self.decoder(vis,skip_vis)
        output = rearrange(output,'B C H W -> B (H W) C')
        return output, txt

# --- From FMISeg-original/net/model.py ---

class FFBI(nn.Module):
    def __init__(self, dim, num,batchf):
        super(FFBI, self).__init__()
        self.cross_attnh = nn.MultiheadAttention(embed_dim=dim,num_heads=num,batch_first=batchf)
        self.cross_attnl = nn.MultiheadAttention(embed_dim=dim,num_heads=num,batch_first=batchf)

    def forward(self, x,y):
        x1, _=self.cross_attnl(query=x,key=y,value=y)
        x2 = x1 + x
        y1, _ = self.cross_attnh(query=y,key=x,value=x)
        y2 = y1+ y
        return x2,y2


# --- Smart Single-Stream Components (FreqMedCLIP Refactor) ---

class SemanticAnchor(nn.Module):
    """
    Feed-forward approximation of M2IB (Multi-modal Information Bottleneck) for training.
    
    The original M2IB in MedCLIP-SAMv2 (iba.py) uses iterative optimization per-sample at inference.
    This module provides a differentiable, feed-forward alternative for end-to-end training.
    
    It computes a semantic attention map by:
    1. Computing similarity between visual features and text CLS token
    2. Generating a learnable gate from the similarity map
    
    CRITICAL: Input MUST be 768-dim hidden_states, NOT 512-dim pooler_output.
    """
    def __init__(self, dim=768):
        super().__init__()
        self.gate_gen = nn.Sequential(
            nn.Conv2d(1, dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, visual_features, text_cls):
        """
        Args:
            visual_features: (B, 768, H, W) - Deep semantic features from ViT Layer 11
            text_cls: (B, 768) - Text CLS token from hidden_states (NOT pooler_output!)
        
        Returns:
            anchor_map: (B, 1, H, W) - Semantic attention mask ("where the text says the object is")
        """
        # Broadcast text to spatial dimensions: (B, 768) -> (B, 768, 1, 1)
        text_spatial = text_cls.unsqueeze(-1).unsqueeze(-1)
        
        # Compute similarity: element-wise product then sum across channels
        # This highlights spatial locations where visual features align with text
        similarity = (visual_features * text_spatial).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Normalize similarity for stable gate generation
        similarity = similarity / (similarity.abs().max() + 1e-8)
        
        # Generate learnable semantic anchor map
        anchor_map = self.gate_gen(similarity)
        
        return anchor_map


class SmartSpatialGate(nn.Module):
    """
    Gates high-frequency detail features using the semantic anchor map.
    
    Implements the "Semantic-First, Boundary-Second" philosophy:
    - Only edges/textures INSIDE the semantic region are kept
    - Irrelevant edges (e.g., bones when looking for tumors) are suppressed
    
    Includes a learnable residual weight to prevent complete signal suppression.
    """
    def __init__(self):
        super().__init__()
        # Learnable residual weight to prevent complete signal loss
        # Initialized to 0.1 to allow some ungated features through
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, anchor_map, high_freq_features):
        """
        Args:
            anchor_map: (B, 1, H_low, W_low) - Semantic anchor from SemanticAnchor module
            high_freq_features: (B, C, H_high, W_high) - Enhanced HF features (Layer 3 + wavelet)
        
        Returns:
            gated_features: (B, C, H_high, W_high) - Semantically gated HF features
        """
        # Upsample anchor to match high-frequency feature resolution
        anchor_up = F.interpolate(
            anchor_map, 
            size=high_freq_features.shape[-2:],
            mode='bilinear', 
            align_corners=False
        )
        
        # Apply gate with residual connection
        # The residual allows some ungated information through for training stability
        gated_features = high_freq_features * anchor_up + high_freq_features * self.residual_weight
        
        return gated_features