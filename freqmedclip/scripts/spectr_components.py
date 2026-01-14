
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ChannelSpectralGate(nn.Module):
    """
    Module B: Channel Spectral Gating
    Source: Inspired by FMISeg's LFFI but applied to the spectral domain.
    Logic: Learning a channel-wise weight from text embeddings to gate DWT frequency channels.
    """
    def __init__(self, in_channels=12, text_dim=768):
        super().__init__()
        self.text_dim = text_dim
        self.in_channels = in_channels
        
        # Simple MLP to map text embedding to channel weights
        self.gate_mlp = nn.Sequential(
            nn.Linear(text_dim, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, dwt_features, text_embeddings):
        """
        dwt_features: (B, 12, H, W)
        text_embeddings: (B, L, C) or (B, C) - we will use the [CLS] token usually
        """
        # Assuming text_embeddings is (B, C) or we take the first token
        if len(text_embeddings.shape) == 3:
            text_cls = text_embeddings[:, 0, :] # Use CLS token
        else:
            text_cls = text_embeddings

        # Compute channel weights: (B, 12)
        channel_weights = self.gate_mlp(text_cls)
        
        # Reshape for broadcasting: (B, 12, 1, 1)
        channel_weights = channel_weights.unsqueeze(-1).unsqueeze(-1)
        
        # Apply gating
        gated_dwt = dwt_features * channel_weights
        
        return gated_dwt, channel_weights


class EdgeHead(nn.Module):
    """
    Module D: Explicit Edge Supervision
    Source: FMISeg / ESPNet concepts.
    Logic: Simple Conv2d stack on HF features to predict binary edge map.
    """
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.conv_block(x)


class MorphologicalEdgeTarget(nn.Module):
    """
    Utility to compute edge targets from GT masks.
    Logic: Edge_GT = (Dilation(Mask_GT) - Erosion(Mask_GT))
    """
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, mask):
        """
        mask: (B, 1, H, W) binary mask
        """
        # Dilation: MaxPool
        dilated = self.pool(mask)
        
        # Erosion: -MaxPool(-X)
        eroded = -self.pool(-mask)
        
        edge_target = dilated - eroded
        return edge_target


# --- LFFI Components (Copied/Adapted from FMISeg) ---

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
    """
    Module C: Language-Guided Frequency Interaction (Injection)
    Source: FMISeg/net/decoder.py
    """
    def __init__(self, in_channels:int, output_text_len:int, input_text_len:int=77, embed_dim:int=768):
        super(LFFI, self).__init__()
        self.in_channels = in_channels
        self.augment = SelfAugment(in_channels)
        self.cross_attn_norm = nn.LayerNorm(in_channels)
        self.cross_attn1 = nn.MultiheadAttention(embed_dim=in_channels,num_heads=4,batch_first=True)
        self.cross_attn2 = nn.MultiheadAttention(embed_dim=in_channels,num_heads=4,batch_first=True)
        
        # Note: FMISeg uses input_text_len=24, but BiomedCLIP uses 77 usually.
        # The logic below projects input_text_len -> output_text_len.
        # We will use Conv1d for this projection as in FMISeg.
        self.text_project = nn.Sequential(
            nn.Conv1d(input_text_len,output_text_len,kernel_size=1,stride=1),
            nn.GELU(),
            nn.Linear(embed_dim,in_channels),
            nn.LeakyReLU(),
        )
        self.vis_pos = PositionalEncoding(in_channels)
        self.txt_pos = PositionalEncoding(in_channels,max_len=output_text_len)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.norm3 = nn.LayerNorm(in_channels)
        self.norm4 = nn.LayerNorm(in_channels)
        self.norm5 = nn.LayerNorm(in_channels)
        self.scale = nn.Parameter(torch.tensor(1.0),requires_grad=True)
        self.fl1=FeedLinear(in_channels,in_channels*2)
        self.fl2=FeedLinear(in_channels,in_channels*2)
        self.line=nn.Linear(output_text_len, in_channels)

    def forward(self, x, txt):
        '''
        x:[B N C1]
        txt:[B,L,C]
        '''
        # Project text: (B, L_in, C_emb) -> (B, L_out, C_hidden)
        txt = self.text_project(txt)
        
        vis = self.augment(x)
        vis2 = self.norm1(vis)
        
        # Dual Cross Attention
        # Image queries Text
        vis2_v,_ = self.cross_attn1(query=self.vis_pos(vis2),
                                   key=self.txt_pos(txt),
                                   value=txt)  
        # Text queries Image
        vis2_l,_ = self.cross_attn2(query=self.txt_pos(txt),
                                   key=self.vis_pos(vis2),
                                   value=vis2)
        
        vis2_v=self.norm2(vis2_v+vis2)
        vis2_l=self.norm3(vis2_l+txt)
        vis2_v=self.norm4(self.fl1(vis2_v)+vis2_v)
        vis2_l=self.norm5(self.fl2(vis2_l)+vis2_l)
        
        # Gating interaction
        interaction = torch.matmul(vis2_v,vis2_l.transpose(1, 2)) # (B, N, L_out)
        vis2=vis2_v+self.line(interaction)
        
        vis2 = self.cross_attn_norm(vis2)
        vis = vis*self.scale*vis2
        return vis


# --- Recoupling Loss Logic ---

class RecouplingLoss(nn.Module):
    """
    Mandatory Recoupling Loss.
    Source: Logic from RecLMIS.cond_cons_loss
    Constraint: Uses High-Res ROI Pooling.
    """
    def __init__(self, in_channels=24, text_dim=768, logit_scale_init=2.6592, loss_weight=1.0):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)
        self.loss_weight = loss_weight
        self.cross_entropy = nn.CrossEntropyLoss()
        
        # Project visual features to text dimension if needed
        if in_channels != text_dim:
            self.projector = nn.Linear(in_channels, text_dim)
        else:
            self.projector = nn.Identity()

    def forward(self, fused_features, mask_logits, text_features):
        """
        fused_features: (B, C, H, W) - High resolution features from Decoder
        mask_logits: (B, 1, H, W) - Predicted masks (logits)
        text_features: (B, 1, C) or (B, C) - Text embeddings (CLS token)
        """
        if len(text_features.shape) == 2:
            text_features = text_features.unsqueeze(1) # (B, 1, C)
        
        B, C, H, W = fused_features.shape
        
        # 1. ROI Pooling (Soft)
        # Convert logits to prob: Sigmoid
        mask_prob = torch.sigmoid(mask_logits) # (B, 1, H, W)
        
        # Weighted sum of features
        # (B, C, H, W) * (B, 1, H, W) -> sum over H,W -> (B, C)
        numerator = (fused_features * mask_prob).sum(dim=(2, 3))
        denominator = mask_prob.sum(dim=(2, 3)) + 1e-6
        
        roi_features = numerator / denominator # (B, C)
        
        # Project to text dim
        roi_features = self.projector(roi_features)
        
        # Normalize features
        roi_features = roi_features / roi_features.norm(dim=-1, keepdim=True)
        text_cls = text_features.squeeze(1) # (B, C)
        text_cls = text_cls / text_cls.norm(dim=-1, keepdim=True)
        
        # 2. Alignment (Contrastive Loss)
        # Similar to CLIP loss: (B, B) matrix of similarities
        # roi_features (B, C), text_cls (B, C)
        
        # Logits: (B, B)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * roi_features @ text_cls.t()
        logits_per_text = logit_scale * text_cls @ roi_features.t()
        
        labels = torch.arange(B, device=fused_features.device)
        
        loss_i = self.cross_entropy(logits_per_image, labels)
        loss_t = self.cross_entropy(logits_per_text, labels)
        
        loss = (loss_i + loss_t) / 2
        return loss * self.loss_weight
