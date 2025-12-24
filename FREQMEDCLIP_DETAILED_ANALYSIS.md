# FreqMedCLIP Deep Dive - Kiến Trúc Dual-Branch

## 1. Overall Architecture

```
Input Image (224x224)
    ↓
    ├─────────────────────────────────────┬─────────────────────────────
    ↓ ViT Branch                         ↓ Frequency Branch
    
[BiomedCLIP ViT-B/32]                 [Laplacian High-Freq Extraction]
    ↓                                        ↓
[Multi-layer hidden states]              [FrequencyEncoder]
    ↓                                        ↓
[FPNAdapter]                            [Multi-scale freq features]
    ↓                                        ↓
[4 scales: 768,384,192,96]             [Parallel feature extraction]
    ↓                                        ↓
    └─────────────────────────────────────┬─────────────────────────────
                    ↓
            [FFBI: Bidirectional Fusion]  ← Bottleneck interaction
                    ↓
        ┌───────────────────────────┬───────────────────────────┐
        ↓                           ↓
    [Decoder Chain 1]          [Decoder Chain 2]
    ViT Branch                 Freq Branch
        ↓                           ↓
    [out_1]                     [out_2]
    (Segmentation Mask)        (Segmentation Mask)
        ↓                           ↓
        └───────────────────────────┘
                ↓
            [Ensemble]
            Final mask = (out_1 + out_2) / 2
```

---

## 2. Component Breakdown

### 2.1 ViT Branch (BiomedCLIP)

**File**: `train_freq_fusion.py` lines 207-240

```python
# 1. Extract hidden states from multiple layers
vision_outputs = self.biomedclip.vision_model(pixel_values, output_hidden_states=True)
hidden_states = vision_outputs.hidden_states  # List of 13 tensors (0-12)

# 2. Select specific layers [12, 10, 7, 4] (last to early layers)
# This captures multi-scale hierarchical features
layers_idx = [12, 10, 7, 4]
fpn_inputs = [hidden_states[idx][:, 1:, :] for idx in layers_idx]
# Skip CLS token (index 0), shape: (B, 196, 768) -> 14x14 spatial

# 3. FPNAdapter converts to Feature Pyramid
fpn_feats = self.fpn_adapter(fpn_inputs)
# Output: [s1(14x14, 768), s2(28x28, 384), s3(56x56, 192), s4(112x112, 96)]
```

**Why these layers?**
- Layer 12: Latest/deepest features (global context)
- Layer 10: Intermediate
- Layer 7: Intermediate
- Layer 4: Early layers (local details)

### 2.2 FPNAdapter (New Component)

**File**: `scripts/freq_components.py` lines 218-276

```python
class FPNAdapter(nn.Module):
    """
    Converts ViT's isotropic 14x14 features from different layers
    into a proper Feature Pyramid with multiple scales.
    """
    
    def forward(self, features):
        # Input: all [B, C, 14, 14]
        x12, x9, x6, x3 = features
        
        s1 = self.scale1_conv(x12)      # 14x14 (bottleneck)
        s2 = self.scale2_up(x9)         # 28x28 (upsample 2x)
        s3 = self.scale3_up(x6)         # 56x56 (upsample 4x)
        s4 = self.scale4_up(x3)         # 112x112 (upsample 8x)
        
        return [s1, s2, s3, s4]
```

**Key insight**: ViT features are all the same spatial resolution (14x14), so FPNAdapter upsamples different layers to create a pyramid.

---

### 2.3 Frequency Branch

**File**: `train_freq_fusion.py` lines 241-249

#### A. High-Frequency Extraction
```python
def get_high_freq_image(self, pixel_values):
    # Laplacian filter (edge detector)
    kernel = torch.tensor([[-1, -1, -1], 
                           [-1,  8, -1], 
                           [-1, -1, -1]], dtype=torch.float32)
    high_freq = F.conv2d(pixel_values, kernel, padding=1, groups=3)
    return high_freq  # (B, 3, 224, 224)
```

**Why Laplacian?**
- Captures edges and high-frequency details
- Complements ViT's global/semantic features with local edge info

#### B. FrequencyEncoder

**File**: `scripts/freq_components.py` lines 315-369

```python
class FrequencyEncoder(nn.Module):
    """
    Processes high-freq image through 3 layers with stride=2 downsampling
    + Text-guided SE blocks for modulation.
    """
    
    def forward(self, x, text_embeds=None):
        # x: (B, 3, 224, 224) - or (B, 9, 112, 112) if DWT
        
        f1 = self.layer1(x)          # 112x112, base_channels (64)
        if text_embeds: f1 = self.se1(f1, text_embeds)
        
        f2 = self.layer2(f1)         # 56x56, base_channels*2 (128)
        if text_embeds: f2 = self.se2(f2, text_embeds)
        
        f3 = self.layer3(f2)         # 28x28, base_channels*4 (256)
        if text_embeds: f3 = self.se3(f3, text_embeds)
        
        return [f3, f2, f1]  # [28x28, 56x56, 112x112]
```

**Text-Guided SE Block:**
```python
class TextGuidedSEBlock(nn.Module):
    def forward(self, x, text_embeds):
        # 1. Global average pooling on visual features
        y = self.avg_pool(x)  # (B, C) -> (B, C, 1, 1)
        
        # 2. Project text embeddings to channel space
        t_proj = self.text_proj(text_embeds.mean(dim=1))  # (B, 768) -> (B, C)
        
        # 3. Combine visual and text info
        y = y + t_proj
        
        # 4. Generate channel weights via FC layers
        weights = self.fc(y)  # (B, C, 1, 1)
        
        # 5. Scale channels
        return x * weights
```

**Key point**: Text embeddings **modulate** (scale) frequency features at each level.

---

### 2.4 FFBI - Frequency Fusion Branch Interaction

**File**: `scripts/fmiseg_components.py` lines 150-177

```python
class FFBI(nn.Module):
    """
    Bidirectional cross-attention fusion at bottleneck (14x14, 768D)
    """
    def forward(self, x, y):
        # x: ViT features (B, 196, 768)
        # y: Freq features (B, 196, 768) after projection
        
        # High-freq attends to low-freq (ViT)
        x1, _ = self.cross_attnH(query=x, key=y, value=y)
        x_fused = x1 + x  # Residual
        
        # Low-freq attends to high-freq (ViT)
        y1, _ = self.cross_attnL(query=y, key=x, value=x)
        y_fused = y1 + y  # Residual
        
        return x_fused, y_fused
```

**Flow:**
```
        ViT (x)          Freq (y)
         ↓                 ↓
    Self-norm         Self-norm
         ↓                 ↓
    x queries y        y queries x
    x attends to y  ←→  y attends to x
         ↓                 ↓
        x1+x              y1+y
         ↓                 ↓
      x_fused         y_fused
```

**Why bi-directional?**
- Each branch learns complementary info
- ViT learns from freq edges
- Freq learns from ViT global context
- Information flows both ways

---

### 2.5 Decoder Chain (Dual-Branch)

**File**: `train_freq_fusion.py` lines 250-310

#### Structure:
```
Branch 1 (ViT):          Branch 2 (Freq):
decoder16 (768→384)      decoder16_2
decoder8  (384→192)      decoder8_2
decoder4  (192→96)       decoder4_2
decoder1  (96→24)        decoder1_2
out       (24→1)         out_2 (24→1)
```

#### Key Innovation: Cross-Injection at decoder4
```python
# Use high-res freq features (f0, 112x112, 96ch) as skip for ViT branch
cnn_high_res_skip = image_features2[3]  # f0 from Freq branch
os4, _ = self.decoder4(os8, cnn_high_res_skip, text_embeds)
# ViT branch gets detailed edge info from Freq branch!
```

### 2.6 LFFI - Language-guided Feature Fusion Integration

**File**: `scripts/freq_components.py` lines 160-208 & `scripts/fmiseg_components.py` lines 81-130

Used in each Decoder level:

```python
class Decoder(nn.Module):
    def forward(self, vis, skip_vis, txt):
        # vis: current level features (B, L, C)
        # skip_vis: skip connection from encoder
        # txt: text embeddings (B, 77, 768)
        
        vis = self.lffi_layer(vis, txt)  # Apply LFFI
        # LFFI = Self-Augment + Bidirectional Cross-Attention + FF
        
        vis = rearrange(vis, 'B (H W) C -> B C H W', H=self.spatial_size, W=...)
        skip_vis = rearrange(skip_vis, 'B (H W) C -> B C H W', H=...)
        
        output = self.decoder(vis, skip_vis)  # UnetrUpBlock
        output = rearrange(output, 'B C H W -> B (H W) C')
        
        return output
```

**LFFI Process:**
```
Visual Features (vis)
    ↓
Self-Augment (self-attn on vis)
    ↓
Bidirectional Cross-Attention:
    - vis queries text → vis_to_txt
    - text queries vis → txt_to_vis
    ↓
Feed-Forward + Norms
    ↓
Interaction Matrix: vis_to_txt @ txt_to_vis.T
    ↓
Final: vis * scale * (vis_to_txt + Linear(interaction))
```

**Why at every decoder level?**
- Guides feature refinement using text at all scales
- E.g., "brain tumor" description helps at 56x56, 28x28, 14x14 levels

---

## 3. Data Flow Example - Chi Tiết Text Truyền Vào

**Input**: Image (224x224), Text ("brain tumor")

### Step 0: Text Encoding (TRƯỚC KHI VÀO HAI NHÁNH)
```python
# Dòng 223 trong train_freq_fusion.py
text_outputs = self.biomedclip.text_model(input_ids, output_hidden_states=True)
text_embeds = text_outputs[0]  # Shape: (B, 77, 768)
# BiomedCLIP's BERT tokenizer: 77 tokens, 768 dimension
```

**⭐ ĐẦY LÀ ĐIỂM QUAN TRỌNG:**
- Text được encode **1 lần duy nhất** từ input_ids
- Text embeddings (B, 77, 768) sau đó được **truyền vào CẢ HAI nhánh**
- NOT separate text encoding cho mỗi nhánh

---

### Step 1: ViT Branch (Nhánh ảnh thường)

**File**: `train_freq_fusion.py` lines 207-220

```python
# 1A. Extract visual features from ViT layers
vision_outputs = self.biomedclip.vision_model(pixel_values, output_hidden_states=True)
hidden_states = vision_outputs.hidden_states  # 13 layers, each (B, 197, 768)

# 1B. Select multi-scale layers [12, 10, 7, 4]
layers_idx = [12, 10, 7, 4]
fpn_inputs = [hidden_states[idx][:, 1:, :] for idx in layers_idx]
# Skip CLS token → shape (B, 196, 768) → reshape to (B, 768, 14, 14)

# 1C. Convert to pyramid via FPNAdapter
fpn_feats = self.fpn_adapter(fpn_inputs)
image_features = [s1(14x14,768), s2(28x28,384), s3(56x56,192), s4(112x112,96)]
```

**⚠️ CHÚ Ý:** Nhánh ViT này **KHÔNG dùng text ở encoder**, chỉ dùng image!

---

### Step 2: Frequency Branch (Nhánh high-frequency)

**File**: `train_freq_fusion.py` lines 221-239

```python
# 2A. Extract high-frequency from original image
img_h = self.get_high_freq_image(pixel_values)  # Laplacian filter
# Output: (B, 3, 224, 224) - edge map

# 2B. *** TEXT ENTERS HERE (Nhánh 2) ***
# PASS TEXT EMBEDDINGS TO FREQUENCY ENCODER
freq_feats = self.freq_encoder(img_h, text_embeds=text_embeds)
#                                         ↑↑↑↑↑↑↑↑↑↑↑↑
# text_embeds: (B, 77, 768)
```

**Bên trong FrequencyEncoder:**
```python
# scripts/freq_components.py lines 340-360

class FrequencyEncoder(nn.Module):
    def forward(self, x, text_embeds=None):
        # x: high-freq image (B, 3, 224, 224)
        
        f1 = self.layer1(x)  # 112x112, 64 channels
        if text_embeds is not None:
            f1 = self.se1(f1, text_embeds)  # *** Text-guided SE Block ***
            # SE Block: pooled_features + text_projection → channel weights
        
        f2 = self.layer2(f1)  # 56x56, 128 channels
        if text_embeds is not None:
            f2 = self.se2(f2, text_embeds)  # *** Again! ***
        
        f3 = self.layer3(f2)  # 28x28, 256 channels
        if text_embeds is not None:
            f3 = self.se3(f3, text_embeds)  # *** Again! ***
        
        return [f3, f2, f1]
```

**TextGuidedSEBlock hoạt động như thế nào:**
```
Input: f (image features)        text_embeds (B, 77, 768)
         ↓                              ↓
    AdaptiveAvgPool2d          Linear(768→C) + pool over seq
    (B, C, 1, 1)                    ↓
         ↓                     (B, C) text projection
    ↓                              ↓
    ──────────────────→ Addition ←──────────────
         (channel-wise add)
              ↓
         FC layers → Sigmoid
              ↓
         weights (B, C, 1, 1)
              ↓
         f * weights  ← Scale channels by text!
```

**⭐ KEY POINT:** Text modulates (scales) mỗi channel của frequency features!

---

### Step 3: FFBI Fusion at Bottleneck

**File**: `train_freq_fusion.py` lines 241-249

```python
# 3A. Extract bottleneck features từ cả 2 nhánh
os32 = image_features[0]      # ViT bottleneck (14x14, 768)
os32_2 = image_features2[0]   # Freq bottleneck (14x14, 768)

# 3B. Flatten để dùng attention
os32_flat = rearrange(os32, 'b c h w -> b (h w) c')      # (B, 196, 768)
os32_2_flat = rearrange(os32_2, 'b c h w -> b (h w) c')  # (B, 196, 768)

# 3C. FFBI: Bidirectional fusion (NO TEXT HERE)
fu32_flat, fu32_2_flat = self.ffbi(os32_flat, os32_2_flat)
#         ViT attends to Freq  ⟷  Freq attends to ViT
#         (information exchange without text)
```

**⚠️ FFBI KHÔNG dùng text**, chỉ là cross-attention giữa 2 nhánh!

---

### Step 4: Dual Decoder Chains (TEXT USED AGAIN HERE)

**File**: `train_freq_fusion.py` lines 251-283

```python
# *** TEXT IS PASSED TO EVERY DECODER LEVEL ***

# Decoder 16 (14→28)
os16, _ = self.decoder16(fu32_flat, skips[1], text_embeds)
#                                              ↑↑↑↑↑↑↑↑
# BRANCH 1 (ViT): receives text

os16_2, _ = self.decoder16_2(fu32_2_flat, skips2[1], text_embeds)
#                                                     ↑↑↑↑↑↑↑↑
# BRANCH 2 (Freq): receives SAME text

# Decoder 8 (28→56)
os8, _ = self.decoder8(os16, skips[2], text_embeds)
os8_2, _ = self.decoder8_2(os16_2, skips2[2], text_embeds)

# Decoder 4 (56→112)
# ⭐ CROSS-INJECTION: Freq high-res → ViT branch
cnn_high_res_skip = image_features2[3]  # 112x112, freq features
cnn_high_res_skip_flat = rearrange(cnn_high_res_skip, 'b c h w -> b (h w) c')

os4, _ = self.decoder4(os8, cnn_high_res_skip_flat, text_embeds)
#                                                     ↑↑↑↑↑↑↑↑
# Branch 1 (ViT) gets freq skip + text

os4_2, _ = self.decoder4_2(os8_2, skips2[3], text_embeds)
#                                             ↑↑↑↑↑↑↑↑
# Branch 2 (Freq) also gets text
```

**Bên trong Decoder class:**

```python
# scripts/fmiseg_components.py

class Decoder(nn.Module):
    def forward(self, vis, skip_vis, txt):
        # vis: current level features (B, L, C)
        # skip_vis: skip connection
        # txt: text embeddings (B, 77, 768) ← SAME FOR BOTH BRANCHES!
        
        # *** LFFI Module: Uses text to guide visual features ***
        vis = self.lffi_layer(vis, txt)
        # LFFI = Self-Augment + Bidirectional Cross-Attention with text + FF
        
        # Then: reshape, concat with skip, upsample
        vis = rearrange(vis, 'B (H W) C -> B C H W', ...)
        skip_vis = rearrange(skip_vis, 'B (H W) C -> B C H W', ...)
        
        output = self.decoder(vis, skip_vis)  # UnetrUpBlock
        output = rearrange(output, 'B C H W -> B (H W) C')
        
        return output, ...
```

---

## 4. Complete Text Flow Summary

```
        Input Text (input_ids)
              ↓
    BiomedCLIP Text Encoder
              ↓
        text_embeds (B, 77, 768)
              ↓
    ┌─────────┴──────────┐
    ↓                    ↓
[Frequency Encoder]   [Decoder Chains]
    ↓                    ↓
  Text guides         Text guides
  channel weighting   cross-attention
  at each layer       at decoder levels
    ↓                    ↓
  f1,f2,f3          Refined features
  (freq features)   (both branches)
    ↓                    ↓
    └─────────┬──────────┘
              ↓
        Final Mask Output
```

---

## 5. Visual Architecture with Text Flow

```
                       Input
                    ┌──┴──┐
                    ↓     ↓
            [Image] [Text (input_ids)]
                    |     |
                    |     └─→ BiomedCLIP.text_model
                    |           ↓
                    |     text_embeds (B, 77, 768)
                    |           │
        ┌───────────┴───────────┼───────────┬──────────┐
        ↓                       ↓           ↓          ↓
    [ViT Branch]           [FreqEnc]   [Decoder16]  [Decoder8]
    (NO text here)      (Text via SE)  (Text via    (Text via
    BiomedCLIP.vision        |         LFFI)        LFFI)
         ↓                    ↓         |             |
    [FPNAdapter]         [f1,f2,f3]    ↓             ↓
         ↓                    ↓      [Decoder4]  [Multiple]
    [4 scales]          ┌─────┴──────┐ |             |
         ↓              ↓            ↓ ↓             ↓
    [FFBI Fusion] ←────→ [image_features2]  [text guides]
         ↓              ↓            ↓ ↓             ↓
    [Decoders]  ←──────────→ [Cross-Inject]      [Final]
         ↓              ↓            ↓              ↓
    [Output1]      [Output2]  [Ensemble]    [Sigmoid→Mask]
```

---

## 6. Text Usage Summary Table

| Component | Text Input? | How Used | Both Branches? |
|-----------|-------------|----------|----------------|
| **ViT Encoder** | ❌ No | - | - |
| **FrequencyEncoder** | ✅ Yes | Text-guided SE blocks (channel modulation) | ViT: No, Freq: Yes |
| **FFBI (Bottleneck)** | ❌ No | Direct cross-attention (no text) | N/A |
| **Decoder16** | ✅ Yes | LFFI (language-guided fusion) | ✅ Both |
| **Decoder8** | ✅ Yes | LFFI | ✅ Both |
| **Decoder4** | ✅ Yes | LFFI | ✅ Both |
| **Decoder1** | ❌ No | Simple upsampling | - |

---

## 7. Code Trace (Line by Line)

```
train_freq_fusion.py:223 → text_embeds = self.biomedclip.text_model(input_ids)
                                        # (B, 77, 768) - CREATED ONCE

train_freq_fusion.py:239 → freq_feats = self.freq_encoder(img_h, text_embeds=text_embeds)
                                        # Text → Frequency branch

train_freq_fusion.py:276 → os16, _ = self.decoder16(fu32_flat, skips[1], text_embeds)
train_freq_fusion.py:277 → os16_2, _ = self.decoder16_2(fu32_2_flat, skips2[1], text_embeds)
                                        # Same text → Both decoders

train_freq_fusion.py:279 → os8, _ = self.decoder8(os16, skips[2], text_embeds)
train_freq_fusion.py:280 → os8_2, _ = self.decoder8_2(os16_2, skips2[2], text_embeds)
                                        # Same text → Both decoders

train_freq_fusion.py:287 → os4, _ = self.decoder4(os8, cnn_high_res_skip_flat, text_embeds)
train_freq_fusion.py:288 → os4_2, _ = self.decoder4_2(os8_2, skips2[3], text_embeds)
                                        # Same text → Both decoders

Result: text_embeds (B, 77, 768) used at EVERY decoder level for BOTH branches!
```

---

## 4. Training Details

### 4.1 Loss Functions
**File**: `train_freq_fusion.py` lines 311-360

```python
# Dice Loss for segmentation
dice_loss = DiceLoss(smooth=1.0)

# HardNegativeLoss for contrastive learning
hnl = HardNegativeLoss(...)

# Combined
loss_seg1 = dice_loss(pred1, mask)
loss_seg2 = dice_loss(pred2, mask)
loss_seg = loss_seg1 + loss_seg2

loss_contrastive = hnl(feat1_pooled, feat2_pooled, text_embed, ...)

total_loss = loss_seg + 0.1 * loss_contrastive
```

### 4.2 Optimizer Setup
```python
# Different learning rates for different components
optimizer = torch.optim.AdamW([
    {'params': freq_encoder.parameters(), 'lr': 1e-4},
    {'params': fpn_adapter.parameters(), 'lr': 1e-4},
    {'params': decoders.parameters(), 'lr': 1e-4},
    {'params': biomedclip.parameters(), 'lr': 1e-5}  # Lower for frozen backbone
])
```

### 4.3 Training Loop
```python
for epoch in range(epochs):
    for batch in train_loader:
        # Forward
        pred1, pred2 = model(pixel_values, input_ids)
        
        # Loss
        loss = dice_loss(pred1, mask) + dice_loss(pred2, mask) + contrastive_loss
        
        # Backward
        loss.backward()
        optimizer.step()
        
    # Validation
    for batch in val_loader:
        pred1, pred2 = model(...)
        metrics = compute_metrics(pred1, pred2, mask)
        
    # Save checkpoint
    if metrics['dice'] > best_dice:
        save_checkpoint(model)
```

---

## 5. Results So Far

### Brain Tumors Test Set (600 samples)
- **Checkpoint**: fusion_brain_tumors_epoch145
- **Dice**: 0.8446 ± 0.1550
- **IoU**: 0.7550 ± 0.1833
- **Precision**: 0.8867 ± 0.1536
- **Recall**: 0.8373 ± 0.1874

### Breast Tumors Test Set (113 samples)
- **Checkpoint**: fusion_breast_tumors_epoch106
- **Dice**: 0.7937 ± 0.2140
- **IoU**: 0.6981 ± 0.2301
- **Precision**: 0.7633 ± 0.2479
- **Recall**: 0.8878 ± 0.1747

---

## 6. Key Innovations vs FMISeg

| Aspect | FMISeg | FreqMedCLIP |
|--------|--------|------------|
| **Vision Backbone** | ConvNeXt-tiny | BiomedCLIP ViT-B/32 |
| **Freq Separation** | Fixed DWT preprocessing | Learnable Laplacian filter |
| **Multi-scale** | Custom pyramid | FPNAdapter (learnable upsampling) |
| **Text Guidance** | Local-only (decoders) | Global (text-guided SE blocks) |
| **Freq Encoder** | N/A | New learnable module |
| **Cross-Injection** | At bottleneck only | At multiple decoder levels |
| **Decoder Skip** | Simple concat | LFFI fusion at each level |

---

## 7. Execution Flow for Training

```bash
# Train single dataset
python train_freq_fusion.py --dataset brain_tumors --epochs 100 --batch-size 4

# Train both datasets sequentially
python batch_train_and_eval.py
# Creates: results_{timestamp}/
#   ├── brain_tumors_checkpoints/
#   ├── brain_tumors_eval/visualizations/
#   ├── breast_tumors_checkpoints/
#   ├── breast_tumors_eval/visualizations/
#   └── SUMMARY_REPORT.txt

# Resume training from checkpoint
python resume_training.py
# Loads: checkpoints/results_20251205_011222/{dataset}_checkpoints/*.pth
# Outputs: checkpoints/results_resume_{timestamp}/
```

---

## 8. Visualization Pipeline

**File**: `evaluate_freqmedclip.py`

For each test sample, saves a 3×4 grid showing:
```
Row 1: [Input Image] [GT Mask] [ViT Pred] [Freq Pred]
Row 2: [Freq Feature] [Freq Feature] [Freq Feature] [Overlay GT vs Pred]
Row 3: [FPN Scale 1] [FPN Scale 2] [FPN Scale 3] [Final Binary Mask]
```

---

## 9. Next Steps & Questions

1. **Pseudo-label Generation for Weak SSL**:
   - Use `evaluate_freqmedclip.py` to generate predictions
   - Apply `postprocess_freqmedclip_outputs.py` for cleaning
   - Convert to nnUNet format for weak supervision

2. **Architecture Improvements**:
   - Adjust FPNAdapter channel dims
   - Tune text-guided SE block scales
   - Experiment with different fusion strategies

3. **Data Strategy**:
   - Current: train/val/test split
   - Consider: k-fold cross-validation
   - Planned: weak SSL with generated pseudo-labels
