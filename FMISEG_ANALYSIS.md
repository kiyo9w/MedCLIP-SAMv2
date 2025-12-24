# FMISeg Analysis - Dual Frequency Architecture

## Paper Overview
**Title**: Frequency-domain Multi-modal Fusion for Language-guided Medical Image Segmentation  
**Key Idea**: Sử dụng wavelets để phân tách ảnh thành high-frequency (edges, details) và low-frequency (shapes, structure), rồi xử lý song song với fusion ở giữa.

---

## 1. Core Architecture: Dual-Branch with FFBI (Frequency Fusion Branch Interaction)

### 1.1 Data Preprocessing (wave.py)
**Input**: Original image  
**Process**: Discrete Wavelet Transform (DWT) with Haar wavelet

```
Original Image
      ↓
    DWT2
      ↓
   LL (Low-Low)    LH, HL, HH (High-frequency components)
      ↓                         ↓
   Low-Freq           High-Freq (merge HH+HL+LH)
   Images_L           Images_H
```

**Output**:
- `Images_L`: Low-frequency (LL component) - smooth, structural info
- `Images_H`: High-frequency (LH+HL+HH merged) - edges, textures, details

### 1.2 Dual Vision Encoders (model.py)

```python
# Two parallel ConvNeXt encoders
self.encoder = VisionModel(vision_type, project_dim)   # Processes Images_H
self.encoder2 = VisionModel(vision_type, project_dim)  # Processes Images_L
```

**Each encoder**:
- Pretrained ConvNeXt-tiny-224
- Output: Multi-scale features [768, 384, 192, 96]
- 4 levels of hierarchical features

### 1.3 FFBI (Frequency Fusion Branch Interaction)

```python
class FFBI(nn.Module):
    def forward(self, x, y):  # x=high-freq features, y=low-freq features
        # High-freq attends to low-freq
        x1, _ = self.cross_attnH(query=x, key=y, value=y)
        x2 = x1 + x  # residual
        
        # Low-freq attends to high-freq  
        y1, _ = self.cross_attnL(query=y, key=x, value=x)
        y2 = y1 + y  # residual
        
        return x2, y2  # Fused features
```

**Key**: Bi-directional cross-attention fusion at bottleneck (32x32 resolution)

### 1.4 Parallel Decoders with LFFI (Language-guided Feature Fusion Integration)

```python
# Branch 1 (High-freq)
self.decoder16 → self.decoder8 → self.decoder4 → self.decoder1 → self.out

# Branch 2 (Low-freq)  
self.decoder16_2 → self.decoder8_2 → self.decoder4_2 → self.decoder1_2 → self.out_2
```

Each decoder incorporates:
- **LFFI module**: Cross-attention between visual features and text embeddings
- **Skip connections**: From encoder features
- **Upsampling**: Progressive 2x upsampling to full resolution

### 1.5 LFFI (Language-guided Feature Fusion Integration)

```python
class LFFI(nn.Module):
    def forward(self, vis, txt):
        # 1. Self-augmentation on visual features
        vis = self.augment(vis)
        
        # 2. Bidirectional cross-attention
        vis_to_txt = cross_attn1(query=vis, key=txt, value=txt)  # Visual queries text
        txt_to_vis = cross_attn2(query=txt, key=vis, value=vis)  # Text queries visual
        
        # 3. Feed-forward with layer norms
        vis2_v = norm + ff(vis_to_txt)
        vis2_l = norm + ff(txt_to_vis)
        
        # 4. Combine with learned scale parameter
        vis_final = vis * scale * vis2  # Weighted fusion
        
        return vis_final
```

**Purpose**: Modulate visual features using language guidance at each decoder level

---

## 2. Text Encoder (BERT)

```python
class BERTModel(nn.Module):
    # CXR-BERT-specialized (domain-specific for chest X-ray)
    # Outputs: 
    # - Hidden states from 3 layers (1st, 2nd, last)
    # - Project to project_dim via MLP
```

**Features**:
- Frozen BERT (no fine-tuning)
- Multi-layer hidden states aggregation
- Projects to same dimension as vision features (512D in paper)

---

## 3. Output

Two segmentation masks:
```python
return out, out_2  # (High-freq mask, Low-freq mask)
```

Can be:
- **Averaged**: (out + out_2) / 2
- **Ensembled**: Confidence-based selection
- **Soft ensemble**: Weighted combination

---

## 4. Key Components Summary

| Component | Purpose | Input | Output |
|-----------|---------|-------|--------|
| **DWT** | Decompose image into freq bands | Image RGB | H, L images |
| **ConvNeXt x2** | Extract multi-scale features | H & L images | 4-level features |
| **FFBI** | Fuse high & low freq branches | Top-level features | Fused features |
| **LFFI** | Modulate features with text | Features + text | Language-guided features |
| **Decoders x2** | Upsample to mask | Fused features | Segmentation masks |

---

## 5. Why This Design Works (Inductive Bias)

### High-Frequency Branch Benefits:
- ✓ Captures fine boundaries, edges
- ✓ Emphasizes tumor/lesion details
- ✓ Better localization

### Low-Frequency Branch Benefits:
- ✓ Captures structural context
- ✓ Robust to noise/artifacts
- ✓ Better overall shape consistency

### Fusion Strategy:
- ✓ **Early fusion** (FFBI at bottleneck) preserves individual branch learning
- ✓ **Language guidance** (LFFI) helps align features with task intent
- ✓ **Dual outputs** allow uncertainty estimation or soft ensemble

---

## 6. Comparison with Your FreqMedCLIPSAMv2

### Similarities:
- ✓ Frequency-domain decomposition (wavelets)
- ✓ Dual-branch architecture
- ✓ Fusion mechanisms
- ✓ Language guidance for segmentation

### Differences:
| Aspect | FMISeg | FreqMedCLIPSAMv2 |
|--------|--------|-----------------|
| **Image encoding** | ConvNeXt-tiny | BiomedCLIP + FrequencyEncoder |
| **Frequency method** | DWT preprocessing | FrequencyEncoder network (learnable) |
| **Fusion** | FFBI (cross-attention) | DWT → FPNAdapter (learnable networks) |
| **Text encoder** | CXR-BERT frozen | BiomedCLIP frozen |
| **Decoder** | Custom LFFI + Monai | SAM-based (ImageEncoder + MaskDecoder) |
| **Output** | 2 masks (average) | 2 masks (ViT + Freq branches) |

### Your Innovation:
1. **Learnable frequency encoding** (FrequencyEncoder) vs fixed DWT
2. **SAM integration** for stronger base segmentation model
3. **FPN adaptation** for multi-scale frequency features
4. **BiomedCLIP** for better biomedical understanding

---

## 7. FMISeg Training Strategy

```python
# From train.py
loss = segmentation_loss(out, gt) + segmentation_loss(out_2, gt)
# Both branches learn to segment, then ensemble

optimizer = Adam or SGD
scheduler = step/cosine decay
early_stopping = monitor val_MIoU
```

---

## 8. Key Takeaways for Your Implementation

1. **Frequency decomposition is crucial**: Separating high/low frequencies helps model learn complementary features
2. **Bi-directional fusion**: FFBI shows that allowing each branch to attend to the other improves robustness
3. **Language as modulation**: LFFI modulates features with text, not just concatenates
4. **Dual output**: Training both branches separately then fusing is better than single merged branch
5. **Frozen encoders**: Both BERT and vision encoders are frozen, only decoders/fusion modules are trained

---

## Code Locations Reference

- **Frequency preprocessing**: `utils/wave.py` (DWT with Haar)
- **Dual vision encoding**: `net/model.py` lines 55-56 (encoder, encoder2)
- **Fusion (FFBI)**: `net/model.py` lines 41-48
- **Language fusion (LFFI)**: `net/decoder.py` lines 53-98
- **Dataset loading**: `utils/dataset.py` (loads Images_H and Images_L)
