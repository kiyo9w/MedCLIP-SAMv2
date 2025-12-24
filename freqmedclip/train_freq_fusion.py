import argparse
import os
import sys
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer
import cv2
from einops import rearrange

# Try importing albumentations
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not found. Data augmentation will be disabled.")

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our custom components
from freqmedclip.scripts.freq_components import FPNAdapter, haar_dwt
from freqmedclip.scripts.new_freq_encoder import ConvNeXtTiny12Ch # New Encoder
from freqmedclip.scripts.fmiseg_components import FFBI, Decoder, SubpixelUpsample, UnetOutBlock
from scripts.methods import vision_heatmap_iba
from saliency_maps.text_prompts import *
from loss.hnl import HardNegativeLoss

# --- 0. Data Augmentation ---
def get_transforms(split='train'):
    if not ALBUMENTATIONS_AVAILABLE:
        return None
        
    if split == 'train':
        return A.Compose([
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            # Fix deprecation: remove invalid alpha_affine; optional mild Affine
            A.ElasticTransform(alpha=1, sigma=50, p=0.2),
            A.Affine(scale=(0.95, 1.05), rotate=(-5, 5), shear=(-5, 5), translate_percent=(0.0, 0.02), p=0.2),
            A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ToTensorV2()
        ])

# --- 1. Dataset Class ---
class FreqMedCLIPDataset(Dataset):
    def __init__(self, root_dir, dataset_name, processor, tokenizer, split='train', max_length=77):
        """
        Args:
            root_dir (str): Path to 'data' directory
            dataset_name (str): Name of dataset (e.g., 'breast_tumors')
            processor: BiomedCLIP processor (used if albumentations not available or for text)
            tokenizer: BiomedCLIP tokenizer
            split (str): 'train' or 'val'
        """
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.processor = processor
        self.tokenizer = tokenizer
        self.split = split
        self.max_length = max_length
        
        self.img_dir = os.path.join(root_dir, dataset_name, f"{split}_images")
        self.mask_dir = os.path.join(root_dir, dataset_name, f"{split}_masks")
        
        self.image_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        self.transforms = get_transforms(split)
        
        # Load prompts based on dataset name
        self.prompts = []
        if 'breast' in dataset_name:
            self.prompts = breast_tumor_P2_prompts + benign_breast_tumor_P3_prompts + malignant_breast_tumor_P3_prompts
        elif 'lung' in dataset_name:
            self.prompts = lung_CT_P2_prompts + lung_xray_P2_prompts + covid_lung_P3_prompts + viral_pneumonia_lung_P3_prompts + lung_opacity_P3_prompts
        elif 'brain' in dataset_name:
            self.prompts = brain_tumor_P2_prompts + glioma_brain_tumor_P3_prompts + meningioma_brain_tumor_P3_prompts + pituitary_brain_tumor_P3_prompts
        else:
            self.prompts = ["A medical image showing an abnormality."]
            
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # Load Image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load Mask
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError("Mask not found")
            mask = (mask > 127).astype(np.float32) # Binary mask
        except Exception as e:
            # print(f"Error loading mask {mask_path}: {e}")
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        # Resize to ensure consistent dimensions before augmentation
        target_size = 512
        if image.shape[0] != target_size or image.shape[1] != target_size:
            image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        if mask.shape[0] != target_size or mask.shape[1] != target_size:
            mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

        
        # Prepare Raw Image [0, 1] for Frequency Branch
        # We need to apply spatial transforms but NOT normalization
        # Albumentations doesn't support returning multiple "images" easily with different pipelines
        # So we manually Inverse Normalize or (Cleaner) Re-apply spatial only.
        # Efficient hack: We use the already augmented 'image'(0-255 RGB) if ALBUMENTATIONS_AVAILABLE
        
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            pixel_values = augmented['image'] # Normalized Tensor
            mask_tensor = augmented['mask'].long()
            
            # To get raw image with same spatial transforms:
            # We can use ReplayCompose or just rely on 'additional_targets' if we refactored get_transforms.
            # But here, let's just do a simplified approach:
            # The 'augmented' dict doesn't contain the intermediate un-normalized image.
            # We can inverse normalize 'pixel_values'.
            
            # Mean/Std from get_transforms
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
            
            image_raw = pixel_values * std + mean
            image_raw = torch.clamp(image_raw, 0, 1)
            # image_raw is (C, H, W). We need (H, W, C) for consistency with below or just keep checks.
            image_raw = image_raw.permute(1, 2, 0) # Back to HWC for consistency logic below
            
        else:
            # Fallback to processor if albumentations missing
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)
            mask_resized = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            mask_tensor = torch.from_numpy(mask_resized).long()
            
            # Raw image resized
            image_resized = cv2.resize(image, (512, 512))
            image_raw = torch.from_numpy(image_resized.astype(np.float32) / 255.0)

        # Ensure image_raw is correct shape/type for return
        # Logic above ensures image_raw is HWC, [0,1]

        
        # Randomly select a text prompt
        text_prompt = random.choice(self.prompts)
        text_inputs = self.tokenizer(text_prompt, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = text_inputs['input_ids'].squeeze(0)
        
        return {
            'pixel_values': pixel_values,
            'image_raw': image_raw.permute(2, 0, 1), # (C, H, W) in [0, 1]
            'input_ids': input_ids,
            'mask': mask_tensor,
            'img_name': img_name
        }

# --- 2. Model Wrapper ---
class FrequencyMedCLIPSAMv2(nn.Module):
    def __init__(self, biomedclip_model, freq_encoder, fpn_adapter, args):
        super().__init__()
        self.biomedclip = biomedclip_model
        self.freq_encoder = freq_encoder
        self.fpn_adapter = fpn_adapter
        
        # --- Mandatory Text Projector (512 -> 768) ---
        # Config says hidden=768, proj=512. BiomedCLIP outputs 512.
        self.text_projector = nn.Linear(512, 768)
        
        # --- Freeze BiomedCLIP ---
        for param in self.biomedclip.parameters():
            param.requires_grad = False
        
        # Unfreeze specific layers for multi-scale adaptation
        layers_to_unfreeze = [3, 6, 9, 11] # 0-indexed
        for i in layers_to_unfreeze:
            for param in self.biomedclip.vision_model.encoder.layers[i].parameters():
                param.requires_grad = True
        self.biomedclip.vision_model.post_layernorm.requires_grad_(True)
        
        # --- Dual-Branch Architecture (FMISeg-original) ---
        
        
        # self.spatial_dim = [14, 28, 56, 112] # OLD (224x224)
        # NEW (512x512): 512/16 = 32. Pyramid: [32, 64, 128, 256]
        self.spatial_dim = [32, 64, 128, 256] 
        feature_dim = [768, 384, 192, 96]
        
        # Branch 1: Main (ViT)
        self.decoder16 = Decoder(feature_dim[0], feature_dim[1], self.spatial_dim[0], 77, embed_dim=768) # 768->384, 14->28
        self.decoder8 = Decoder(feature_dim[1], feature_dim[2], self.spatial_dim[1], 77, embed_dim=768)  # 384->192, 28->56
        self.decoder4 = Decoder(feature_dim[2], feature_dim[3], self.spatial_dim[2], 77, embed_dim=768)  # 192->96,  56->112
        self.decoder1 = SubpixelUpsample(2, feature_dim[3], 24, 2) # 96->24, 112->224 (scale=2)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)
        
        # Branch 2: Frequency
        self.decoder16_2 = Decoder(feature_dim[0], feature_dim[1], self.spatial_dim[0], 77, embed_dim=768)
        self.decoder8_2 = Decoder(feature_dim[1], feature_dim[2], self.spatial_dim[1], 77, embed_dim=768)
        self.decoder4_2 = Decoder(feature_dim[2], feature_dim[3], self.spatial_dim[2], 77, embed_dim=768)
        self.decoder1_2 = SubpixelUpsample(2, feature_dim[3], 24, 2)
        self.out_2 = UnetOutBlock(2, in_channels=24, out_channels=1)
        
        # FFBI (Bidirectional Interaction)
        self.ffbi = FFBI(feature_dim[0], 4, True)

        # Fusion Conv for High-Res Skip (96+96 -> 96)
        self.fusion_conv = nn.Conv2d(feature_dim[3] * 2, feature_dim[3], kernel_size=1)
        
    def get_high_freq_image(self, pixel_values):
        # Use Haar Wavelet Transform (DWT) -> returns (B, 12, 112, 112)
        return haar_dwt(pixel_values)

    def forward(self, pixel_values, input_ids, image_raw):
        # 1. Image Encoder (ViT) - Branch 1
        # Ensure input spatial size matches BiomedCLIP's expected image size
        _, _, H, W = pixel_values.shape
        expected_size = None
        vm = self.biomedclip.vision_model
        if hasattr(vm, 'embeddings') and hasattr(vm.embeddings, 'image_size'):
            expected_size = vm.embeddings.image_size
        elif hasattr(self.biomedclip, 'config'):
            # try common config locations
            cfg = getattr(self.biomedclip, 'config')
            if hasattr(cfg, 'vision_config') and getattr(cfg.vision_config, 'image_size', None) is not None:
                expected_size = cfg.vision_config.image_size
            elif getattr(cfg, 'image_size', None) is not None:
                expected_size = cfg.image_size

        if expected_size is None:
            expected_size = 224

        if H != expected_size or W != expected_size:
            pixel_values = F.interpolate(pixel_values, size=(expected_size, expected_size), mode='bilinear', align_corners=False)

        vision_outputs = self.biomedclip.vision_model(pixel_values, output_hidden_states=True)
        hidden_states = vision_outputs.hidden_states
        
        # Extract features for FPNAdapter: [3, 6, 9, 12]     
        layers_idx = [12, 10, 7, 4]
        fpn_inputs = []
        for idx in layers_idx:
            feat = hidden_states[idx][:, 1:, :] 
            B, N, C = feat.shape
            H = W = int(N**0.5) # 14
            feat_reshaped = feat.permute(0, 2, 1).view(B, C, H, W)
            fpn_inputs.append(feat_reshaped)
        
        # FPN Adapter -> [s1(768), s2(384), s3(192), s4(96)]
        fpn_feats = self.fpn_adapter(fpn_inputs)
        image_features = fpn_feats # [s1, s2, s3, s4]
        
        # 2. Frequency Encoder - Branch 2
        # Use Raw Image for DWT (Crucial fix)
        # Resize raw image to the same spatial size used by the ViT encoder
        # so that feature pyramids align spatially.
        image_raw_resized = F.interpolate(image_raw, size=(expected_size, expected_size), mode='bilinear', align_corners=False)
        img_h = self.get_high_freq_image(image_raw_resized) # image_raw is [0,1] tensor
        
        text_outputs = self.biomedclip.text_model(input_ids, output_hidden_states=True)
        text_embeds = text_outputs[0]

        # Apply Text Projection if needed. Some BiomedCLIP text backbones already
        # output 768-d vectors; others output 512-d. Handle both gracefully.
        if hasattr(self.text_projector, 'in_features') and text_embeds.shape[-1] == self.text_projector.in_features:
            text_embeds = self.text_projector(text_embeds)
        elif text_embeds.shape[-1] == getattr(self.text_projector, 'out_features', 768):
            # already projected to desired dim
            pass
        else:
            # Fallback: create a temporary linear to project to desired dim
            proj = nn.Linear(text_embeds.shape[-1], getattr(self.text_projector, 'out_features', 768)).to(text_embeds.device)
            with torch.no_grad():
                proj.weight.zero_(); proj.bias.zero_()
            text_embeds = proj(text_embeds)
        
        # FrequencyEncoder returns [f3(14), f2(28), f1(56)]
        # We assume FrequencyEncoder is updated to return 4 scales if we want full match, 
        # but for now we use what we have and maybe pad or reuse?
        # Actually, let's just use the 3 scales we have and maybe duplicate the last one?
        # Or better, let's assume FrequencyEncoder returns [f3, f2, f1].
        # f3(14, 768), f2(28, 384), f1(56, 192).
        # We need s4(112, 96) equivalent.
        # We can just use f1 upsampled? Or just None?
        # Decoder expects a skip.
        # ConvNeXtTiny12Ch returns [s4(768), s3(384), s2(192), s1(96)]
        # We need to map these to Decoder expectations.
        # Decoder expects: [s1, s2, s3, s4] where s1 is Bottleneck? 
        # Wait, in Init:
        # self.spatial_dim = [14, 28, 56, 112] (For 224 input)
        # With 512 input:
        # ViT-B/16: 512/16 = 32. So bottleneck is 32x32.
        # FPNAdapter output matches ViT stages.
        
        # For Frequency Branch:
        # s4: 16x16 (768), s3: 32x32 (384), s2: 64x64 (192), s1: 128x128 (96)
        
        # Image Features 2 should correspond to:
        # [Bottleneck, Skip1, Skip2, HighResSkip]
        # Bottleneck = s4 (16x16) or s3 (32x32) to match ViT(32x32)?
        # ViT bottleneck is 32x32. So we should use s3?
        # But ConvNeXt is strong.
        # Let's align with ViT stages.
        # ViT features from FPNAdapter are [s1, s2, s3, s4] in original code (14, 28, 56, 112).
        # Actually FPNAdapter in original code returns [s1(14), s2(28), s3(56), s4(112)].
        # Here with 512 input, ViT features will be [32, 64, 128, 256] ? No.
        # Layer 12 -> 32x32.
        # So FPNAdapter will output [32, 64, 128, 256].
        
        # ConvNeXt returns features: [s4(768, 16x16), s3(384, 32x32), s2(192, 64x64), s1(96, 128x128)]
        # Note: Resolutions above assume 512x512 input. 
        # Since DWT halves input to 256x256, actual ConvNeXt feature sizes are:
        # s4(8x8), s3(16x16), s2(32x32), s1(64x64)
        
        
        # --- Semantic Alignment & Coarse Map ---
        
        # 1. Unpack Frequency Features
        # With Dilated Convolutions (Stride 1 in Stages 2 & 3):
        # DWT(256) -> s1(64) -> s2(32) -> s3(32) -> s4(32)
        freq_feats = self.freq_encoder(img_h) 
        s4, s3, s2, s1 = freq_feats # [768, 384, 192, 96]

        # Ensure frequency high-level map (`s4`) matches ViT bottleneck spatial size
        # so gating below broadcasts correctly.
        vit_bottleneck = image_features[0]
        _, _, vbH, vbW = vit_bottleneck.shape
        if s4.shape[2] != vbH or s4.shape[3] != vbW:
            s4 = F.interpolate(s4, size=(vbH, vbW), mode='bilinear', align_corners=False)
        
        # 2. Coarse Map Calculation (M2IB-style Text Guidance)
        # This re-injects semantic awareness into the frequency branch as requested.
        # Dot product of ViT Bottleneck (image_features[0]) and Text CLS token.
        
        vit_bottleneck = image_features[0] # (B, 768, 32, 32)
        text_cls = text_embeds[:, 0, :]   # (B, 768) - Already projected to 768
        
        # (B, 768, 32, 32) * (B, 768, 1, 1) -> Sum(dim=1) -> (B, 1, 32, 32)
        # This map highlights where the text thinks the object is.
        coarse_map = (vit_bottleneck * text_cls.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)
        coarse_map = torch.sigmoid(coarse_map) # Gate [0, 1]
        
        # 3. Gate the High-Level Frequency Features
        # "Only look at edges where the ViT sees a tumor"
        s4 = s4 * coarse_map 
        
        # 4. Alignment & Pyramid Construction
        # Target: [Bottleneck(32x32), Skip1(64x64), Skip2(128x128), HighRes(256x256)]
        
        # s4 (32x32) -> Matches Bottleneck (32x32). No upsample needed!
        bn = s4 
        
        # s3 (32x32) -> Needs 64x64. Upsample 2x.
        sk1 = F.interpolate(s3, scale_factor=2, mode='bilinear', align_corners=False)
        
        # s2 (32x32) -> Needs 128x128. Upsample 4x.
        sk2 = F.interpolate(s2, scale_factor=4, mode='bilinear', align_corners=False)
        
        # s1 (64x64) -> Needs 256x256. Upsample 4x.
        f0  = F.interpolate(s1, scale_factor=4, mode='bilinear', align_corners=False)
        
        image_features2 = [bn, sk1, sk2, f0] 
        # --- ALIGNMENT END ---
        
        # 3. Bottleneck Interaction (FFBI)
        # My Bottleneck is index 0 (768, 14x14).
        os32 = image_features[0]
        os32_2 = image_features2[0]
        
        # Flatten for Attention: (B, C, H, W) -> (B, H*W, C) -> (B, L, C)
        # FFBI expects (B, L, C) if batch_first=True.
        
        os32_flat = rearrange(os32, 'b c h w -> b (h w) c')
        os32_2_flat = rearrange(os32_2, 'b c h w -> b (h w) c')
        
        fu32_flat, fu32_2_flat = self.ffbi(os32_flat, os32_2_flat)
        
        # 4. Decoding
        # Branch 1
        # Prepare skips (flatten)
        skips = [rearrange(item, 'b c h w -> b (h w) c') for item in image_features]

        # Align frequency branch pyramid spatial sizes to ViT branch before flattening
        aligned_image_features2 = []
        for ref, feat in zip(image_features, image_features2):
            _, _, ref_h, ref_w = ref.shape
            if feat.shape[2] != ref_h or feat.shape[3] != ref_w:
                feat = F.interpolate(feat, size=(ref_h, ref_w), mode='bilinear', align_corners=False)
            aligned_image_features2.append(feat)

        skips2 = [rearrange(item, 'b c h w -> b (h w) c') for item in aligned_image_features2]
        
        # Decoder 16 (14->28)
        # Branch 1
        os16, _ = self.decoder16(fu32_flat, skips[1], text_embeds)
        # Branch 2
        os16_2, _ = self.decoder16_2(fu32_2_flat, skips2[1], text_embeds)
        
        # Decoder 8 (28->56)
        os8, _ = self.decoder8(os16, skips[2], text_embeds)
        os8_2, _ = self.decoder8_2(os16_2, skips2[2], text_embeds)
        
        # Decoder 4 (56->112)
        # Cross-Injection
        # Inject f0 (112x112) from Frequency Branch (image_features2[3]) into ViT Branch
        # image_features2[3] is f0, which has shape (B, 96, 112, 112)
        
        # Fusion Strategy: Fuse ViT (s4) and Freq (f0)
        vit_s4 = image_features[3]   # (B, 96, 112, 112) - now valid
        freq_f0 = image_features2[3] # (B, 96, 112, 112)
        
        fused_skip = torch.cat([vit_s4, freq_f0], dim=1)
        fused_skip = self.fusion_conv(fused_skip)
        
        cnn_high_res_skip_flat = rearrange(fused_skip, 'b c h w -> b (h w) c')
        os4, _ = self.decoder4(os8, cnn_high_res_skip_flat, text_embeds)
        os4_2, _ = self.decoder4_2(os8_2, skips2[3], text_embeds)
        
        # Reshape for SubpixelUpsample (expects B, C, H, W)
        # Decoder returns (B, HW, C).
        # Last spatial size was 256 (spatial_dim[3]).
        # Infer spatial size from token length instead of hardcoding (supports different input sizes)
        L4 = os4.shape[1]
        H4 = int(L4 ** 0.5)
        W4 = H4
        os4 = rearrange(os4, 'B (H W) C -> B C H W', H=H4, W=W4)

        L4_2 = os4_2.shape[1]
        H4_2 = int(L4_2 ** 0.5)
        W4_2 = H4_2
        os4_2 = rearrange(os4_2, 'B (H W) C -> B C H W', H=H4_2, W=W4_2)
        
        # Decoder 1 (112->224)
        os1 = self.decoder1(os4)
        os1_2 = self.decoder1_2(os4_2)
        
        # Output
        out = self.out(os1) # Logits (Sigmoid applied in loss or later)
        out_2 = self.out_2(os1_2)
        
        # Prepare features for HNL (Contrastive Loss)
        # Pool bottleneck features: (B, C, 14, 14) -> (B, C)
        # Use fused features
        fu32 = rearrange(fu32_flat, 'b (h w) c -> b c h w', h=14, w=14)
        img_feats_pooled = F.adaptive_avg_pool2d(fu32, (1, 1)).squeeze(-1).squeeze(-1)
        text_feats_pooled = text_embeds[:, 0, :] # Use [CLS] token
        
        return out, out_2, img_feats_pooled, text_feats_pooled

# --- 3. Loss Function ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice

# --- 4. Main Training Loop ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., breast_tumors)')
    parser.add_argument('--data-root', type=str, default='../data', help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32) 
    parser.add_argument('--grad-accum-steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4) 
    parser.add_argument('--backbone-lr', type=float, default=1e-5) 
    parser.add_argument('--save-dir', type=str, default='../checkpoints')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--dry-run', action='store_true', help='Run a single batch for debugging')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load BiomedCLIP from local model
    print("Loading BiomedCLIP from local model...")
    model_name = "../saliency_maps/model"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    biomedclip = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    
    # Initialize Components
    print("Initializing Fusion Components...")
    fpn_adapter = FPNAdapter(in_channels=768, out_channels=[768, 384, 192, 96]).to(device)
    
    # Frequency Encoder (New ConvNeXt)
    freq_encoder = ConvNeXtTiny12Ch(pretrained=True).to(device) # No more in_channels/base args needed meant for custom
    
    # Model Wrapper
    model = FrequencyMedCLIPSAMv2(biomedclip, freq_encoder, fpn_adapter, args).to(device)
    
    # Dataset & DataLoader
    print(f"Loading Dataset: {args.dataset}...")
    train_dataset = FreqMedCLIPDataset(args.data_root, args.dataset, processor, tokenizer, split='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0) 
    
    # Optimizer
    backbone_params = filter(lambda p: p.requires_grad, model.biomedclip.parameters())
    decoder_params = list(model.decoder16.parameters()) + list(model.decoder8.parameters()) + \
                     list(model.decoder4.parameters()) + list(model.decoder1.parameters()) + \
                     list(model.out.parameters()) + \
                     list(model.decoder16_2.parameters()) + list(model.decoder8_2.parameters()) + \
                     list(model.decoder4_2.parameters()) + list(model.decoder1_2.parameters()) + \
                     list(model.out_2.parameters()) + \
                     list(model.freq_encoder.parameters()) + \
                     list(model.fpn_adapter.parameters()) + \
                     list(model.ffbi.parameters())
                     
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.backbone_lr},
        {'params': decoder_params, 'lr': args.lr}
    ])
    
    # Loss
    dice_criterion = DiceLoss()
    bce_criterion = nn.BCEWithLogitsLoss()
    hnl_criterion = HardNegativeLoss()
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # Check if checkpoint is wrapped or direct state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch']
            else:
                # Direct state_dict
                model.load_state_dict(checkpoint)
                # Extract epoch from filename
                import re
                match = re.search(r'epoch(\d+)', os.path.basename(args.resume))
                if match:
                    start_epoch = int(match.group(1))
            
            print(f"Resumed from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {args.resume}, starting from scratch")
    
    # Training Loop
    print("Starting Training...")
    os.makedirs(args.save_dir, exist_ok=True)
    
    val_dataset = FreqMedCLIPDataset(args.data_root, args.dataset, processor, tokenizer, split='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    best_dice = 0.0
    best_epoch = 0
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad() 
        
        for batch_idx, batch in enumerate(pbar):
            pixel_values = batch['pixel_values'].to(device)
            image_raw = batch['image_raw'].to(device) # New input
            input_ids = batch['input_ids'].to(device)
            masks = batch['mask'].to(device).float()
            
            # Forward
            preds1, preds2, img_feats, text_feats = model(pixel_values, input_ids, image_raw) 
            preds1 = preds1.squeeze(1)
            preds2 = preds2.squeeze(1)
            # Resize ground-truth masks to match prediction spatial size
            if masks.dim() == 3:
                masks_resized = F.interpolate(masks.unsqueeze(1), size=preds1.shape[-2:], mode='nearest').squeeze(1)
            else:
                masks_resized = masks
            
            # Loss (Deep Supervision on both branches)
            loss_dice1 = dice_criterion(preds1, masks_resized)
            loss_bce1 = bce_criterion(preds1, masks_resized)
            
            loss_dice2 = dice_criterion(preds2, masks_resized)
            loss_bce2 = bce_criterion(preds2, masks_resized)
            
            loss_hnl = hnl_criterion(img_feats, text_feats, batch_size=pixel_values.shape[0])
            
            # Total Loss
            # Total Loss (0.2 weight for frequency branch as per critique)
            loss = (loss_dice1 + loss_bce1) + 0.2 * (loss_dice2 + loss_bce2) + 0.1 * loss_hnl
            
            loss = loss / args.grad_accum_steps
            loss.backward()
            
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'd1': loss_dice1.item(), 'd2': loss_dice2.item()})
            
            if args.dry_run:
                print("Dry run completed successfully.")
                return
        
        avg_loss = epoch_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_dice_scores = []
        val_iou_scores = []
        
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(device)
                image_raw = batch['image_raw'].to(device)
                input_ids = batch['input_ids'].to(device)
                masks = batch['mask'].to(device).float()
                
                preds1, preds2, _, _ = model(pixel_values, input_ids, image_raw)
                # Use Main Branch (preds1) for metrics
                preds = preds1.squeeze(1)

                # Resize masks to prediction size
                if masks.dim() == 3:
                    masks_resized = F.interpolate(masks.unsqueeze(1), size=preds.shape[-2:], mode='nearest').squeeze(1)
                else:
                    masks_resized = masks

                for i in range(preds.shape[0]):
                    pred_binary = (torch.sigmoid(preds[i]) > 0.5).float()
                    target = masks_resized[i]

                    intersection = (pred_binary * target).sum()
                    union = pred_binary.sum() + target.sum()
                    dice = (2. * intersection + 1e-8) / (union + 1e-8)
                    iou = (intersection + 1e-8) / (pred_binary.sum() + target.sum() - intersection + 1e-8)

                    val_dice_scores.append(dice.item())
                    val_iou_scores.append(iou.item())
        
        avg_dice = np.mean(val_dice_scores)
        avg_iou = np.mean(val_iou_scores)
        
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f} | Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f}")
        
        if avg_dice > best_dice:
            best_dice = avg_dice
            best_epoch = epoch + 1
            
            old_checkpoints = [f for f in os.listdir(args.save_dir) if f.startswith(f"fusion_{args.dataset}_") and f.endswith('.pth')]
            for old_ckpt in old_checkpoints:
                os.remove(os.path.join(args.save_dir, old_ckpt))
            
            checkpoint_path = os.path.join(args.save_dir, f"fusion_{args.dataset}_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[BEST] New best model saved! Dice: {best_dice:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Training completed! Best epoch: {best_epoch} (Dice: {best_dice:.4f})")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()