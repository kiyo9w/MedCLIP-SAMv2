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
from freqmedclip.scripts.freq_components import DWTForward, FrequencyEncoder, FPNAdapter, IDWTInverse
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
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
            A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
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

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            pixel_values = augmented['image']
            mask_tensor = augmented['mask'].long()
        else:
            # Fallback to processor if albumentations missing
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)
            mask_resized = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
            mask_tensor = torch.from_numpy(mask_resized).long()
        
        # Randomly select a text prompt
        text_prompt = random.choice(self.prompts)
        text_inputs = self.tokenizer(text_prompt, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = text_inputs['input_ids'].squeeze(0)
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'mask': mask_tensor,
            'img_name': img_name
        }

# --- 2. Model Wrapper ---
class FrequencyMedCLIPSAMv2(nn.Module):
    def __init__(self, biomedclip_model, dwt_module, idwt_module, freq_encoder, fpn_adapter, args):
        super().__init__()
        self.biomedclip = biomedclip_model
        self.dwt_module = dwt_module
        self.idwt_module = idwt_module
        self.freq_encoder = freq_encoder
        self.fpn_adapter = fpn_adapter
        
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
        
        self.spatial_dim = [14, 28, 56, 112] # Adapted for ViT-B/16 (14x14 start)
        feature_dim = [768, 384, 192, 96]
        
        # Branch 1: Main (ViT)
        self.decoder16 = Decoder(feature_dim[0], feature_dim[1], self.spatial_dim[0], 77) # 768->384, 14->28
        self.decoder8 = Decoder(feature_dim[1], feature_dim[2], self.spatial_dim[1], 77)  # 384->192, 28->56
        self.decoder4 = Decoder(feature_dim[2], feature_dim[3], self.spatial_dim[2], 77)  # 192->96,  56->112
        self.decoder1 = SubpixelUpsample(2, feature_dim[3], 24, 2) # 96->24, 112->224 (scale=2)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)
        
        # Branch 2: Frequency
        self.decoder16_2 = Decoder(feature_dim[0], feature_dim[1], self.spatial_dim[0], 77)
        self.decoder8_2 = Decoder(feature_dim[1], feature_dim[2], self.spatial_dim[1], 77)
        self.decoder4_2 = Decoder(feature_dim[2], feature_dim[3], self.spatial_dim[2], 77)
        self.decoder1_2 = SubpixelUpsample(2, feature_dim[3], 24, 2)
        self.out_2 = UnetOutBlock(2, in_channels=24, out_channels=1)
        
        # FFBI (Bidirectional Interaction)
        self.ffbi = FFBI(feature_dim[0], 4, True)
        
    def get_high_freq_image(self, pixel_values):
        # 1. Denormalize
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=pixel_values.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=pixel_values.device).view(1, 3, 1, 1)
        denorm_imgs = pixel_values * std + mean
        
        # 2. DWT
        dwt_feats = self.dwt_module(denorm_imgs) # (B, 9, 112, 112) -> Actually 9 channels: LH, HL, HH per RGB
        
        # DWTForward returns cat([LH, HL, HH]). It discards LL.
        # So we have only High Freq components.
        # We need to reconstruct Image_H.
        # IDWTInverse expects LL to be present (or we assume it's zero).
        # Our IDWTInverse implementation expects (B, 12, 112, 112) if we include LL?
        # DWTForward returns 9 channels (3 * 3).
        # We need to prepend Zero LL for each channel.
        
        b, c9, h, w = dwt_feats.shape
        # Reshape to (B, 3, 3, H, W)
        dwt_reshaped = dwt_feats.view(b, 3, 3, h, w)
        
        # Create Zero LL: (B, 3, 1, H, W)
        ll_zero = torch.zeros(b, 3, 1, h, w, device=dwt_feats.device)
        
        # Concat: LL, LH, HL, HH -> (B, 3, 4, H, W)
        full_coeffs = torch.cat([ll_zero, dwt_reshaped], dim=2)
        
        # Flatten: (B, 12, H, W)
        full_coeffs_flat = full_coeffs.view(b, 12, h, w)
        
        # 3. IDWT
        img_h = self.idwt_module(full_coeffs_flat) # (B, 3, 224, 224)
        
        return img_h

    def forward(self, pixel_values, input_ids):
        # 1. Image Encoder (ViT) - Branch 1
        vision_outputs = self.biomedclip.vision_model(pixel_values, output_hidden_states=True)
        hidden_states = vision_outputs.hidden_states
        
        # Extract features for FPNAdapter: [3, 6, 9, 12]
        layers_idx = [12, 9, 6, 3] # Order: s1(14), s2(28), s3(56), s4(112)
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
        img_h = self.get_high_freq_image(pixel_values)
        text_outputs = self.biomedclip.text_model(input_ids, output_hidden_states=True)
        text_embeds = text_outputs[0] # (B, L, D)
        
        # FrequencyEncoder returns [f3(14), f2(28), f1(56)]
        # We assume FrequencyEncoder is updated to return 4 scales if we want full match, 
        # but for now we use what we have and maybe pad or reuse?
        # Actually, let's just use the 3 scales we have and maybe duplicate the last one?
        # Or better, let's assume FrequencyEncoder returns [f3, f2, f1].
        # f3(14, 768), f2(28, 384), f1(56, 192).
        # We need s4(112, 96) equivalent.
        # We can just use f1 upsampled? Or just None?
        # Decoder expects a skip.
        # Let's fake the 4th scale by upsampling f1.
        
        freq_feats = self.freq_encoder(img_h, text_embeds=text_embeds)
        # freq_feats: [f3, f2, f1]
        
        # Create f0 (112, 96) from f1 (56, 192)
        f1 = freq_feats[2]
        f0 = F.interpolate(f1, scale_factor=2, mode='bilinear', align_corners=False)
        # Reduce channels 192 -> 96?
        # We don't have a layer for that.
        # Let's just slice it? Or use a 1x1 conv if we had one.
        # For now, let's just slice: f0[:, :96, :, :]
        f0 = f0[:, :96, :, :]
        
        image_features2 = [freq_feats[0], freq_feats[1], freq_feats[2], f0]
        
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
        skips2 = [rearrange(item, 'b c h w -> b (h w) c') for item in image_features2]
        
        # Decoder 16 (14->28)
        os16 = self.decoder16(fu32_flat, skips[1], text_embeds)
        os16_2 = self.decoder16_2(fu32_2_flat, skips2[1], text_embeds)
        
        # Decoder 8 (28->56)
        os8 = self.decoder8(os16, skips[2], text_embeds)
        os8_2 = self.decoder8_2(os16_2, skips2[2], text_embeds)
        
        # Decoder 4 (56->112)
        os4 = self.decoder4(os8, skips[3], text_embeds)
        os4_2 = self.decoder4_2(os8_2, skips2[3], text_embeds)
        
        # Reshape for SubpixelUpsample (expects B, C, H, W)
        # Decoder returns (B, HW, C).
        # Last spatial size was 112.
        os4 = rearrange(os4, 'B (H W) C -> B C H W', H=112, W=112)
        os4_2 = rearrange(os4_2, 'B (H W) C -> B C H W', H=112, W=112)
        
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
        text_feats_pooled = torch.max(text_embeds, dim=1)[0]
        
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
    dwt = DWTForward().to(device)
    idwt = IDWTInverse().to(device)
    
    fpn_adapter = FPNAdapter(in_channels=768, out_channels=[768, 384, 192, 96]).to(device)
    
    # Frequency Encoder (Modified for 3 channels, 4 scales)
    # We need to ensure it returns 4 scales.
    freq_encoder = FrequencyEncoder(in_channels=3, base_channels=96, text_dim=768).to(device)
    
    # Model Wrapper
    model = FrequencyMedCLIPSAMv2(biomedclip, dwt, idwt, freq_encoder, fpn_adapter, args).to(device)
    
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
    
    # Training Loop
    print("Starting Training...")
    os.makedirs(args.save_dir, exist_ok=True)
    
    val_dataset = FreqMedCLIPDataset(args.data_root, args.dataset, processor, tokenizer, split='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    best_dice = 0.0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad() 
        
        for batch_idx, batch in enumerate(pbar):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            masks = batch['mask'].to(device).float()
            
            # Forward
            preds1, preds2, img_feats, text_feats = model(pixel_values, input_ids) 
            preds1 = preds1.squeeze(1)
            preds2 = preds2.squeeze(1)
            
            # Loss (Deep Supervision on both branches)
            loss_dice1 = dice_criterion(preds1, masks)
            loss_bce1 = bce_criterion(preds1, masks)
            
            loss_dice2 = dice_criterion(preds2, masks)
            loss_bce2 = bce_criterion(preds2, masks)
            
            loss_hnl = hnl_criterion(img_feats, text_feats, batch_size=pixel_values.shape[0])
            
            # Total Loss
            loss = (loss_dice1 + loss_bce1) + 0.5 * (loss_dice2 + loss_bce2) + 0.1 * loss_hnl
            
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
                input_ids = batch['input_ids'].to(device)
                masks = batch['mask'].to(device).float()
                
                preds1, preds2, _, _ = model(pixel_values, input_ids)
                # Use Main Branch (preds1) for metrics, or average?
                # FMISeg likely uses main branch.
                preds = preds1.squeeze(1)
                
                for i in range(preds.shape[0]):
                    pred_binary = (torch.sigmoid(preds[i]) > 0.5).float()
                    target = masks[i]
                    
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
