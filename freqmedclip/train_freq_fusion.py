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
from freqmedclip.scripts.new_freq_encoder import WaveletInjector
from freqmedclip.scripts.fmiseg_components import (
    FFBI, Decoder, SubpixelUpsample, UnetOutBlock,
    SemanticAnchor, SmartSpatialGate
)
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

# --- 2. Model Wrapper (Smart Single-Stream Architecture) ---
class FrequencyMedCLIPSAMv2(nn.Module):
    """
    Smart Single-Stream FreqMedCLIP Architecture.
    
    Key changes from original dual-branch:
    1. Uses WaveletInjector (lightweight) instead of ConvNeXtTiny12Ch (heavy)
    2. Uses SemanticAnchor for feed-forward M2IB approximation
    3. Uses SmartSpatialGate for semantic-first gating
    4. Single decoder branch (removes redundant frequency decoder)
    5. Uses 768-dim hidden_states instead of 512-dim pooler_output
    
    Data Flow:
    Image -> BiomedCLIP -> Layer 3 (HF) + Layer 11 (LF)
    Image -> DWT -> WaveletInjector -> F_wav
    F_HF = Layer3 + F_wav
    Anchor = SemanticAnchor(Layer11, Text_CLS)
    Gated = SmartSpatialGate(Anchor, F_HF)
    Output = Decoder(Gated, Text) -> Saliency Map
    """
    def __init__(self, biomedclip_model, args):
        super().__init__()
        self.biomedclip = biomedclip_model
        
        # --- Freeze BiomedCLIP ---
        for param in self.biomedclip.parameters():
            param.requires_grad = False
        
        # --- Smart Single-Stream Components ---
        # Lightweight wavelet projection (replaces heavy ConvNeXt)
        self.wavelet_injector = WaveletInjector(in_channels=12, out_channels=768)
        
        # Feed-forward M2IB approximation (replaces iterative iba.py)
        self.semantic_anchor = SemanticAnchor(dim=768)
        
        # Semantic-first gating
        self.smart_gate = SmartSpatialGate()
        
        # FPN Adapter for multi-scale skip connections
        self.fpn_adapter = FPNAdapter(in_channels=768, out_channels=[768, 384, 192, 96])
        
        # --- Single Decoder Branch ---
        # Spatial dimensions for 224x224 input: 14, 28, 56, 112
        # For 512x512: would be 32, 64, 128, 256 but we resize to 224
        self.spatial_dim = [14, 28, 56, 112]
        feature_dim = [768, 384, 192, 96]
        
        # Single decoder path with LFFI for text guidance
        self.decoder16 = Decoder(feature_dim[0], feature_dim[1], self.spatial_dim[0], 77, embed_dim=768)
        self.decoder8 = Decoder(feature_dim[1], feature_dim[2], self.spatial_dim[1], 77, embed_dim=768)
        self.decoder4 = Decoder(feature_dim[2], feature_dim[3], self.spatial_dim[2], 77, embed_dim=768)
        self.decoder1 = SubpixelUpsample(2, feature_dim[3], 24, 2)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)
        
    def get_high_freq_image(self, pixel_values):
        """Apply Haar DWT to extract frequency domain features."""
        # Returns (B, 12, H/2, W/2) - 4 subbands per RGB channel
        return haar_dwt(pixel_values)

    def forward(self, pixel_values, input_ids, image_raw):
        """
        Smart Single-Stream forward pass.
        
        Data Flow:
        1. BiomedCLIP extracts Layer 3 (shallow/HF) and Layer 11 (deep/LF)
        2. DWT + WaveletInjector creates wavelet features
        3. F_HF = Layer3 + WaveletFeatures (enhanced high-frequency)
        4. SemanticAnchor creates attention mask from Layer11 + Text
        5. SmartSpatialGate applies semantic gating to F_HF
        6. Single decoder path produces saliency map
        """
        # === 1. Get BiomedCLIP expected image size ===
        _, _, H, W = pixel_values.shape
        expected_size = 224  # BiomedCLIP default
        vm = self.biomedclip.vision_model
        if hasattr(vm, 'embeddings') and hasattr(vm.embeddings, 'image_size'):
            expected_size = vm.embeddings.image_size
        elif hasattr(self.biomedclip, 'config'):
            cfg = getattr(self.biomedclip, 'config')
            if hasattr(cfg, 'vision_config') and getattr(cfg.vision_config, 'image_size', None) is not None:
                expected_size = cfg.vision_config.image_size

        # Resize to BiomedCLIP expected size
        if H != expected_size or W != expected_size:
            pixel_values = F.interpolate(pixel_values, size=(expected_size, expected_size), 
                                        mode='bilinear', align_corners=False)

        # === 2. Extract hidden states from BiomedCLIP (768-dim) ===
        vision_outputs = self.biomedclip.vision_model(
            pixel_values, 
            output_hidden_states=True,
            return_dict=True
        )
        # Handle both object and tuple outputs (transformers version compatibility)
        if hasattr(vision_outputs, 'hidden_states'):
            hidden_states = vision_outputs.hidden_states
        else:
            # Tuple format: (last_hidden_state, pooler_output, hidden_states)
            hidden_states = vision_outputs[2] if len(vision_outputs) > 2 else vision_outputs[0]
        
        # Layer 3 (shallow) for high-frequency details - remove CLS token
        # Layer 11 (deep) for semantic features - remove CLS token
        # hidden_states[i] shape: (B, num_patches+1, 768)
        f_hf_raw = hidden_states[3][:, 1:, :]   # (B, 196, 768) for 224x224
        f_lf_raw = hidden_states[11][:, 1:, :]  # (B, 196, 768) for 224x224
        
        # Reshape to spatial format: (B, N, C) -> (B, C, H, W)
        B, N, C = f_lf_raw.shape
        spatial_size = int(N ** 0.5)  # 14 for 224x224 input
        f_hf = f_hf_raw.permute(0, 2, 1).view(B, C, spatial_size, spatial_size)  # (B, 768, 14, 14)
        f_lf = f_lf_raw.permute(0, 2, 1).view(B, C, spatial_size, spatial_size)  # (B, 768, 14, 14)
        
        # === 3. DWT + Wavelet Injection ===
        # Resize raw image to match BiomedCLIP input size
        image_raw_resized = F.interpolate(image_raw, size=(expected_size, expected_size), 
                                         mode='bilinear', align_corners=False)
        dwt_feats = self.get_high_freq_image(image_raw_resized)  # (B, 12, 112, 112) for 224
        f_wav = self.wavelet_injector(dwt_feats)  # (B, 768, 112, 112)
        
        # Resize wavelet features to match Layer 3 spatial size
        f_wav = F.interpolate(f_wav, size=(spatial_size, spatial_size), mode='bilinear', align_corners=False)
        
        # Enhanced high-frequency features = Layer3 + Wavelet
        f_hf_enhanced = f_hf + f_wav  # (B, 768, 14, 14)
        
        # === 4. Text Features (768-dim hidden states) ===
        # Call text_model with return_dict=True to get proper ModelOutput
        text_outputs = self.biomedclip.text_model(
            input_ids=input_ids, 
            output_hidden_states=True,
            return_dict=True
        )
        # Handle both object and tuple outputs (transformers version compatibility)
        if hasattr(text_outputs, 'hidden_states'):
            text_embeds = text_outputs.hidden_states[-1]  # (B, seq_len, 768)
        else:
            # Tuple format: (last_hidden_state, pooler_output, hidden_states)
            # hidden_states is index 2, we want the last layer [-1]
            text_embeds = text_outputs[2][-1] if len(text_outputs) > 2 else text_outputs[0]
        # Get CLS token for SemanticAnchor
        text_cls = text_embeds[:, 0, :]  # (B, 768)
        
        # === 5. SemanticAnchor (M2IB approximation) ===
        # Creates attention mask highlighting where text concept appears
        anchor_map = self.semantic_anchor(f_lf, text_cls)  # (B, 1, 14, 14)
        
        # === 6. SmartSpatialGate ===
        # Gates HF features using semantic anchor ("only edges inside target region")
        gated_features = self.smart_gate(anchor_map, f_hf_enhanced)  # (B, 768, 14, 14)
        
        # === 7. Build FPN Pyramid from ViT layers for skip connections ===
        # Extract multi-scale features from different layers
        layers_idx = [12, 10, 7, 4]  # deep to shallow
        fpn_inputs = []
        for idx in layers_idx:
            feat = hidden_states[idx][:, 1:, :]  # Remove CLS
            feat_reshaped = feat.permute(0, 2, 1).view(B, C, spatial_size, spatial_size)
            fpn_inputs.append(feat_reshaped)
        
        # FPN Adapter creates multi-scale features
        fpn_feats = self.fpn_adapter(fpn_inputs)  # [768@14, 384@28, 192@56, 96@112]
        
        # === 8. Single Decoder Path with text guidance ===
        # Use gated features as the bottleneck input
        bottleneck_flat = rearrange(gated_features, 'b c h w -> b (h w) c')
        
        # Prepare skip connections
        skips = [rearrange(item, 'b c h w -> b (h w) c') for item in fpn_feats]
        
        # Decoder 16: 768->384, 14x14->28x28
        os16, _ = self.decoder16(bottleneck_flat, skips[1], text_embeds)
        
        # Decoder 8: 384->192, 28x28->56x56
        os8, _ = self.decoder8(os16, skips[2], text_embeds)
        
        # Decoder 4: 192->96, 56x56->112x112
        os4, _ = self.decoder4(os8, skips[3], text_embeds)
        
        # Reshape for SubpixelUpsample
        L4 = os4.shape[1]
        H4 = int(L4 ** 0.5)
        os4 = rearrange(os4, 'B (H W) C -> B C H W', H=H4, W=H4)
        
        # Final upsample: 96->24, 112->224
        os1 = self.decoder1(os4)
        
        # Output head
        out = self.out(os1)  # (B, 1, H, W) logits
        
        # === 9. Features for DHN-NCE Loss ===
        # Pool bottleneck features for contrastive loss
        img_feats_pooled = F.adaptive_avg_pool2d(gated_features, (1, 1)).squeeze(-1).squeeze(-1)  # (B, 768)
        text_feats_pooled = text_cls  # (B, 768)
        
        # Return single output (no dual branch)
        # Return None for second output to maintain backward compatibility
        return out, None, img_feats_pooled, text_feats_pooled

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
    
    # Initialize Model (Smart Single-Stream Architecture)
    print("Initializing Smart Single-Stream FreqMedCLIP...")
    model = FrequencyMedCLIPSAMv2(biomedclip, args).to(device)
    
    # Dataset & DataLoader
    print(f"Loading Dataset: {args.dataset}...")
    train_dataset = FreqMedCLIPDataset(args.data_root, args.dataset, processor, tokenizer, split='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0) 
    
    # Optimizer (Smart Single-Stream - trainable modules only)
    backbone_params = filter(lambda p: p.requires_grad, model.biomedclip.parameters())
    
    # Trainable components: WaveletInjector, SemanticAnchor, SmartSpatialGate, FPN, Decoder
    trainable_params = (
        list(model.wavelet_injector.parameters()) +
        list(model.semantic_anchor.parameters()) +
        list(model.smart_gate.parameters()) +
        list(model.fpn_adapter.parameters()) +
        list(model.decoder16.parameters()) +
        list(model.decoder8.parameters()) +
        list(model.decoder4.parameters()) +
        list(model.decoder1.parameters()) +
        list(model.out.parameters())
    )
                     
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.backbone_lr},
        {'params': trainable_params, 'lr': args.lr}
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
            image_raw = batch['image_raw'].to(device)
            input_ids = batch['input_ids'].to(device)
            masks = batch['mask'].to(device).float()
            
            # Forward (Smart Single-Stream - single output)
            preds, _, img_feats, text_feats = model(pixel_values, input_ids, image_raw) 
            preds = preds.squeeze(1)
            
            # Resize ground-truth masks to match prediction spatial size
            if masks.dim() == 3:
                masks_resized = F.interpolate(masks.unsqueeze(1), size=preds.shape[-2:], mode='nearest').squeeze(1)
            else:
                masks_resized = masks
            
            # Loss (Single branch - simplified)
            loss_dice = dice_criterion(preds, masks_resized)
            loss_bce = bce_criterion(preds, masks_resized)
            loss_hnl = hnl_criterion(img_feats, text_feats, batch_size=pixel_values.shape[0])
            
            # Total Loss: Dice + BCE + DHN-NCE
            loss = loss_dice + loss_bce + 0.1 * loss_hnl
            
            loss = loss / args.grad_accum_steps
            loss.backward()
            
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'dice': loss_dice.item(), 'hnl': loss_hnl.item()})
            
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
                
                preds, _, _, _ = model(pixel_values, input_ids, image_raw)
                preds = preds.squeeze(1)

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