"""
Training script for FreqMedCLIP Saliency Model.

This model combines:
- DWT frequency injection
- FFBI bidirectional feature fusion
- LFFI text-guided decoders
- Dual branch deep supervision
"""

import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer
import cv2

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from freqmedclip.scripts.saliency_sam_model import SaliencyModel
from saliency_maps.text_prompts import *
from loss.hnl import HardNegativeLoss


def get_transforms(split='train', input_size=224):
    if not ALBUMENTATIONS_AVAILABLE:
        return None
    if split == 'train':
        return A.Compose([
            A.Resize(input_size, input_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                       std=(0.26862954, 0.26130258, 0.27577711)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(input_size, input_size),
            A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                       std=(0.26862954, 0.26130258, 0.27577711)),
            ToTensorV2()
        ])


class SaliencyDataset(Dataset):
    def __init__(self, root_dir, dataset_name, processor, tokenizer, split='train', 
                 max_length=24, input_size=224):  # FMISeg uses max_length=24!
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.processor = processor
        self.tokenizer = tokenizer
        self.split = split
        self.max_length = max_length
        self.input_size = input_size
        
        self.img_dir = os.path.join(root_dir, dataset_name, f"{split}_images")
        self.mask_dir = os.path.join(root_dir, dataset_name, f"{split}_masks")
        
        self.image_files = sorted([
            f for f in os.listdir(self.img_dir) 
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        self.transforms = get_transforms(split, input_size)
        
        # Load prompts
        self.prompts = []
        if 'breast' in dataset_name:
            self.prompts = (breast_tumor_P2_prompts + benign_breast_tumor_P3_prompts + 
                          malignant_breast_tumor_P3_prompts)
        elif 'lung' in dataset_name:
            self.prompts = (lung_CT_P2_prompts + lung_xray_P2_prompts + covid_lung_P3_prompts + 
                          viral_pneumonia_lung_P3_prompts + lung_opacity_P3_prompts)
        elif 'brain' in dataset_name:
            self.prompts = (brain_tumor_P2_prompts + glioma_brain_tumor_P3_prompts + 
                          meningioma_brain_tumor_P3_prompts + pituitary_brain_tumor_P3_prompts)
        else:
            self.prompts = ["A medical image showing an abnormality."]
            
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            base_name = os.path.splitext(img_name)[0]
            for ext in ['.png', '.jpg', '.jpeg']:
                alt_path = os.path.join(self.mask_dir, base_name + ext)
                if os.path.exists(alt_path):
                    mask = cv2.imread(alt_path, cv2.IMREAD_GRAYSCALE)
                    break
        if mask is None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # CRITICAL: Ensure mask matches image size BEFORE transforms
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        mask = (mask > 127).astype(np.float32)
        
        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            pixel_values = transformed['image']
            mask_tensor = transformed['mask']
        else:
            image = cv2.resize(image, (self.input_size, self.input_size))
            mask = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
            pixel_values = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask_tensor = torch.from_numpy(mask).float()
        
        raw_image = torch.from_numpy(
            cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), 
                      (self.input_size, self.input_size)).transpose(2, 0, 1)
        ).float() / 255.0
        
        prompt = random.choice(self.prompts)
        text_inputs = self.tokenizer(
            prompt, padding='max_length', max_length=self.max_length,
            truncation=True, return_tensors='pt'
        )
        
        return {
            'pixel_values': pixel_values,
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'mask': mask_tensor,
            'raw_image': raw_image,
            'img_name': img_name
        }


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1 - dice


def calculate_metrics(pred, target):
    pred_binary = (pred > 0.5).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum()
    dice = (2. * intersection + 1e-8) / (union + 1e-8)
    iou = (intersection + 1e-8) / (pred_binary.sum() + target.sum() - intersection + 1e-8)
    return dice.item(), iou.item()


def main():
    parser = argparse.ArgumentParser(description='Train FreqMedCLIP Saliency Model')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data-root', type=str, default='../data')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--grad-accum-steps', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--backbone-lr', type=float, default=1e-5)
    parser.add_argument('--save-dir', type=str, default='../checkpoints/saliency')
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load BiomedCLIP
    print("Loading BiomedCLIP...")
    model_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    local_model_path = "../saliency_maps/model"
    model_path = local_model_path if os.path.exists(local_model_path) else model_name
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    biomedclip = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
    
    # Initialize Model
    print("Initializing Saliency Model...")
    model = SaliencyModel(
        biomedclip_model=biomedclip,
        input_size=args.input_size,
        freeze_biomedclip=True,
        unfreeze_layers=[3, 6, 9, 11],
        max_text_len=24  # FMISeg uses 24!
    ).to(device)
    
    # Resume if specified
    start_epoch = 0
    best_dice = 0.0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            start_epoch = checkpoint.get('epoch', 0)
            best_dice = checkpoint.get('best_dice', 0.0)
        else:
            model.load_state_dict(checkpoint, strict=False)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Datasets
    print(f"Loading Dataset: {args.dataset}...")
    train_dataset = SaliencyDataset(args.data_root, args.dataset, processor, tokenizer, 
                                    split='train', max_length=24, input_size=args.input_size)
    val_dataset = SaliencyDataset(args.data_root, args.dataset, processor, tokenizer, 
                                  split='val', max_length=24, input_size=args.input_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Optimizer
    backbone_params = []
    new_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'biomedclip' in name:
                backbone_params.append(param)
            else:
                new_params.append(param)
                
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.backbone_lr},
        {'params': new_params, 'lr': args.lr}
    ], weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Loss
    dice_loss = DiceLoss()
    bce_loss = nn.BCELoss()
    hnl_loss = HardNegativeLoss()
    
    # Training
    print("Starting Training...")
    os.makedirs(args.save_dir, exist_ok=True)
    best_epoch = 0
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            masks = batch['mask'].to(device)
            raw_images = batch['raw_image'].to(device)
            
            outputs = model(pixel_values, input_ids, raw_images)
            
            pred1 = outputs['pred1'].squeeze(1)
            pred2 = outputs['pred2'].squeeze(1)
            
            # Resize masks if needed
            if masks.shape[-2:] != pred1.shape[-2:]:
                masks_resized = F.interpolate(masks.unsqueeze(1), size=pred1.shape[-2:], 
                                             mode='nearest').squeeze(1)
            else:
                masks_resized = masks
            
            # Deep supervision loss
            loss_dice1 = dice_loss(pred1, masks_resized)
            loss_bce1 = bce_loss(pred1, masks_resized)
            loss_dice2 = dice_loss(pred2, masks_resized)
            loss_bce2 = bce_loss(pred2, masks_resized)
            
            loss_hnl = hnl_loss(outputs['image_embed'], outputs['text_embed'], pixel_values.shape[0])
            
            # Total loss: main branch + 0.5 * aux branch + 0.1 * contrastive
            loss = (loss_dice1 + loss_bce1) + 0.5 * (loss_dice2 + loss_bce2) + 0.1 * loss_hnl
            
            loss = loss / args.grad_accum_steps
            loss.backward()
            
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * args.grad_accum_steps
            pbar.set_postfix({
                'loss': f"{loss.item() * args.grad_accum_steps:.4f}",
                'd1': f"{1 - loss_dice1.item():.4f}",
                'd2': f"{1 - loss_dice2.item():.4f}"
            })
            
            if args.dry_run:
                print(f"\nDry run completed!")
                print(f"Pred1 shape: {pred1.shape}, Pred2 shape: {pred2.shape}")
                return
        
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()
        
        # Validation - use train mode to avoid BatchNorm issues
        model.train()
        val_dice_scores = []
        val_iou_scores = []
        
        first_batch = True
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                masks = batch['mask'].to(device)
                raw_images = batch['raw_image'].to(device)
                
                outputs = model(pixel_values, input_ids, raw_images)
                
                # Use average of both branches
                pred = (outputs['pred1'] + outputs['pred2']) / 2
                pred = pred.squeeze(1)
                
                if first_batch:
                    print(f"  Pred: min={pred.min():.4f}, max={pred.max():.4f}, mean={pred.mean():.4f}")
                    first_batch = False
                
                if masks.shape[-2:] != pred.shape[-2:]:
                    masks_resized = F.interpolate(masks.unsqueeze(1), size=pred.shape[-2:], 
                                                 mode='nearest').squeeze(1)
                else:
                    masks_resized = masks
                
                for i in range(pred.shape[0]):
                    dice, iou = calculate_metrics(pred[i], masks_resized[i])
                    val_dice_scores.append(dice)
                    val_iou_scores.append(iou)
        
        avg_dice = np.mean(val_dice_scores)
        avg_iou = np.mean(val_iou_scores)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Loss: {avg_loss:.4f} | Val Dice: {avg_dice:.4f} | Val IoU: {avg_iou:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
        
        if avg_dice > best_dice:
            best_dice = avg_dice
            best_epoch = epoch + 1
            
            old_ckpts = [f for f in os.listdir(args.save_dir) 
                        if f.startswith(f"saliency_{args.dataset}_") and f.endswith('.pth')]
            for old in old_ckpts:
                os.remove(os.path.join(args.save_dir, old))
            
            ckpt_path = os.path.join(args.save_dir, f"saliency_{args.dataset}_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'args': vars(args)
            }, ckpt_path)
            print(f"  [BEST] New best model! Dice: {best_dice:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Training completed! Best epoch: {best_epoch} (Dice: {best_dice:.4f})")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
