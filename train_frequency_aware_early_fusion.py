"""
Frequency-Aware MedCLIP-SAMv2 Training Script (Following brainstorm.md)

Training strategy (CLIP alignment only - no segmentation training):
- Stage 1: Warm-up frequency modules with CLIP loss (BiomedCLIP frozen)
- Stage 2: Fine-tune vision encoder with frequency injection (Optional, Risky)
- Stage 3: End-to-end vision encoder optimization (Optional, High Risk)

Goal: Learn frequency-aware features that improve saliency maps at inference time,
      WITHOUT training segmentation - preserve zero-shot capability.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from PIL import Image
from torchvision import transforms
import csv
# Import HardNegativeLoss (DHN-NCE)
from loss.hnl import HardNegativeLoss

# Import frequency-aware modules
from frequency_aware import (
    DualStreamPreprocessor,
    HighFreqProjection,
    FeatureFusionGate,
    FrequencyAwareSaliencyGenerator,
    FrequencyAwarePipeline
)

# Import BiomedCLIP (adjust path as needed)
from transformers import AutoModel, AutoProcessor, AutoTokenizer

# Setup logging with timestamped log file
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
log_filename = log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to file: {log_filename}")


# Removed DiceLoss and BoundaryLoss - not needed for CLIP training
# Following brainstorm.md: only train with contrastive loss, no segmentation supervision


class MedPixDataset(Dataset):
    """MedPix caption-only dataset for BiomedCLIP fine-tuning."""

    def __init__(self, csv_path, tokenizer, image_size=224, max_text_len=64):
        self.csv_path = Path(csv_path)
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_text_len = max_text_len

        self.samples = []
        with self.csv_path.open('r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if 'Caption' not in reader.fieldnames or 'filename' not in reader.fieldnames:
                raise ValueError("CSV must have columns: Caption, filename")
            missing = 0
            kept = 0
            # Known dataset root: MedCLIP-SAMv2-finetune/data/medpix_dataset
            ds_root = self.csv_path.parent  # .../data/medpix_dataset
            # Attempt to find repository root (MedCLIP-SAMv2-finetune)
            # csv_path: .../MedCLIP-SAMv2-finetune/data/medpix_dataset/medpix_dataset.csv
            # parents[0] -> .../data/medpix_dataset, parents[1] -> .../data, parents[2] -> .../MedCLIP-SAMv2-finetune
            try:
                repo_root = self.csv_path.parents[2]
            except Exception:
                repo_root = ds_root.parent
            for row in reader:
                caption = row['Caption']
                raw_path = row['filename'].replace('\\', '/')
                # Normalize duplicate segments like 'data/data/medpix_dataset' → 'data/medpix_dataset'
                raw_path = raw_path.replace('data/data/medpix_dataset', 'data/medpix_dataset')
                # Resolve image path candidates
                candidates = []
                p = Path(raw_path)
                if p.is_absolute():
                    candidates.append(p)
                else:
                    # If the CSV path contains the repository-relative path (data/medpix_dataset/...)
                        if 'data/medpix_dataset' in raw_path:
                            # raw_path is like 'data/medpix_dataset/images/xxx.jpg'
                            candidates.append(repo_root / raw_path)
                        # Common case: filename is just 'images/xxx.jpg' or a bare path
                        candidates.append(ds_root / raw_path)
                        # Directly under images folder with basename
                        candidates.append(ds_root / 'images' / Path(raw_path).name)
                chosen = None
                for c in candidates:
                    if c.exists():
                        chosen = c
                        break
                # If not found, try basename lookup in images folder
                if chosen is None:
                    basename = Path(raw_path).name
                    images_dir = ds_root / 'images'
                    candidate2 = images_dir / basename
                    if candidate2.exists():
                        chosen = candidate2
                    else:
                        # As last resort, search recursively for the basename under images_dir
                        try:
                            found = next(images_dir.rglob(basename))
                            chosen = found
                        except StopIteration:
                            chosen = None
                if chosen is None:
                    missing += 1
                    continue
                self.samples.append((caption, str(chosen.resolve())))
                kept += 1
        logger.info(f"MedPix: kept {kept} samples, missing {missing} images from CSV")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        caption, img_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        text_tokens = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_len,
            return_tensors='pt'
        )
        # Squeeze batch dimension from tokenizer outputs
        text_tokens = {k: v.squeeze(0) for k, v in text_tokens.items()}

        # Caption-only training - no masks needed
        return img, text_tokens


def collate_fn_clip(batch):
    """Custom collate function for CLIP training."""
    images, texts = zip(*batch)
    
    # Stack images
    images = torch.stack(images)
    
    # Collate text tokens (dict of tensors)
    text_ids = torch.stack([t['input_ids'] for t in texts])
    text_attention = torch.stack([t['attention_mask'] for t in texts])
    texts_collated = {
        'input_ids': text_ids,
        'attention_mask': text_attention
    }
    
    return images, texts_collated


def create_medpix_dataloaders(config, tokenizer):
    """Create train/val dataloaders for MedPix caption-only dataset."""
    csv_path = config.get('csv_path', 'medpix_dataset/medpix_dataset.csv')
    val_ratio = config.get('val_ratio', 0.1)
    batch_size = config.get('batch_size', 8)
    num_workers = config.get('num_workers', 4)

    dataset = MedPixDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        image_size=config.get('image_size', 224),
        max_text_len=config.get('max_text_len', 64)
    )

    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_clip
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_clip
    )

    return train_loader, val_loader


class FreqAwareModel(nn.Module):
    """
    Frequency-aware model with EARLY FUSION (inject wavelet before transformer blocks).
    Feature_input = PatchEmbed(Image) + α × Projection(HighFreqWavelet)
    Then Feature_input goes through transformer blocks.
    """
    def __init__(self, config, device='cuda'):
        super().__init__()
        self.config = config
        self.device = device
        
        # Load BiomedCLIP from saliency_maps/model
        logger.info("Loading BiomedCLIP model...")
        saliency_model_path = Path('saliency_maps/model')
        
        if not saliency_model_path.exists():
            raise FileNotFoundError(
                f"BiomedCLIP model not found at {saliency_model_path.absolute()}\n"
                f"Please ensure pytorch_model.bin, config.json, configuration_biomed_clip.py, "
                f"modeling_biomed_clip.py, and processing_biomed_clip.py exist in this directory."
            )
        
        logger.info(f"Loading BiomedCLIP from: {saliency_model_path.absolute()}")
        self.biomedclip = AutoModel.from_pretrained(
            str(saliency_model_path.absolute()),
            trust_remote_code=True
        ).to(device)
        logger.info("BiomedCLIP loaded successfully from saliency_maps/model")
        
        # Initialize frequency-aware modules
        self.preprocessor = DualStreamPreprocessor(
            wavelet_type=config.get('wavelet_type', 'haar'),
            image_size=config.get('image_size', 224)
        )
        
        self.high_freq_proj = HighFreqProjection(
            embedding_dim=config.get('embedding_dim', 768),
            num_patches=config.get('num_patches', 196)
        ).to(device)
        
        self.fusion_gate = FeatureFusionGate(
            embedding_dim=config.get('embedding_dim', 768),
            fusion_ratio=config.get('fusion_ratio', 0.1)
        ).to(device)
        
        self.saliency_gen = FrequencyAwareSaliencyGenerator(
            blur_kernel=config.get('blur_kernel', 5),
            morphology_kernel=config.get('morphology_kernel', 5),
            frequency_weight=config.get('frequency_weight', 0.3)
        ).to(device)
        
        logger.info("Model initialized successfully")
    
    def get_fused_features(self, images):
        """Get frequency-aware fused image features with EARLY fusion (inject before transformer blocks)."""
        # Dual-stream preprocessing
        original_streams = []
        high_freq_streams = []
        
        for img in images:
            prep_result = self.preprocessor(img.cpu().numpy().transpose(1, 2, 0))
            original_streams.append(prep_result['original_stream'])
            high_freq_streams.append(prep_result['high_freq_enhanced'])
        
        original_stream = torch.stack(original_streams).to(self.device)
        high_freq_stream = torch.stack(high_freq_streams).to(self.device)
        
        # Project high-frequency features FIRST (before BiomedCLIP)
        high_freq_proj_raw = self.high_freq_proj(high_freq_stream)  # [B, many_patches, 768]
        
        # === EARLY FUSION: Inject wavelet into patch embeddings ===
        # Get patch embeddings from vision model (before transformer blocks)
        vision_model = self.biomedclip.vision_model
        
        # 1. Patch embedding using the embeddings layer
        with torch.no_grad() if not self.training else torch.enable_grad():
            # Process through patch embedding layer (Conv2d)
            patch_out = vision_model.embeddings.patch_embedding(original_stream)  # [B, 768, 14, 14]
            
            # Reshape to [B, N, 768]
            x = patch_out.flatten(2).transpose(1, 2)  # [B, 196, 768]
            
            # Add class token
            batch_size = x.shape[0]
            class_tokens = vision_model.embeddings.class_embedding.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 1, 768]
            x = torch.cat([class_tokens, x], dim=1)  # [B, 197, 768]
            
            # Add positional encoding using position_ids
            x = x + vision_model.embeddings.position_embedding(vision_model.embeddings.position_ids)
        
        # Resize high_freq_proj to match patch count (196 patches, excluding cls token)
        # If high_freq_proj has more patches, take the first 196
        # If fewer, pad with zeros
        num_patches = x.shape[1] - 1  # 196 patches
        if high_freq_proj_raw.shape[1] > num_patches:
            high_freq_proj = high_freq_proj_raw[:, :num_patches, :]  # [B, 196, 768]
        elif high_freq_proj_raw.shape[1] < num_patches:
            # Pad with zeros
            padding = torch.zeros(
                high_freq_proj_raw.shape[0],
                num_patches - high_freq_proj_raw.shape[1],
                high_freq_proj_raw.shape[2],
                device=high_freq_proj_raw.device,
                dtype=high_freq_proj_raw.dtype
            )
            high_freq_proj = torch.cat([high_freq_proj_raw, padding], dim=1)  # [B, 196, 768]
        else:
            high_freq_proj = high_freq_proj_raw  # [B, 196, 768]
        
        # Add zero for cls token dimension
        cls_freq = torch.zeros(high_freq_proj.shape[0], 1, high_freq_proj.shape[2], 
                              device=high_freq_proj.device, dtype=high_freq_proj.dtype)
        high_freq_proj = torch.cat([cls_freq, high_freq_proj], dim=1)  # [B, 197, 768]
        
        # 2. EARLY FUSION: Feature_input = PatchEmbed(Image) + α × Projection(HighFreqWavelet)
        fusion_alpha = self.fusion_gate.fusion_alpha
        x_fused = x + fusion_alpha * high_freq_proj  # Early injection!
        
        # 3. Pass through transformer blocks with fused features
        # Create attention mask (no masking needed - all positions attend to all positions)
        batch_size, seq_len = x_fused.shape[0], x_fused.shape[1]
        attention_mask = torch.ones((batch_size, 1, seq_len, seq_len), dtype=torch.float32, device=self.device)
        
        # Pass through all transformer layers
        for layer in vision_model.encoder.layers:
            x_fused = layer(x_fused, attention_mask=attention_mask)[0]
        
        # Apply post layernorm (no separate encoder norm needed - it's included in pre-norm layers)
        x_fused = vision_model.post_layernorm(x_fused)
        
        # Pool to [B, 768] for contrastive learning using CLS token
        fused_pooled = x_fused[:, 0]  # CLS token
        
        return fused_pooled
    
    def get_text_features(self, texts):
        """Get text features from BiomedCLIP."""
        text_output = self.biomedclip.text_model(
            input_ids=texts['input_ids'].to(self.device),
            attention_mask=texts['attention_mask'].to(self.device)
        )
        # text_output is a tuple: (last_hidden_state, pooler_output)
        return text_output[1]  # [B, 768] - pooler_output


class Trainer:
    """
    Trainer for frequency-aware CLIP alignment.
    No segmentation training - only vision-language alignment.
    """
    def __init__(self, config, device='cuda', checkpoint_dir=None):
        self.config = config
        self.device = device
        self.current_stage = 1
        
        # Initialize model
        self.model = FreqAwareModel(config, device)
        
        # Loss function: DHN-NCE contrastive loss only
        self.clip_loss = HardNegativeLoss(alpha=0)
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Metrics tracking
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_metrics = []
        
        # Checkpointing
        if checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints/early_fusion'))
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Checkpoint directory: {self.checkpoint_dir.absolute()}")
    
    def setup_stage(self, stage):
        """Setup optimizer and scheduler for each stage (CLIP alignment only)."""
        self.current_stage = stage
        
        if stage == 1:
            # Stage 1: Freeze BiomedCLIP, train frequency modules with CLIP loss
            logger.info("=== STAGE 1: Warm-up Frequency Modules (CLIP Alignment - EARLY FUSION) ===")
            logger.info("Goal: Learn frequency representations compatible with BiomedCLIP semantic space")
            logger.info("Method: Inject wavelet features BEFORE transformer blocks (early fusion)")
            logger.info("Formula: Feature_input = PatchEmbed(Image) + alpha * Projection(HighFreqWavelet)")
            logger.info("Loss: DHN-NCE contrastive (image-text alignment)")
            logger.info("Freezing BiomedCLIP, training frequency modules only")
            
            for param in self.model.biomedclip.parameters():
                param.requires_grad = False
            
            trainable_params = (
                list(self.model.high_freq_proj.parameters()) +
                list(self.model.fusion_gate.parameters())
            )
            
            self.optimizer = Adam(
                trainable_params,
                lr=self.config.get('stage1_lr', 1e-4),
                weight_decay=self.config.get('weight_decay', 1e-5)
            )
            self.scheduler = None
            self.epochs = self.config.get('stage1_epochs', 10)
            
        elif stage == 2:
            # Stage 2: Fine-tune vision encoder (Optional, Risky)
            logger.info("=== STAGE 2: Vision Encoder Fine-tuning (Optional - Risky) ===")
            logger.info("WARNING: May hurt zero-shot performance on other domains")
            logger.info("Goal: Adapt vision encoder to be frequency-aware")
            logger.info("Loss: Still DHN-NCE contrastive (NOT segmentation)")
            logger.info("Unfreezing last transformer blocks, keeping text encoder frozen")
            
            # Freeze text encoder (preserve zero-shot)
            for param in self.model.biomedclip.text_model.parameters():
                param.requires_grad = False
            
            # Unfreeze last N blocks of vision encoder
            N = self.config.get('unfreeze_blocks', 2)
            blocks = self.model.biomedclip.vision_model.encoder.layers
            
            for i, block in enumerate(blocks):
                if i >= len(blocks) - N:
                    for param in block.parameters():
                        param.requires_grad = True
                        logger.info(f"  Unfrozen vision block {i}")
            
            # Different learning rates for different modules
            param_groups = [
                {
                    'params': self.model.high_freq_proj.parameters(),
                    'lr': self.config.get('stage2_freq_lr', 5e-5)
                },
                {
                    'params': self.model.fusion_gate.parameters(),
                    'lr': self.config.get('stage2_freq_lr', 5e-5)
                },
                {
                    'params': [p for p in self.model.biomedclip.vision_model.parameters()
                              if p.requires_grad],
                    'lr': self.config.get('stage2_vision_lr', 1e-5)  # Low LR to preserve pretrained
                }
            ]
            
            self.optimizer = AdamW(
                param_groups,
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('stage2_epochs', 10)
            )
            self.epochs = self.config.get('stage2_epochs', 10)
            
        elif stage == 3:
            # Stage 3: Full vision encoder (High Risk)
            logger.info("=== STAGE 3: End-to-End Vision Encoder (High Risk - Not Recommended) ===")
            logger.info("HIGH RISK: Likely to damage zero-shot capability significantly")
            logger.info("Goal: Full adaptation to specific dataset (domain-specific)")
            logger.info("Loss: Still DHN-NCE contrastive")
            logger.info("Unfreezing entire vision encoder, text encoder stays frozen")
            
            for param in self.model.biomedclip.vision_model.parameters():
                param.requires_grad = True
            for param in self.model.biomedclip.text_model.parameters():
                param.requires_grad = False
            
            param_groups = [
                {
                    'params': self.model.high_freq_proj.parameters(),
                    'lr': self.config.get('stage3_freq_lr', 2e-5)
                },
                {
                    'params': self.model.fusion_gate.parameters(),
                    'lr': self.config.get('stage3_freq_lr', 2e-5)
                },
                {
                    'params': self.model.biomedclip.vision_model.parameters(),
                    'lr': self.config.get('stage3_vision_lr', 5e-6)  # Very low to avoid catastrophic forgetting
                }
            ]
            
            self.optimizer = AdamW(
                param_groups,
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('stage3_epochs', 20)
            )
            self.epochs = self.config.get('stage3_epochs', 20)
        
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
        logger.info(f"Training for {self.epochs} epochs")
    
    def compute_loss(self, images, texts):
        """Compute CLIP contrastive loss with EARLY-fused frequency-aware features."""
        # Get frequency-aware image features
        image_features = self.model.get_fused_features(images)  # [B, 768]
        
        # Get text features
        text_features = self.model.get_text_features(texts)  # [B, 768]
        
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # DHN-NCE contrastive loss
        batch_size = image_features.shape[0]
        contrastive_loss = self.clip_loss(image_features, text_features, batch_size)
        
        # Regularization: encourage fusion_alpha to stay small (0.05-0.15)
        fusion_alpha = self.model.fusion_gate.fusion_alpha
        alpha_reg = 0.01 * torch.abs(fusion_alpha - 0.1)  # Target alpha = 0.1
        
        total_loss = contrastive_loss + alpha_reg
        
        loss_dict = {
            'contrastive': contrastive_loss.item(),
            'alpha_reg': alpha_reg.item(),
            'fusion_alpha': fusion_alpha.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def train_epoch(self, dataloader):
        """Train for one epoch with CLIP contrastive loss."""
        self.model.train()
        total_loss = 0
        loss_components = {}
        
        pbar = tqdm(dataloader, desc=f"Stage {self.current_stage} Training")
        
        for batch_idx, batch in enumerate(pbar):
            images, texts = batch
            images = images.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    loss, loss_dict = self.compute_loss(images, texts)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=1.0
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, loss_dict = self.compute_loss(images, texts)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=1.0
                )
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Accumulate loss components
            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0
                loss_components[key] += value
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'alpha': f"{loss_dict['fusion_alpha']:.3f}"
            })
        
        # Average losses
        avg_loss = total_loss / len(dataloader)
        for key in loss_components:
            loss_components[key] /= len(dataloader)
        
        return avg_loss, loss_components
    
    @torch.no_grad()
    def validate(self, dataloader):
        """Validation pass with CLIP metrics."""
        self.model.eval()
        total_loss = 0
        loss_components = {}
        
        for batch in tqdm(dataloader, desc="Validation"):
            images, texts = batch
            images = images.to(self.device)
            
            loss, loss_dict = self.compute_loss(images, texts)
            total_loss += loss.item()
            
            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0
                loss_components[key] += value
        
        # Average
        avg_loss = total_loss / len(dataloader)
        for key in loss_components:
            loss_components[key] /= len(dataloader)
        
        metrics = {
            'val_loss': avg_loss,
            **loss_components
        }
        
        return metrics
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint = {
            'stage': self.current_stage,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        # Allow passing either:
        # - a bare filename (e.g. 'stage1_best.pth') -> resolved under self.checkpoint_dir
        # - a path that already includes the checkpoint_dir (relative or absolute)
        # - an absolute path
        candidate = Path(filename)
        if candidate.is_absolute() and candidate.exists():
            path = candidate
        else:
            # If the file exists as given (relative to cwd), use it
            if candidate.exists():
                path = candidate
            else:
                # Try resolving under the configured checkpoint_dir
                path_under_dir = self.checkpoint_dir / candidate
                if path_under_dir.exists():
                    path = path_under_dir
                else:
                    # Try if filename itself already contains checkpoint_dir twice (handle user mistakes)
                    alt = Path(str(self.checkpoint_dir) + os.sep + str(candidate))
                    if alt.exists():
                        path = alt
                    else:
                        raise FileNotFoundError(f"Checkpoint not found: checked {candidate} and {path_under_dir}")

        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Only load optimizer/scheduler if they have the same parameter groups
        # (different stages may have different unfrozen parameters)
        try:
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except ValueError as e:
            logger.warning(f"Could not load optimizer state due to different param groups: {e}")
            logger.info("Optimizer and scheduler will use fresh state for this stage")
        
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        logger.info(f"Checkpoint loaded: {path}")
    
    def train(self, train_loader, val_loader):
        """Main training loop for current stage."""
        logger.info(f"\nStarting Stage {self.current_stage} training...")
        
        for epoch in range(self.epochs):
            logger.info(f"\nEpoch {epoch+1}/{self.epochs}")
            
            # Train
            train_loss, loss_components = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Log
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"  Contrastive: {loss_components['contrastive']:.4f}")
            logger.info(f"  Alpha Reg: {loss_components['alpha_reg']:.4f}")
            logger.info(f"  Fusion Alpha: {loss_components['fusion_alpha']:.4f}")
            logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"  Val Contrastive: {val_metrics['contrastive']:.4f}")
            
            # LR scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Save checkpoints
            self.save_checkpoint(f'stage{self.current_stage}_epoch{epoch+1}.pth')
            
            # Save best model (lower loss is better)
            if val_metrics['val_loss'] < self.best_loss:
                self.best_loss = val_metrics['val_loss']
                self.save_checkpoint(f'stage{self.current_stage}_best.pth')
                logger.info(f"New best loss: {self.best_loss:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint(f'stage{self.current_stage}_final.pth')
        logger.info(f"Stage {self.current_stage} training complete!")


def main():
    parser = argparse.ArgumentParser(description='Frequency-Aware MedCLIP-SAMv2 Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3], default=1,
                       help='Training stage (1, 2, or 3)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Custom directory to save checkpoints')

    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Tokenizer for captions
    tokenizer = AutoTokenizer.from_pretrained(
        config.get('tokenizer_name', 'chuhac/BiomedCLIP-vit-bert-hf'),
        trust_remote_code=True
    )

    # Build dataloaders (caption-only)
    train_loader, val_loader = create_medpix_dataloaders(config, tokenizer)
    logger.info(f"Dataset loaded: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val samples")

    # Initialize trainer
    trainer = Trainer(config, device, checkpoint_dir=args.checkpoint_dir)
    
    # Setup stage
    trainer.setup_stage(args.stage)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
