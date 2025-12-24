"""
Save Raw Predictions from FreqMedCLIP Model
Generates saliency maps WITHOUT postprocessing for later batch processing.

Usage:
    python save_freqmedclip_predictions.py --dataset breast_tumors --checkpoint checkpoints/breast_tumors/fusion_breast_tumors_epoch100.pth --output predictions/breast_tumors
"""
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

import sys
# sys.path.insert(0, 'd:/Documents/LMIS/MedCLIP-SAMv2')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from freqmedclip.scripts.freq_components import FrequencyEncoder, FPNAdapter, DWTForward
from freqmedclip.train_freq_fusion import FreqMedCLIPDataset, FrequencyMedCLIPSAMv2
from freqmedclip.scripts.fmiseg_components import Decoder, SubpixelUpsample, UnetOutBlock, FFBI


def load_model(checkpoint_path, device):
    """Load trained FreqMedCLIP model with proper architecture"""
    model_name = "chuhac/BiomedCLIP-vit-bert-hf"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    biomedclip = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

    # Initialize architecture components
    freq_encoder = FrequencyEncoder(in_channels=3, base_channels=96, text_dim=768).to(device)
    fpn_adapter = FPNAdapter(in_channels=768, out_channels=[768, 384, 192, 96]).to(device)
    
    # Wrap model
    class Args: pass
    args = Args()
    model = FrequencyMedCLIPSAMv2(biomedclip, freq_encoder, fpn_adapter, args).to(device)
    model.eval()
    
    # Load checkpoint (only for fusion/decoder weights if available)
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # Try loading full state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            elif isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint, strict=False)
            else:
                print(f"⚠️ Warning: Checkpoint format not recognized, using random weights")
            print(f"✓ Loaded checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"⚠️ Warning: Failed to load checkpoint ({e}), using random weights")
    else:
        print(f"⚠️ Warning: Checkpoint not found at {checkpoint_path}, using random weights")
    
    return model, processor, tokenizer


def save_predictions(dataset_name, checkpoint_path, output_dir, split='test', save_format='png'):
    """Generate and save raw predictions"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Load model
    model, processor, tokenizer = load_model(checkpoint_path, device)
    
    # Load dataset
    dataset = FreqMedCLIPDataset('data', dataset_name, processor, tokenizer, split=split)
    print(f"📊 Dataset: {dataset_name} ({split} split) - {len(dataset)} samples")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each sample
    for idx in tqdm(range(len(dataset)), desc="Generating predictions"):
        sample = dataset[idx]
        img_name = sample['img_name']
        pixel_values = sample['pixel_values'].unsqueeze(0).to(device)
        input_ids = sample['input_ids'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Forward pass returns two branch logits; take mean and apply sigmoid
            pred1, pred2, _, _ = model(pixel_values, input_ids)
            pred_logits = (pred1 + pred2) / 2.0
            pred_mask = torch.sigmoid(pred_logits)
            pred_mask = pred_mask.squeeze().cpu().numpy()
            
            # Normalize to [0, 255]
            pred_normalized = (pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min() + 1e-8)
            pred_uint8 = (pred_normalized * 255).astype(np.uint8)
            
            # Save raw prediction
            output_path = os.path.join(output_dir, img_name)
            cv2.imwrite(output_path, pred_uint8)
    
    print(f"\n✅ Predictions saved to: {output_dir}")
    print(f"   - Total files: {len(dataset)}")
    print(f"\n📌 Next steps:")
    print(f"   1. Postprocess predictions:")
    print(f"      python postprocess_freqmedclip_outputs.py --input {output_dir} --output {output_dir}_cleaned")
    print(f"   2. Evaluate results:")
    print(f"      python evaluation/eval.py --pred-dir {output_dir}_cleaned --gt-dir data/{dataset_name}/{split}_masks")


def main():
    parser = argparse.ArgumentParser(description='Save raw FreqMedCLIP predictions')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Dataset name (e.g., breast_tumors)')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to trained fusion checkpoint (.pth)')
    parser.add_argument('--output', type=str, required=True, 
                        help='Output directory for raw predictions')
    parser.add_argument('--split', type=str, default='test', 
                        choices=['train', 'val', 'test'], 
                        help='Dataset split (default: test)')
    args = parser.parse_args()
    
    save_predictions(
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        split=args.split
    )


if __name__ == '__main__':
    main()
