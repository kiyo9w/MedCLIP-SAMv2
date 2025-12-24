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
sys.path.insert(0, 'd:/Documents/LMIS/MedCLIP-SAMv2')

from freqmedclip.scripts.freq_components import SmartFusionBlock, DWTForward
from freqmedclip.train_freq_fusion import FreqMedCLIPDataset, FrequencyMedCLIPSAMv2


def load_model(checkpoint_path, device):
    """Load trained FreqMedCLIP model"""
    model_name = "chuhac/BiomedCLIP-vit-bert-hf"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    biomedclip = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

    # Initialize components
    dwt = DWTForward().to(device)
    fusion = SmartFusionBlock(hf_channels=777, lf_channels=1, out_channels=32).to(device)
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        fusion.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✓ Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"⚠️ Warning: Checkpoint not found at {checkpoint_path}, using random weights")
    
    # Wrap model
    class Args: pass
    args = Args()
    model = FrequencyMedCLIPSAMv2(biomedclip, fusion, dwt, args).to(device)
    model.eval()
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
            # Vision features
            vision_outputs = model.biomedclip.vision_model(pixel_values, output_hidden_states=True)
            last_hidden_state = vision_outputs.last_hidden_state
            
            # Text features
            text_outputs = model.biomedclip.text_model(input_ids, output_hidden_states=True)
            text_embeds = text_outputs[1] if isinstance(text_outputs, tuple) else text_outputs.pooler_output
            
            # Coarse map (dot product)
            patch_embeddings = last_hidden_state[:, 1:, :]
            patch_embeddings = F.normalize(patch_embeddings, dim=-1)
            text_embeds = F.normalize(text_embeds, dim=-1)
            coarse_map_flat = torch.bmm(patch_embeddings, text_embeds.unsqueeze(-1))
            H_feat = W_feat = int(np.sqrt(patch_embeddings.shape[1]))
            coarse_map = coarse_map_flat.view(-1, 1, H_feat, W_feat)
            
            # HF features
            dwt_feats = model.dwt_module(pixel_values)
            shallow_feats = vision_outputs.hidden_states[3][:, 1:, :]
            shallow_feats = shallow_feats.permute(0,2,1).view(-1,768,H_feat,W_feat)
            shallow_up = F.interpolate(shallow_feats, size=(112,112), mode='bilinear', align_corners=False)
            hf_features = torch.cat([shallow_up, dwt_feats], dim=1)
            
            # Fusion (fine map)
            fine_map = model.fusion_block(hf_features, coarse_map)
            
            # Upsample to 224x224
            pred_mask = F.interpolate(fine_map, size=(224,224), mode='bilinear', align_corners=False)
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
