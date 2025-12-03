import argparse
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from transformers import AutoModel, AutoProcessor, AutoTokenizer
import cv2
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import components
from freqmedclip.scripts.freq_components import DWTForward, FrequencyEncoder, FPNAdapter, IDWTInverse
from freqmedclip.train_freq_fusion import FreqMedCLIPDataset, FrequencyMedCLIPSAMv2

def calculate_metrics(pred, target, threshold=0.5):
    """Calculate Dice, IoU, Precision, Recall"""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = target.float()
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum()
    
    dice = (2. * intersection + 1e-8) / (union + 1e-8)
    iou = (intersection + 1e-8) / (pred_binary.sum() + target_binary.sum() - intersection + 1e-8)
    
    tp = (pred_binary * target_binary).sum()
    fp = (pred_binary * (1 - target_binary)).sum()
    fn = ((1 - pred_binary) * target_binary).sum()
    
    precision = (tp + 1e-8) / (tp + fp + 1e-8)
    recall = (tp + 1e-8) / (tp + fn + 1e-8)
    
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-root', type=str, default='../data')
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load BiomedCLIP from local model
    print("\nLoading BiomedCLIP from local model...")
    model_path = "../saliency_maps/model"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    biomedclip = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
    
    # Initialize components
    print("Initializing components...")
    dwt = DWTForward().to(device)
    idwt = IDWTInverse().to(device)
    fpn_adapter = FPNAdapter(in_channels=768, out_channels=[768, 384, 192, 96]).to(device)
    freq_encoder = FrequencyEncoder(in_channels=3, base_channels=96, text_dim=768).to(device)
    
    # Create model
    # Note: args passed here are just for config if needed, we can pass a dummy object or the parser args
    model = FrequencyMedCLIPSAMv2(biomedclip, dwt, idwt, freq_encoder, fpn_adapter, args).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle state dict keys if they were saved with 'module.' prefix or similar
    # Also handle if checkpoint is full state dict or just model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # Load state dict
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Warning: Strict loading failed ({e}). Trying with strict=False...")
        model.load_state_dict(state_dict, strict=False)
        
    model.eval()
    print("✓ Model loaded successfully")
    
    # Load test dataset
    print(f"\nLoading test dataset: {args.dataset}...")
    # Note: Using 'test' split. Ensure 'test_images' and 'test_masks' exist in data root.
    # If not, user might want to use 'val' or 'train' for testing.
    # We'll assume 'test' as per user code, but fallback to 'val' if needed could be an option.
    test_dataset = FreqMedCLIPDataset(args.data_root, args.dataset, processor, tokenizer, split='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_dataset)}")
    
    # Evaluate
    print("\n" + "="*60)
    print("Evaluating on Test Set")
    print("="*60)
    
    all_metrics = []
    
    # Create visualizations directory
    vis_dir = f"visualizations/{args.dataset}_eval"
    os.makedirs(vis_dir, exist_ok=True)
    print(f"Visualizations will be saved to: {vis_dir}")
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            masks = batch['mask'].to(device)
            img_names = batch['img_name']
            
            # Forward
            # Model returns: out, out_2, img_feats_pooled, text_feats_pooled
            # We use 'out' (Main Branch) for evaluation
            preds1, preds2, _, _ = model(pixel_values, input_ids)
            preds = preds1.squeeze(1)
            
            # Calculate metrics and save visualizations for each sample in batch
            for i in range(preds.shape[0]):
                metrics = calculate_metrics(preds[i], masks[i])
                all_metrics.append(metrics)
                
                # Save visualization every 10 samples or first 5
                if sample_count < 5 or sample_count % 10 == 0:
                    pred_binary = (torch.sigmoid(preds[i]) > 0.5).cpu().numpy()
                    mask_np = masks[i].cpu().numpy()
                    
                    # Create visualization
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Ground truth
                    axes[0].imshow(mask_np, cmap='gray')
                    axes[0].set_title('Ground Truth')
                    axes[0].axis('off')
                    
                    # Prediction
                    axes[1].imshow(pred_binary, cmap='gray')
                    axes[1].set_title(f'Prediction\nDice: {metrics["dice"]:.3f}')
                    axes[1].axis('off')
                    
                    # Overlay
                    overlay = np.zeros((*mask_np.shape, 3))
                    overlay[mask_np > 0.5] = [0, 1, 0]  # Green for GT
                    overlay[pred_binary > 0.5] = [1, 0, 0]  # Red for pred
                    overlay[(mask_np > 0.5) & (pred_binary > 0.5)] = [1, 1, 0]  # Yellow for overlap
                    axes[2].imshow(overlay)
                    axes[2].set_title('Overlay (GT=Green, Pred=Red)')
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    save_path = os.path.join(vis_dir, f"{img_names[i].replace('.png', '')}_vis.png")
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
                
                sample_count += 1
    
    # Aggregate results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    avg_metrics = {
        'dice': np.mean([m['dice'] for m in all_metrics]),
        'iou': np.mean([m['iou'] for m in all_metrics]),
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'recall': np.mean([m['recall'] for m in all_metrics])
    }
    
    std_metrics = {
        'dice': np.std([m['dice'] for m in all_metrics]),
        'iou': np.std([m['iou'] for m in all_metrics]),
        'precision': np.std([m['precision'] for m in all_metrics]),
        'recall': np.std([m['recall'] for m in all_metrics])
    }
    
    print(f"\nDataset: {args.dataset}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"\nMetrics (Mean ± Std):")
    print(f"  Dice Score:  {avg_metrics['dice']:.4f} ± {std_metrics['dice']:.4f}")
    print(f"  IoU:         {avg_metrics['iou']:.4f} ± {std_metrics['iou']:.4f}")
    print(f"  Precision:   {avg_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
    print(f"  Recall:      {avg_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
    
    # Save results
    ckpt_name = os.path.basename(args.checkpoint).replace('.pth', '')
    results_file = f"results_{args.dataset}_{ckpt_name}.txt"
    with open(results_file, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test samples: {len(test_dataset)}\n\n")
        f.write(f"Dice Score:  {avg_metrics['dice']:.4f} ± {std_metrics['dice']:.4f}\n")
        f.write(f"IoU:         {avg_metrics['iou']:.4f} ± {std_metrics['iou']:.4f}\n")
        f.write(f"Precision:   {avg_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}\n")
        f.write(f"Recall:      {avg_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}\n")
    
    print(f"\n✓ Results saved to: {results_file}")
    print(f"✓ Visualizations saved to: {vis_dir}/ ({sample_count} samples)")
    print("="*60)

if __name__ == '__main__':
    main()
