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
from matplotlib.gridspec import GridSpec

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import components (Smart Single-Stream Architecture)
from freqmedclip.train_freq_fusion import FreqMedCLIPDataset, FrequencyMedCLIPSAMv2

# Storage for intermediate outputs
intermediate_outputs = {}

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

def visualize_intermediate_steps(vis_dir, img_name, original_image, mask_np, pred_binary, 
                                 freq_features, fpn_features, pred1, pred2, metrics):
    """Visualize intermediate outputs: frequency encoding, FPN features, both branches, final prediction"""
    
    # Normalize to 0-1 for visualization
    def norm_tensor(t):
        if t is None:
            return None
        t = t.detach().cpu()
        if len(t.shape) > 2:
            t = t.mean(dim=0)  # Average channels
        tmin = t.min()
        tmax = t.max()
        if tmax > tmin:
            t = (t - tmin) / (tmax - tmin)
        return t.numpy()
    
    # Create a comprehensive figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Row 1: Input, GT, Pred1 (ViT), Pred2 (Frequency)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_image)
    ax1.set_title('Input Image')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(mask_np, cmap='gray')
    ax2.set_title('Ground Truth Mask')
    ax2.axis('off')
    
    pred1_vis = norm_tensor(pred1.squeeze(0) if len(pred1.shape) > 2 else pred1)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(pred1_vis, cmap='gray')
    ax3.set_title('Prediction (Smart Single-Stream)')
    ax3.axis('off')
    
    # In Smart Single-Stream, pred2 is same as pred1 (no dual branch)
    pred2_vis = norm_tensor(pred2.squeeze(0) if len(pred2.shape) > 2 else pred2)
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(pred2_vis, cmap='gray')
    ax4.set_title('Pred (Sigmoid)')
    ax4.axis('off')
    
    # Row 2: Frequency features (up to 4 channels)
    if freq_features is not None and len(freq_features.shape) > 2:
        freq_vis = norm_tensor(freq_features)
        for idx in range(min(3, freq_features.shape[0])):
            ax = fig.add_subplot(gs[1, idx])
            freq_ch = norm_tensor(freq_features[idx])
            ax.imshow(freq_ch, cmap='viridis')
            ax.set_title(f'Freq Feature Ch{idx}')
            ax.axis('off')
    
    # Row 2, col 3: Overlay (GT vs Final Pred)
    ax_overlay = fig.add_subplot(gs[1, 3])
    overlay = np.zeros((*mask_np.shape, 3))
    overlay[mask_np > 0.5] = [0, 1, 0]  # Green for GT
    overlay[pred_binary > 0.5] = [1, 0, 0]  # Red for pred
    overlay[(mask_np > 0.5) & (pred_binary > 0.5)] = [1, 1, 0]  # Yellow for overlap
    ax_overlay.imshow(overlay)
    ax_overlay.set_title(f'Overlay\nDice: {metrics["dice"]:.3f}')
    ax_overlay.axis('off')
    
    # Row 3: FPN features (multi-scale)
    if fpn_features is not None:
        for idx, fpn_feat in enumerate(fpn_features[:3]):
            fpn_vis = norm_tensor(fpn_feat)
            ax = fig.add_subplot(gs[2, idx])
            if fpn_vis is not None:
                ax.imshow(fpn_vis, cmap='hot')
                ax.set_title(f'FPN Scale {idx}')
            ax.axis('off')
    
    # Row 3, col 3: Final prediction
    ax_final = fig.add_subplot(gs[2, 3])
    ax_final.imshow(pred_binary, cmap='gray')
    ax_final.set_title(f'Final Pred (Threshold=0.5)\nIoU: {metrics["iou"]:.3f}')
    ax_final.axis('off')
    
    plt.suptitle(f'{img_name} | Precision: {metrics["precision"]:.3f} | Recall: {metrics["recall"]:.3f}', 
                 fontsize=14, fontweight='bold')
    
    save_path = os.path.join(vis_dir, f"{img_name.replace('.png', '')}_steps.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


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
    
    # Create model (Smart Single-Stream Architecture)
    print("Initializing Smart Single-Stream model...")
    model = FrequencyMedCLIPSAMv2(biomedclip, args).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle state dict keys if they were saved with 'module.' prefix or similar
    # Also handle if checkpoint is full state dict or just model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and all(k.startswith('module.') for k in checkpoint.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    else:
        state_dict = checkpoint
    
    # Also strip module prefix from state_dict if present
    if isinstance(state_dict, dict) and any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
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
            
            # Forward pass - capture intermediate outputs from model
            # Patch model forward to capture intermediate tensors
            freq_features_list = []
            fpn_features_list = []
            
            # Hook into FrequencyEncoder to capture freq features
            def freq_hook(module, input, output):
                if isinstance(output, (list, tuple)):
                    freq_features_list.extend([o.detach() if isinstance(o, torch.Tensor) else o for o in output])
                elif isinstance(output, torch.Tensor):
                    freq_features_list.append(output.detach())
                else:
                    freq_features_list.append(output)
            
            # Hook into FPNAdapter to capture FPN features
            def fpn_hook(module, input, output):
                if isinstance(output, (list, tuple)):
                    fpn_features_list.extend([o.detach() for o in output if isinstance(o, torch.Tensor)])
                else:
                    fpn_features_list.append(output.detach())
            
            # Register hooks if available (Smart Single-Stream uses wavelet_injector instead of freq_encoder)
            freq_hook_handle = None
            fpn_hook_handle = None
            
            try:
                if hasattr(model, 'wavelet_injector'):
                    freq_hook_handle = model.wavelet_injector.register_forward_hook(freq_hook)
                elif hasattr(model, 'freq_encoder'):
                    freq_hook_handle = model.freq_encoder.register_forward_hook(freq_hook)
                if hasattr(model, 'fpn_adapter'):
                    fpn_hook_handle = model.fpn_adapter.register_forward_hook(fpn_hook)
            except:
                pass
            
            # Forward (Smart Single-Stream - single output)
            # Get image_raw from batch (needed for DWT)
            image_raw = batch['image_raw'].to(device)
            preds1, preds2, _, _ = model(pixel_values, input_ids, image_raw)
            # Smart Single-Stream returns preds2 as None
            preds = preds1 if preds2 is None else (preds1 + preds2) / 2
            preds = preds.squeeze(1)
            
            # Remove hooks
            if freq_hook_handle is not None:
                freq_hook_handle.remove()
            if fpn_hook_handle is not None:
                fpn_hook_handle.remove()
            
            # Resize masks to match prediction size (model outputs 224x224, masks are 512x512)
            if masks.shape[-2:] != preds.shape[-2:]:
                masks_resized = F.interpolate(
                    masks.unsqueeze(1).float(), 
                    size=preds.shape[-2:], 
                    mode='nearest'
                ).squeeze(1)
            else:
                masks_resized = masks.float()
            
            # Calculate metrics and save visualizations for each sample in batch
            for i in range(preds.shape[0]):
                metrics = calculate_metrics(preds[i], masks_resized[i])
                all_metrics.append(metrics)
                
                # Save comprehensive visualization for first 5 samples + every 10th
                if sample_count < 5 or sample_count % 10 == 0:
                    pred_binary = (torch.sigmoid(preds[i]) > 0.5).cpu().numpy()
                    mask_np = masks_resized[i].cpu().numpy()
                    
                    # Get original image from batch for context
                    # Denormalize pixel_values to get image
                    img_np = pixel_values[i].cpu().numpy()
                    img_np = np.transpose(img_np, (1, 2, 0))
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                    
                    # Extract intermediate features
                    freq_feat = freq_features_list[0][i] if freq_features_list else None
                    fpn_feats = [f[i] for f in fpn_features_list] if fpn_features_list else []
                    
                    # Visualize all steps
                    # Smart Single-Stream: show raw logits and sigmoid output
                    pred1_viz = preds1[i] if preds1 is not None else preds[i]
                    pred2_viz = torch.sigmoid(pred1_viz)  # Show sigmoid for second slot
                    visualize_intermediate_steps(
                        vis_dir, 
                        img_names[i],
                        img_np,
                        mask_np,
                        pred_binary,
                        freq_feat,
                        fpn_feats if fpn_feats else None,
                        pred1_viz,
                        pred2_viz,
                        metrics
                    )
                
                sample_count += 1
            
            # Clear intermediate lists for next batch
            freq_features_list.clear()
            fpn_features_list.clear()
    
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
