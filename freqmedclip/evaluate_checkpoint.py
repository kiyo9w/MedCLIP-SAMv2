import os
import sys
import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Ensure repo root in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from train_freq_fusion import FreqMedCLIPDataset, FrequencyMedCLIPSAMv2, ConvNeXtTiny12Ch, FPNAdapter
from transformers import AutoProcessor, AutoTokenizer, AutoModel


def evaluate(checkpoint_path, dataset, data_root, batch_size, out_path, device, split='val'):
    device = torch.device(device)
    print(f"Using device: {device}")

    model_name = os.path.join(os.path.dirname(__file__), '..', 'saliency_maps', 'model')
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    biomedclip = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

    fpn_adapter = FPNAdapter(in_channels=768, out_channels=[768,384,192,96]).to(device)
    freq_encoder = ConvNeXtTiny12Ch(pretrained=False).to(device)
    model = FrequencyMedCLIPSAMv2(biomedclip, freq_encoder, fpn_adapter, args=argparse.Namespace()).to(device)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and all(k.startswith('module.') for k in ckpt.keys()):
        # maybe DataParallel
        state = {k.replace('module.', ''): v for k, v in ckpt.items()}
    elif isinstance(ckpt, dict):
        # direct state dict
        state = ckpt
    else:
        state = ckpt

    try:
        model.load_state_dict(state)
    except Exception as e:
        print("Failed to load full state_dict directly, trying strict=False",
              flush=True)
        model.load_state_dict(state, strict=False)

    model.eval()

    eval_dataset = FreqMedCLIPDataset(data_root, dataset, processor, tokenizer, split=split)
    val_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    dice_list = []
    iou_list = []
    per_image = []

    sigmoid = torch.nn.Sigmoid()

    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch['pixel_values'].to(device)
            image_raw = batch['image_raw'].to(device)
            input_ids = batch['input_ids'].to(device)
            masks = batch['mask'].to(device).float()

            preds1, preds2, _, _ = model(pixel_values, input_ids, image_raw)
            preds = preds1.squeeze(1)
            probs = sigmoid(preds)  # (B, Hp, Wp)

            # Ensure predictions and masks share the same spatial size
            # masks: (B, H, W)
            if probs.dim() == 3:
                probs_unsq = probs.unsqueeze(1)  # (B,1,Hp,Wp)
            else:
                probs_unsq = probs

            target_h, target_w = masks.shape[1], masks.shape[2]
            if probs_unsq.shape[-2] != target_h or probs_unsq.shape[-1] != target_w:
                probs_resized = F.interpolate(probs_unsq, size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(1)
            else:
                probs_resized = probs

            for i in range(probs_resized.shape[0]):
                p = (probs_resized[i] > 0.5).float()
                t = masks[i]
                intersection = (p * t).sum().item()
                union = p.sum().item() + t.sum().item()
                dice = (2. * intersection + 1e-8) / (union + 1e-8)
                iou = (intersection + 1e-8) / (p.sum().item() + t.sum().item() - intersection + 1e-8)
                dice_list.append(dice)
                iou_list.append(iou)
                per_image.append({
                    'img_name': batch['img_name'][i],
                    'dice': float(dice),
                    'iou': float(iou)
                })

    metrics = {
        'dataset': dataset,
        'split': split,
        'num_images': len(per_image),
        'mean_dice': float(np.mean(dice_list) if dice_list else 0.0),
        'mean_iou': float(np.mean(iou_list) if iou_list else 0.0),
        'per_image': per_image
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Also save CSV summary
    csv_path = out_path.replace('.json', '.csv')
    with open(csv_path, 'w') as f:
        f.write('img_name,dice,iou\n')
        for it in per_image:
            f.write(f"{it['img_name']},{it['dice']:.6f},{it['iou']:.6f}\n")

    print(f"Saved metrics: {out_path} and {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--data-root', default='../data')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--out', default='./checkpoints/metrics.json')
    parser.add_argument('--split', default='val', choices=['train','val','test'], help='Dataset split to evaluate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    evaluate(args.checkpoint, args.dataset, args.data_root, args.batch_size, args.out, args.device, split=args.split)
