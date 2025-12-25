import os
import sys
import argparse
import json
import torch
import numpy as np
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from train_freq_fusion import FreqMedCLIPDataset, FrequencyMedCLIPSAMv2, ConvNeXtTiny12Ch, FPNAdapter
from transformers import AutoProcessor, AutoTokenizer, AutoModel


def compute_metrics(checkpoint, dataset, data_root, batch_size, out_path, device):
    device = torch.device(device)
    model_name = os.path.join(os.path.dirname(__file__), '..', 'saliency_maps', 'model')
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    biomedclip = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

    fpn_adapter = FPNAdapter(in_channels=768, out_channels=[768,384,192,96]).to(device)
    freq_encoder = ConvNeXtTiny12Ch(pretrained=False).to(device)
    model = FrequencyMedCLIPSAMv2(biomedclip, freq_encoder, fpn_adapter, args=argparse.Namespace()).to(device)

    ckpt = torch.load(checkpoint, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and all(k.startswith('module.') for k in ckpt.keys()):
        state = {k.replace('module.', ''): v for k, v in ckpt.items()}
    else:
        state = ckpt

    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict(state, strict=False)

    model.eval()

    eval_dataset = FreqMedCLIPDataset(data_root, dataset, processor, tokenizer, split='test')
    loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    sigmoid = torch.nn.Sigmoid()

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch['pixel_values'].to(device)
            image_raw = batch['image_raw'].to(device)
            input_ids = batch['input_ids'].to(device)
            masks = batch['mask'].to(device).float()

            preds1, _, _, _ = model(pixel_values, input_ids, image_raw)
            preds = preds1.squeeze(1)  # (B, Hp, Wp)
            probs = sigmoid(preds)

            # Resize probs to mask size
            B = probs.shape[0]
            target_h, target_w = masks.shape[1], masks.shape[2]
            probs_resized = F.interpolate(probs.unsqueeze(1), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(1)
            preds_bin = (probs_resized > 0.5).long()

            # accumulate
            TP += int(((preds_bin == 1) & (masks == 1)).sum().item())
            FP += int(((preds_bin == 1) & (masks == 0)).sum().item())
            FN += int(((preds_bin == 0) & (masks == 1)).sum().item())
            TN += int(((preds_bin == 0) & (masks == 0)).sum().item())

    # compute metrics
    eps = 1e-8
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    iou = TP / (TP + FP + FN + eps)
    dice = (2 * TP) / (2 * TP + FP + FN + eps)

    metrics = {
        'checkpoint': os.path.abspath(checkpoint),
        'dataset': dataset,
        'num_pixels_TP': TP,
        'num_pixels_FP': FP,
        'num_pixels_FN': FN,
        'num_pixels_TN': TN,
        'precision': float(precision),
        'recall': float(recall),
        'iou': float(iou),
        'dice': float(dice)
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print('Saved summary metrics to', out_path)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--data-root', default='../data')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--out', default='./checkpoints/metrics_summary.json')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    compute_metrics(args.checkpoint, args.dataset, args.data_root, args.batch_size, args.out, args.device)
