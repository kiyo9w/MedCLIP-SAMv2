import os
import sys
import argparse
import random
import json
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from train_freq_fusion import FreqMedCLIPDataset, FrequencyMedCLIPSAMv2, ConvNeXtTiny12Ch, FPNAdapter
from transformers import AutoProcessor, AutoTokenizer, AutoModel


def save_overlay(orig, gt_mask, pred_mask, out_path):
    # orig: HWC uint8
    # gt_mask, pred_mask: HxW binary (0/1)
    orig_img = Image.fromarray(orig)
    gt = Image.fromarray((gt_mask * 255).astype(np.uint8)).convert('L')
    pred = Image.fromarray((pred_mask * 255).astype(np.uint8)).convert('L')

    # Create color overlays
    overlay = Image.new('RGB', orig_img.size)
    overlay.paste(orig_img)

    # Blend predicted mask in red and gt in green side-by-side
    w, h = orig_img.size
    canvas = Image.new('RGB', (w * 3, h))
    canvas.paste(orig_img, (0, 0))

    gt_col = Image.new('RGB', (w, h), (0, 0, 0))
    pred_col = Image.new('RGB', (w, h), (0, 0, 0))

    gt_arr = np.array(gt)
    pred_arr = np.array(pred)

    gt_col_arr = np.zeros((h, w, 3), dtype=np.uint8)
    pred_col_arr = np.zeros((h, w, 3), dtype=np.uint8)

    gt_col_arr[gt_arr > 127] = [0, 255, 0]
    pred_col_arr[pred_arr > 127] = [255, 0, 0]

    gt_col = Image.fromarray(gt_col_arr)
    pred_col = Image.fromarray(pred_col_arr)

    # overlay predicted on original
    blended_pred = Image.blend(orig_img, pred_col, alpha=0.5)
    blended_gt = Image.blend(orig_img, gt_col, alpha=0.5)

    canvas.paste(blended_gt, (w, 0))
    canvas.paste(blended_pred, (w * 2, 0))

    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--data-root', default='../data')
    parser.add_argument('--out-dir', default='./checkpoints/brain_tumors/visualizations_epoch48')
    parser.add_argument('--sample-fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--split', default='val', choices=['train','val','test'], help='Which split to visualize')
    args = parser.parse_args()

    device = torch.device(args.device)
    model_name = os.path.join(os.path.dirname(__file__), '..', 'saliency_maps', 'model')
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    biomedclip = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

    fpn_adapter = FPNAdapter(in_channels=768, out_channels=[768,384,192,96]).to(device)
    freq_encoder = ConvNeXtTiny12Ch(pretrained=False).to(device)
    model = FrequencyMedCLIPSAMv2(biomedclip, freq_encoder, fpn_adapter, args=argparse.Namespace()).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    # unwrap possible wrapper
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

    val_dataset = FreqMedCLIPDataset(args.data_root, args.dataset, processor, tokenizer, split=args.split)
    N = len(val_dataset)
    k = max(1, int(N * args.sample_fraction))
    random.seed(args.seed)
    indices = random.sample(range(N), k)

    os.makedirs(args.out_dir, exist_ok=True)

    sigmoid = torch.nn.Sigmoid()

    with torch.no_grad():
        for idx in indices:
            item = val_dataset[idx]
            pixel_values = item['pixel_values'].unsqueeze(0).to(device)
            image_raw = item['image_raw'].unsqueeze(0).to(device)
            input_ids = item['input_ids'].unsqueeze(0).to(device)
            mask = item['mask']

            preds1, preds2, _, _ = model(pixel_values, input_ids, image_raw)
            pred = preds1.squeeze(0).squeeze(0)  # Hp, Wp
            probs = sigmoid(pred.unsqueeze(0)).squeeze(0)

            # resize to mask size
            target_h, target_w = mask.shape[0], mask.shape[1]
            probs_resized = F.interpolate(probs.unsqueeze(0).unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze()
            pred_bin = (probs_resized > 0.5).cpu().numpy().astype(np.uint8)

            # original image
            img_raw = item['image_raw'].permute(1,2,0).cpu().numpy()
            img_uint8 = (np.clip(img_raw, 0, 1) * 255).astype(np.uint8)

            gt_mask = mask.cpu().numpy().astype(np.uint8)

            out_name = os.path.splitext(item['img_name'])[0] + '_viz.png'
            out_path = os.path.join(args.out_dir, out_name)
            save_overlay(img_uint8, gt_mask, pred_bin, out_path)

    print(f"Saved {k} visualizations to {args.out_dir}")

if __name__ == '__main__':
    main()
