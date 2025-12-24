"""
Minimal, compatible trainer for FreqMedCLIP that uses the local BiomedCLIP checkpoint.

This script is intentionally lightweight: it loads the local BiomedCLIP from
`D:\\Documents\\LMIS\\MedCLIP-SAMv2\\saliency_maps\\model` and the existing
components in `freqmedclip.scripts`. It trains for a small number of epochs
and saves checkpoints to `checkpoints/<dataset>`.

Usage (example):
    python train_freqmedclip.py --dataset brain_tumors --epochs 2 --batch-size 2

"""
import os
import sys
import argparse
from types import SimpleNamespace
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import AutoModel, AutoProcessor, AutoTokenizer
from tqdm import tqdm
import logging
from datetime import datetime

# Add project root to path so imports from freqmedclip.scripts work
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent))

from freqmedclip.scripts.postprocess import postprocess_saliency_kmeans  # sanity import

# Try to import dataset and wrapper from existing training module if available.
try:
    from freqmedclip.train_freq_fusion import FreqMedCLIPDataset, FrequencyMedCLIPSAMv2, DiceLoss
except Exception:
    FreqMedCLIPDataset = None
    FrequencyMedCLIPSAMv2 = None
    DiceLoss = None

from freqmedclip.scripts.freq_components import FrequencyEncoder, FPNAdapter, haar_dwt
from freqmedclip.scripts.new_freq_encoder import ConvNeXtTiny12Ch


def build_args_namespace(args):
    a = SimpleNamespace()
    a.image_size = args.image_size
    a.text_len = 77
    a.embed_dim = 768
    a.use_iba = False
    a.coarse_downsample = 4
    return a


class SimpleDataset(torch.utils.data.Dataset):
    """Fallback simple dataset if `FreqMedCLIPDataset` import fails.

    Expects structure: data/<dataset>/train_images & train_masks (png/jpg)
    """
    def __init__(self, root_dir, split='train', transform=None):
        self.root = Path(root_dir)
        imgs = list((self.root / f"{split}_images").glob("*.*"))
        masks = list((self.root / f"{split}_masks").glob("*.*"))
        imgs.sort(); masks.sort()
        self.imgs = imgs
        self.masks = masks if masks else [None] * len(imgs)
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.imgs[idx]).convert('RGB')
        mask = None
        if self.masks[idx] is not None:
            mask = Image.open(self.masks[idx]).convert('L')

        sample = { 'image': img, 'mask': mask }
        return sample


def save_checkpoint(model, optimizer, epoch, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'freqmedclip_epoch{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    logging.getLogger('freqmedclip_train').info(f"Saved checkpoint: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data-root', type=str, default=str(Path(__file__).parent.parent / 'data'))
    parser.add_argument('--biomedclip-path', type=str, default=r'D:\\Documents\\LMIS\\MedCLIP-SAMv2\\saliency_maps\\model')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint-dir', type=str, default=str(Path(__file__).parent.parent / 'checkpoints'))

    args = parser.parse_args()
    device = torch.device(args.device)

    # Configure logging: file per run + console
    logs_dir = Path(args.checkpoint_dir) / args.dataset
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f"train_{ts}.log"
    logger = logging.getLogger('freqmedclip_train')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode='a')
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)

    model_path = args.biomedclip_path
    logger.info(f"Loading BiomedCLIP from: {model_path}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    biomedclip = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)

    # Components
    # Use ConvNeXt adapted for 12-channel DWT input (matches train_freq_fusion usage)
    freq_encoder = ConvNeXtTiny12Ch(pretrained=True).to(device)
    fpn_adapter = FPNAdapter(in_channels=768, out_channels=[768, 384, 192, 96]).to(device)

    # Build args for wrapper
    wrapper_args = build_args_namespace(args)

    if FrequencyMedCLIPSAMv2 is None:
        raise RuntimeError('Could not import FrequencyMedCLIPSAMv2 from train_freq_fusion.py; keep that file in place for now.')

    model = FrequencyMedCLIPSAMv2(biomedclip, freq_encoder, fpn_adapter, wrapper_args).to(device)

    # Ensure text_projector input dimension matches BiomedCLIP text hidden size.
    try:
        tokenizer_dummy = tokenizer(["A test prompt"], padding='max_length', truncation=True, max_length=77, return_tensors='pt')
        input_ids_dummy = tokenizer_dummy['input_ids'].to(device)
        with torch.no_grad():
            text_out = biomedclip.text_model(input_ids_dummy, output_hidden_states=True)
        # text_out[0] shape: (B, L, hidden)
        text_hidden_dim = text_out[0].shape[-1]
        proj = getattr(model, 'text_projector', None)
        if proj is not None:
            proj_in = proj.in_features if hasattr(proj, 'in_features') else None
            proj_out = proj.out_features if hasattr(proj, 'out_features') else None
            if proj_in is not None and proj_in != text_hidden_dim:
                # Replace projector to match runtime hidden dim -> keep output dim same
                new_proj = nn.Linear(text_hidden_dim, proj_out if proj_out is not None else text_hidden_dim)
                model.text_projector = new_proj.to(device)
                print(f"Adjusted model.text_projector to in_features={text_hidden_dim}, out_features={proj_out}")
    except Exception as e:
        print("Warning: could not auto-adjust text_projector:", e)
    model.train()

    # Determine ViT expected image size once and frequency raw size (DWT expects double)
    vit_expected_size = None
    try:
        vit_expected_size = getattr(biomedclip.vision_model.config, 'image_size', None)
    except Exception:
        vit_expected_size = None

    if vit_expected_size is None:
        try:
            ip = getattr(processor, 'image_processor', None) or getattr(processor, 'feature_extractor', None)
            if ip is not None:
                size_attr = getattr(ip, 'size', None)
                if isinstance(size_attr, dict):
                    vit_expected_size = size_attr.get('height') or size_attr.get('width')
                elif isinstance(size_attr, int):
                    vit_expected_size = size_attr
        except Exception:
            vit_expected_size = None

    if vit_expected_size is None:
        vit_expected_size = 224

    # Frequency branch expects image_raw at double resolution so DWT halves it
    freq_image_raw_size = vit_expected_size * 2

    # Dataset
    # Note: `FreqMedCLIPDataset` expects (root_dir, dataset_name, ...)
    # so pass `args.data_root` and `args.dataset` separately to avoid
    # constructing a double path like data/brain_tumors/brain_tumors
    dataset_dir = os.path.join(args.data_root, args.dataset)
    logging.getLogger('freqmedclip_train').info(f"Dataset dir (for fallback): {dataset_dir}")
    if FreqMedCLIPDataset is not None:
        dataset = FreqMedCLIPDataset(args.data_root, args.dataset, processor, tokenizer, split='train')
    else:
        dataset = SimpleDataset(dataset_dir, split='train')

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_ckpt_dir = os.path.join(args.checkpoint_dir, args.dataset)

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            # Expect dataset to return processed dict if using FreqMedCLIPDataset
            if isinstance(batch, dict):
                inputs = batch
            else:
                # fallback simple dataset
                imgs = [b['image'] for b in batch]
                proc = processor(images=imgs, return_tensors='pt')
                inputs = {'pixel_values': proc['pixel_values']}

            # Move tensors to device if present
            for k, v in list(inputs.items()):
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            optimizer.zero_grad()
            # Prepare model call arguments: FrequencyMedCLIPSAMv2.forward(pixel_values, input_ids, image_raw)
            # inputs may be a dict from FreqMedCLIPDataset or a fallback SimpleDataset sample list
            if isinstance(batch, dict):
                inputs = batch
            else:
                imgs = [b['image'] for b in batch]
                proc = processor(images=imgs, return_tensors='pt')
                # image_raw: CRITICAL - must have compatible spatial dimensions with pixel_values
                # Provide image_raw at double the ViT size so DWT halves it back.
                image_raw_size = freq_image_raw_size
                image_raw_list = []
                for im in imgs:
                    import numpy as _np
                    arr = _np.array(im.resize((image_raw_size, image_raw_size))).astype('float32') / 255.0
                    # Convert to C,H,W
                    arr = _np.transpose(arr, (2, 0, 1))
                    image_raw_list.append(torch.from_numpy(arr))
                image_raw = torch.stack(image_raw_list, dim=0)
                inputs = {
                    'pixel_values': proc['pixel_values'],
                    'image_raw': image_raw,
                    'input_ids': tokenizer(["A medical image showing an abnormality."] * len(imgs), padding='max_length', truncation=True, max_length=77, return_tensors='pt')['input_ids']
                }

            # Move tensors to device if present and remove mask from call kwargs
            call_kwargs = {}
            for k, v in list(inputs.items()):
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
                # Only include expected args
                if k in ('pixel_values', 'input_ids', 'image_raw'):
                    call_kwargs[k] = inputs[k]

            # Ensure pixel_values match BiomedCLIP expected image size (avoid ValueError)
            pv = call_kwargs['pixel_values']
            # pixel_values shape: (B, C, H, W)
            h = pv.shape[-2]
            w = pv.shape[-1]
            if h != vit_expected_size or w != vit_expected_size:
                pv = nn.functional.interpolate(pv, size=(vit_expected_size, vit_expected_size), mode='bilinear', align_corners=False)
                call_kwargs['pixel_values'] = pv

            # Call model with explicit args
            try:
                out = model(call_kwargs['pixel_values'], call_kwargs['input_ids'], call_kwargs['image_raw'])
            except Exception as e:
                print('Model forward error:', e)
                raise

            # Model returns tuple: (out, out_2, img_feats_pooled, text_feats_pooled)
            if isinstance(out, tuple) or isinstance(out, list):
                pred = out[0]
            elif isinstance(out, dict):
                pred = out.get('pred', next(iter(out.values())))
            else:
                pred = out

            # For supervised training we need a mask; try to extract
            target = None
            if 'mask' in inputs:
                target = inputs['mask'].float().to(device)

            if target is None:
                print('No target masks available in dataset; skipping backward (dry-run).')
                break

            # preds are logits possibly with shape (B, 1, H, W)
            if pred.dim() == 4 and pred.shape[1] == 1:
                pred_tensor = pred.squeeze(1)
            else:
                pred_tensor = pred

            # Resize pred/target to same shape if necessary
            if pred_tensor.shape != target.shape:
                pred_tensor = nn.functional.interpolate(pred_tensor.unsqueeze(1), size=target.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)

            loss = criterion(pred_tensor, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch {epoch}/{args.epochs} - avg_loss: {avg_loss:.4f}")
        save_checkpoint(model, optimizer, epoch, out_ckpt_dir)


if __name__ == '__main__':
    main()
