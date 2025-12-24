"""
Zero-shot saliency + SAM pipeline runner

Usage examples:
  # Prepare a small JSON mapping (see below) and run:
  python scripts/zero_shot_pipeline.py --csv config_train.yaml --mode original --out results/zeroshot/original --max-samples 50
  python scripts/zero_shot_pipeline.py --csv config_train.yaml --mode stage1  --out results/zeroshot/stage1  --max-samples 50

This script:
 - reads the CSV defined in `config_train.yaml`
 - builds a small mapping JSON of caption -> image
 - loads either the HF original BiomedCLIP or the repo local model and injects Stage-1 checkpoint when `--mode stage1`
 - generates saliency maps using `vision_heatmap_freq_aware` from `scripts.methods`
 - calls postprocessing and SAM prompt scripts on the outputs

Note: Running full datasets and SAM may be slow. Use `--max-samples` to limit.
"""

import argparse
import json
import os
from pathlib import Path
import csv
import random
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoProcessor, AutoTokenizer
import sys
# Ensure repo root on sys.path so `scripts.*` imports work when running this file directly
repo_root = Path(__file__).resolve().parents[1]

# Prefer artifacts inside MedCLIP-SAMv2-finetune when present. Insert finetune paths
# at the front of sys.path so imports resolve to the finetune copies first.
finetune_root = repo_root / 'MedCLIP-SAMv2-finetune'
FINETUNE_AVAILABLE = False
if finetune_root.exists():
    FINETUNE_AVAILABLE = True
    finetune_root = finetune_root.resolve()
    fin_saliency = str(finetune_root / 'saliency_maps')
    # insert finetune saliency first
    if fin_saliency not in sys.path:
        sys.path.insert(0, fin_saliency)
    if str(finetune_root) not in sys.path:
        sys.path.insert(0, str(finetune_root))

# Next ensure repo-level imports are available but do NOT shadow the finetune folder.
repo_root_str = str(repo_root.resolve())
if repo_root_str not in sys.path:
    # put repo root after finetune entries
    if FINETUNE_AVAILABLE and len(sys.path) > 0:
        sys.path.insert(1, repo_root_str)
    else:
        sys.path.insert(0, repo_root_str)

# Also add repository's saliency_maps as fallback (after finetune if present)
saliency_pkg = str(repo_root / 'saliency_maps')
if saliency_pkg not in sys.path:
    if FINETUNE_AVAILABLE and len(sys.path) > 1:
        sys.path.insert(2, saliency_pkg)
    else:
        sys.path.insert(0, saliency_pkg)

# Default paths (prefer finetune bundle when available)
DEFAULT_STAGE1_CKPT = str(finetune_root / 'checkpoints' / 'early_fusion' / 'stage1_best.pth') if FINETUNE_AVAILABLE else str(repo_root / 'checkpoints' / 'early_fusion' / 'stage1_best.pth')
POSTPROCESS_SCRIPT = str(finetune_root / 'postprocessing' / 'postprocess_saliency_maps.py') if FINETUNE_AVAILABLE else str(repo_root / 'postprocessing' / 'postprocess_saliency_maps.py')
SAM_PROMPT_SCRIPT = str(finetune_root / 'segment-anything' / 'prompt_sam.py') if FINETUNE_AVAILABLE else str(repo_root / 'segment-anything' / 'prompt_sam.py')
SAM_CHECKPOINT = str(finetune_root / 'segment-anything' / 'sam_checkpoints' / 'sam_vit_h_4b8939.pth') if FINETUNE_AVAILABLE else str(repo_root / 'segment-anything' / 'sam_checkpoints' / 'sam_vit_h_4b8939.pth')
import importlib.util

# Try to import the expected symbols from the active saliency_maps package; if missing,
# fall back to loading the original repo copies by file path to preserve API.
try:
    from saliency_maps.scripts.methods import vision_heatmap_freq_aware
except Exception:
    # fall back to repo copy
    repo_methods = Path(repo_root) / 'saliency_maps' / 'scripts' / 'methods.py'
    repo_freq = Path(repo_root) / 'saliency_maps' / 'scripts' / 'freq_components.py'
    repo_iba = Path(repo_root) / 'saliency_maps' / 'scripts' / 'iba.py'
    if repo_methods.exists():
        # load dependent modules into sys.modules under the expected top-level package name 'scripts'
        if repo_freq.exists():
            spec_f = importlib.util.spec_from_file_location('scripts.freq_components', str(repo_freq))
            freq_mod = importlib.util.module_from_spec(spec_f)
            sys.modules['scripts.freq_components'] = freq_mod
            spec_f.loader.exec_module(freq_mod)
        if repo_iba.exists():
            spec_iba = importlib.util.spec_from_file_location('scripts.iba', str(repo_iba))
            iba_mod = importlib.util.module_from_spec(spec_iba)
            sys.modules['scripts.iba'] = iba_mod
            spec_iba.loader.exec_module(iba_mod)

        spec = importlib.util.spec_from_file_location('repo_saliency_methods', str(repo_methods))
        repo_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(repo_mod)
        vision_heatmap_freq_aware = getattr(repo_mod, 'vision_heatmap_freq_aware', None)
    else:
        raise

try:
    from saliency_maps.scripts.freq_components import SmartFusionBlock, DWTForward
except Exception:
    repo_freq = Path(repo_root) / 'saliency_maps' / 'scripts' / 'freq_components.py'
    if repo_freq.exists():
        spec = importlib.util.spec_from_file_location('scripts.freq_components', str(repo_freq))
        repo_freq_mod = importlib.util.module_from_spec(spec)
        # ensure the module is accessible under the name scripts.freq_components
        sys.modules['scripts.freq_components'] = repo_freq_mod
        spec.loader.exec_module(repo_freq_mod)
        SmartFusionBlock = getattr(repo_freq_mod, 'SmartFusionBlock', None)
        DWTForward = getattr(repo_freq_mod, 'DWTForward', None)
    else:
        raise


def read_config(path='config_train.yaml'):
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_samples_from_csv(csv_path):
    csvp = Path(csv_path)
    samples = []
    with csvp.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            caption = row.get('Caption') or row.get('caption')
            raw = row.get('filename') or row.get('file') or row.get('image')
            if not raw or not caption:
                continue
            raw = raw.replace('\\', '/').replace('data/data/medpix_dataset', 'data/medpix_dataset')
            samples.append((caption, raw))
    return samples


def build_samples_from_json(mapping_json, images_dir):
    p = Path(mapping_json)
    imgs = Path(images_dir)
    samples = []
    with p.open('r', encoding='utf-8') as f:
        mapping = json.load(f)
    for fname, caption in mapping.items():
        # try several candidate locations
        candidates = [imgs / fname, imgs / 'images' / fname, Path(fname)]
        chosen = None
        for c in candidates:
            if c.exists():
                chosen = str(c.resolve())
                break
        if chosen is None:
            # try recursive search
            try:
                found = next(imgs.rglob(fname))
                chosen = str(found.resolve())
            except StopIteration:
                chosen = None
        if chosen is not None:
            samples.append((caption, chosen))
    return samples


def resolve_image_path(raw, csv_path):
    csvp = Path(csv_path)
    ds_root = csvp.parent
    p = Path(raw)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        if 'data/medpix_dataset' in raw:
            candidates.append(Path('.') / raw)
        candidates.append(ds_root / raw)
        candidates.append(ds_root / 'images' / Path(raw).name)
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    # fallback: rglob
    images_dir = ds_root / 'images'
    try:
        found = next(images_dir.rglob(Path(raw).name))
        return str(found.resolve())
    except StopIteration:
        return None


def make_transform(processor):
    # generator uses AutoProcessor to produce pixel_values; keep behaviour similar
    def _transform(img):
        return processor(images=img, return_tensors='pt')['pixel_values']
    return _transform


def run_saliency(mode, samples, cfg, out_dir, device, max_samples=None, stage1_ckpt=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if stage1_ckpt is None:
        stage1_ckpt = DEFAULT_STAGE1_CKPT
    if mode == 'original':
        # prefer local copy shipped with repo
        local = Path('saliency_maps/model')
        if local.exists():
            model = AutoModel.from_pretrained(str(local.absolute()), trust_remote_code=True).to(device)
            processor = AutoProcessor.from_pretrained('chuhac/BiomedCLIP-vit-bert-hf', trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained('chuhac/BiomedCLIP-vit-bert-hf', trust_remote_code=True)
        else:
            model = AutoModel.from_pretrained('chuhac/BiomedCLIP-vit-bert-hf', trust_remote_code=True).to(device)
            processor = AutoProcessor.from_pretrained('chuhac/BiomedCLIP-vit-bert-hf', trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained('chuhac/BiomedCLIP-vit-bert-hf', trust_remote_code=True)
    elif mode == 'stage1':
        # load repo model then try to inject stage1 checkpoint
        model = AutoModel.from_pretrained('./saliency_maps/model', trust_remote_code=True).to(device)
        processor = AutoProcessor.from_pretrained('chuhac/BiomedCLIP-vit-bert-hf', trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained('chuhac/BiomedCLIP-vit-bert-hf', trust_remote_code=True)
        ckpt_p = Path(stage1_ckpt)
        if ckpt_p.exists():
            ckpt = torch.load(str(ckpt_p), map_location='cpu')
            sd = ckpt.get('model_state_dict', ckpt)
            try:
                model.load_state_dict(sd, strict=False)
                print(f"Loaded Stage-1 weights from {ckpt_p}")
            except Exception as e:
                print(f"Warning: could not fully load stage1 state_dict: {e}")
        else:
            print(f"Warning: stage1 checkpoint not found at {ckpt_p}; continuing with repo model")
    else:
        raise ValueError('mode must be original or stage1')

    # DWT module
    dwt_module = DWTForward().to(device)

    # Create simple cross-attention wrapper, attn projection, and shallow fusion fallback
    class CrossAttnWrapper(nn.Module):
        def __init__(self, embed_dim=768, num_heads=12):
            super().__init__()
            # use batch_first=True to accept (B, Seq, Dim)
            self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        def forward(self, query, key, value, **kwargs):
            return self.mha(query, key, value)

    cross_attn = CrossAttnWrapper(embed_dim=cfg.get('embedding_dim', 768), num_heads=12).to(device)
    attn_proj = nn.Linear(cfg.get('embedding_dim', 768), 1).to(device)

    class ShallowFusionFallback(nn.Module):
        def __init__(self, in_ch, out_ch=32):
            super().__init__()
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        def forward(self, dwt_feats, early_feats):
            # dwt_feats: (B, C', H, W)
            # project to out_ch
            x = self.proj(dwt_feats)
            return x

    # infer DWT output channels by running a tiny tensor on same device
    dwt_sample = torch.zeros(1,3,32,32).to(device)
    with torch.no_grad():
        dwt_out = dwt_module(dwt_sample)
    in_ch = dwt_out.shape[1]
    shallow_fusion = ShallowFusionFallback(in_ch=in_ch, out_ch=in_ch).to(device)

    # Now create fusion block expecting HF channels = in_ch
    fusion_block = SmartFusionBlock(hf_channels=in_ch, lf_channels=1, out_channels=32).to(device)
    fusion_block.eval()

    transform = lambda img: processor(images=img, return_tensors='pt')['pixel_values']

    n = 0
    for caption, img_path in tqdm(samples):
        if max_samples is not None and n >= max_samples:
            break
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Skip {img_path}: {e}")
            continue

        pixel_values = processor(images=img, return_tensors='pt')['pixel_values'].to(device)
        text_ids = torch.tensor([tokenizer.encode(caption, add_special_tokens=True)]).to(device)

        with torch.no_grad():
            vmap = vision_heatmap_freq_aware(
                text_ids,
                pixel_values,
                model,
                7,
                0.1,
                1.0,
                fusion_block,
                cross_attn,
                attn_proj,
                shallow_fusion,
                dwt_module,
                ensemble=False,
                progbar=False
            )

        img_np = np.array(img)
        vmap_resized = np.array(vmap)
        vmap_resized = np.clip(vmap_resized, 0.0, 1.0)
        vmap_resized = (vmap_resized * 255).astype('uint8')

        out_path = out_dir / Path(img_path).name
        from cv2 import imwrite
        imwrite(str(out_path), vmap_resized)
        n += 1

    print(f"Wrote {n} saliency maps to {out_dir}")
    return out_dir


def call_postprocess_and_sam(saliency_dir, coarse_out_dir, sam_out_dir, dataset_images_dir):
    # run postprocessing script then SAM prompt runner
    import subprocess
    coarse_out_dir = Path(coarse_out_dir)
    coarse_out_dir.mkdir(parents=True, exist_ok=True)
    sam_out_dir = Path(sam_out_dir)
    sam_out_dir.mkdir(parents=True, exist_ok=True)
    # Try local kmeans postprocessing first (avoids pydensecrf dependency)
    try:
        from sklearn.cluster import KMeans
        import cv2
        print('Running local kmeans postprocessing...')
        files = list(Path(saliency_dir).iterdir())
        for f in files:
            try:
                sal = cv2.imread(str(f), 0).astype('float32') / 255.0
                h, w = sal.shape
                img = cv2.resize(sal, (256, 256), interpolation=cv2.INTER_NEAREST)
                flat = img.reshape(-1, 1)
                kmeans = KMeans(n_clusters=2, random_state=10).fit_predict(flat)
                labels = kmeans.reshape(256, 256)
                centroids = None
                # decide background cluster as one with lower mean
                # compute centroids quickly
                # reconstruct mask
                c0_mean = flat[kmeans == 0].mean() if (kmeans == 0).any() else 0
                c1_mean = flat[kmeans == 1].mean() if (kmeans == 1).any() else 1
                background = 0 if c0_mean < c1_mean else 1
                mask = (labels != background).astype('uint8') * 255
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                outp = Path(coarse_out_dir) / f.name
                cv2.imwrite(str(outp), mask)
            except Exception as e:
                print(f"Local postprocess failed for {f}: {e}")
        print('Local postprocessing done.')
    except Exception as e:
        print(f"Local kmeans postprocessing not available: {e}. Falling back to external postprocess script.")
        # postprocess via external script
        cmd_post = [
            sys.executable, POSTPROCESS_SCRIPT,
            '--input-path', str(dataset_images_dir),
            '--output-path', str(coarse_out_dir),
            '--sal-path', str(saliency_dir),
            '--postprocess', 'kmeans',
            '--filter'
        ]
        print('Running postprocessing:', ' '.join(cmd_post))
        try:
            subprocess.run(cmd_post, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Postprocessing failed: {e}. Skipping postprocessing/SAM for this run.")
            return

    # call SAM prompts
    cmd_sam = [
        sys.executable, SAM_PROMPT_SCRIPT,
        '--input', str(dataset_images_dir),
        '--mask-input', str(coarse_out_dir),
        '--output', str(sam_out_dir),
        '--model-type', 'vit_h',
        '--checkpoint', SAM_CHECKPOINT,
        '--prompts', 'boxes'
    ]
    print('Running SAM prompt script:', ' '.join(cmd_sam))
    try:
        subprocess.run(cmd_sam, check=True)
    except subprocess.CalledProcessError as e:
        print(f"SAM prompt script failed: {e}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='config_train.yaml')
    parser.add_argument('--mapping-json', type=str, default=None, help='Optional JSON mapping file (filename->caption)')
    parser.add_argument('--mode', choices=['original','stage1'], required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--max-samples', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--stage1-ckpt', type=str, default='checkpoints/early_fusion/stage1_best.pth')
    args = parser.parse_args()

    cfg = read_config(args.csv)
    csv_path = cfg.get('csv_path')
    if args.mapping_json is not None:
        # mapping_json provided: use it with dataset folder adjacent to mapping file or specified images dir
        # assume mapping_json is in saliency_maps/text_prompts or similar and images dir is provided in same folder hierarchy
        # The user will run with --mapping-json and input dataset dir as part of invocation via --out base
        # Here we try to infer images_dir from mapping location's parent or from standard dataset locations
        mapping_path = Path(args.mapping_json)
        # try to infer images_dir from mapping filename: look for 'brain_tumors' or 'breast_tumors'
        images_dir_guess = Path('data')
        # default: look for data/brain_tumors and data/breast_tumors
        samples_all = build_samples_from_json(str(mapping_path), images_dir_guess)
        resolved = samples_all
    else:
        samples_all = build_samples_from_csv(csv_path)

        resolved = []
        for cap, raw in samples_all:
            p = resolve_image_path(raw, csv_path)
            if p is not None:
                resolved.append((cap, p))

    random.seed(42)
    random.shuffle(resolved)
    samples = resolved[:args.max_samples]

    out_dir = run_saliency(args.mode, samples, cfg, args.out, args.device, max_samples=args.max_samples, stage1_ckpt=args.stage1_ckpt)

    # postprocess and run SAM (assume dataset images dir is parent of csv images folder)
    csvp = Path(csv_path)
    dataset_images_dir = csvp.parent / 'images'
    coarse_dir = str(Path(args.out).parent / (Path(args.out).name + '_coarse'))
    sam_dir = str(Path(args.out).parent / (Path(args.out).name + '_sam'))

    call_postprocess_and_sam(out_dir, coarse_dir, sam_dir, dataset_images_dir)
