import os
import sys
from pathlib import Path as _Path
# Ensure repo root is on sys.path so local modules can be imported
repo_root = str(_Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
import yaml
import csv
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Load model & training code from repo
from train_frequency_aware_early_fusion import FreqAwareModel
from transformers import AutoTokenizer, AutoModel


def read_config(path='config_train.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_samples(csv_path):
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


def resolve_paths(samples, csv_path):
    csvp = Path(csv_path)
    ds_root = csvp.parent
    try:
        repo_root = csvp.parents[2]
    except Exception:
        repo_root = ds_root.parent

    resolved = []
    for caption, raw in samples:
        p = Path(raw)
        candidates = []
        if p.is_absolute():
            candidates.append(p)
        else:
            if 'data/medpix_dataset' in raw:
                candidates.append(repo_root / raw)
            candidates.append(ds_root / raw)
            candidates.append(ds_root / 'images' / Path(raw).name)

        chosen = None
        for c in candidates:
            if c.exists():
                chosen = c
                break
        if chosen is None:
            basename = Path(raw).name
            images_dir = ds_root / 'images'
            if (images_dir / basename).exists():
                chosen = images_dir / basename
            else:
                try:
                    found = next(images_dir.rglob(basename))
                    chosen = found
                except StopIteration:
                    chosen = None

        if chosen is not None:
            resolved.append((caption, str(chosen.resolve())))
    return resolved


def split_samples(samples, val_ratio=0.1, seed=42):
    import random
    rng = random.Random(seed)
    rng.shuffle(samples)
    n = max(1, int(len(samples) * val_ratio))
    val = samples[:n]
    train = samples[n:]
    return train, val


def make_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])


def compute_embeddings(model, tokenizer, samples, device, batch_size=32, image_size=224):
    transform = make_transform(image_size)
    imgs = []
    texts = []
    for caption, img_path in samples:
        imgs.append((img_path, transform(Image.open(img_path).convert('RGB'))))
        texts.append(caption)

    image_features = []
    text_features = []

    model.to(device)
    model.eval()

    # Images
    with torch.no_grad():
        for i in range(0, len(imgs), batch_size):
            batch = imgs[i:i+batch_size]
            imgs_t = torch.stack([t for _, t in batch]).to(device)
            feats = model.get_fused_features(imgs_t)
            feats = F.normalize(feats, dim=-1).cpu()
            image_features.append(feats)

    # Texts
    toks = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors='pt')
    input_ids = toks['input_ids']
    attn = toks['attention_mask']
    with torch.no_grad():
        for i in range(0, input_ids.shape[0], batch_size):
            ids = input_ids[i:i+batch_size].to(device)
            mask = attn[i:i+batch_size].to(device)
            out = model.get_text_features({'input_ids': ids, 'attention_mask': mask})
            out = F.normalize(out, dim=-1).cpu()
            text_features.append(out)

    image_features = torch.cat(image_features, dim=0)
    text_features = torch.cat(text_features, dim=0)
    return image_features, text_features


def recall_at_k(sim, ks=(1,5,10)):
    # sim: [N_images, N_texts]
    sims = sim.numpy()
    import numpy as np
    N = sims.shape[0]
    ranks = np.argsort(-sims, axis=1)
    res = {}
    for k in ks:
        correct = 0
        for i in range(N):
            if i in ranks[i, :k]:
                correct += 1
        res[f'recall@{k}'] = correct / N
    return res


def main():
    cfg = read_config('config_train.yaml')
    csv_path = cfg.get('csv_path')
    val_ratio = cfg.get('val_ratio', 0.1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    samples = build_samples(csv_path)
    resolved = resolve_paths(samples, csv_path)
    train_s, val_s = split_samples(resolved, val_ratio=val_ratio)
    print(f"Using {len(val_s)} validation samples")

    tokenizer = AutoTokenizer.from_pretrained(cfg.get('tokenizer_name', 'chuhac/BiomedCLIP-vit-bert-hf'), trust_remote_code=True)

    # Load original model: BiomedCLIP from saliency_maps/model but wrapped as FreqAwareModel with fusion disabled
    orig_model = FreqAwareModel(cfg, device=device)
    # Replace internal biomedclip with the original checkpoint
    orig_biomed = AutoModel.from_pretrained(str(Path('saliency_maps/model').absolute()), trust_remote_code=True)
    orig_model.biomedclip = orig_biomed.to(device)
    # disable fusion
    try:
        orig_model.fusion_gate.fusion_alpha = torch.tensor(0.0, device=device)
    except Exception:
        pass
    # zero high freq proj weights
    for p in orig_model.high_freq_proj.parameters():
        p.data.zero_()

    # Load new model from checkpoint (stage3_best.pth)
    new_model = FreqAwareModel(cfg, device=device)
    ckpt_path = Path('checkpoints/early_fusion/stage3_best.pth')
    if not ckpt_path.exists():
        ckpt_path = Path('checkpoints/early_fusion/stage1_best.pth')
    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    model_sd = ckpt.get('model_state_dict', ckpt)
    new_model.load_state_dict(model_sd, strict=False)
    new_model.to(device)
    new_model.eval()

    # Compute embeddings
    print('Computing embeddings for original model...')
    orig_img_feats, orig_text_feats = compute_embeddings(orig_model, tokenizer, val_s, device, batch_size=32, image_size=cfg.get('image_size', 224))
    print('Computing embeddings for new model...')
    new_img_feats, new_text_feats = compute_embeddings(new_model, tokenizer, val_s, device, batch_size=32, image_size=cfg.get('image_size', 224))

    # Similarities
    print('Computing similarities and recall...')
    sim_orig = orig_img_feats @ orig_text_feats.t()
    sim_new = new_img_feats @ new_text_feats.t()

    metrics = {
        'original': recall_at_k(sim_orig, ks=(1,5,10)),
        'new': recall_at_k(sim_new, ks=(1,5,10))
    }

    out_dir = Path('results')
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / 'retrieval_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print('Done. Metrics:')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
