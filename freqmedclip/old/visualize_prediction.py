import os
import random
import torch
import torch.nn.functional as F
import numpy as np
# Use Agg backend to avoid GUI/OpenMP issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Allow duplicate OpenMP libs (common on Windows)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer

# Add project root to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from freqmedclip.scripts.freq_components import SmartFusionBlock, DWTForward
from freqmedclip.scripts.postprocess import postprocess_saliency_kmeans, postprocess_saliency_threshold
from freqmedclip.train_freq_fusion import FreqMedCLIPDataset, FrequencyMedCLIPSAMv2


def load_model(checkpoint_path, device):
    # Load BiomedCLIP
    model_name = "chuhac/BiomedCLIP-vit-bert-hf"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    biomedclip = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

    # Initialize components
    dwt = DWTForward().to(device)
    fusion = SmartFusionBlock(hf_channels=777, lf_channels=1, out_channels=32).to(device)
    # Load checkpoint (fusion weights only)
    fusion.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # Wrap model
    class Args: pass
    args = Args()
    model = FrequencyMedCLIPSAMv2(biomedclip, fusion, dwt, args).to(device)
    model.eval()
    return model, processor, tokenizer


def visualize_sample(sample_idx=0, checkpoint_path='checkpoints/breast_tumors/fusion_breast_tumors_epoch100.pth', output_dir='visualizations'):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, processor, tokenizer = load_model(checkpoint_path, device)

    # Load dataset (test split)
    dataset = FreqMedCLIPDataset('data', 'breast_tumors', processor, tokenizer, split='test')
    if sample_idx >= len(dataset):
        sample_idx = random.randint(0, len(dataset)-1)
    sample = dataset[sample_idx]
    img_name = sample['img_name']
    pixel_values = sample['pixel_values'].unsqueeze(0).to(device)
    input_ids = sample['input_ids'].unsqueeze(0).to(device)
    gt_mask = sample['mask'].unsqueeze(0).to(device).float()

    # Forward pass with intermediate outputs
    with torch.no_grad():
        # Vision features
        vision_outputs = model.biomedclip.vision_model(pixel_values, output_hidden_states=True)
        last_hidden_state = vision_outputs.last_hidden_state
        # Text features (tuple output)
        text_outputs = model.biomedclip.text_model(input_ids, output_hidden_states=True)
        text_embeds = text_outputs[1] if isinstance(text_outputs, tuple) else text_outputs.pooler_output
        # Coarse map (dot product)
        patch_embeddings = last_hidden_state[:, 1:, :]
        patch_embeddings = F.normalize(patch_embeddings, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        coarse_map_flat = torch.bmm(patch_embeddings, text_embeds.unsqueeze(-1))
        H_feat = W_feat = int(np.sqrt(patch_embeddings.shape[1]))
        coarse_map = coarse_map_flat.view(-1, 1, H_feat, W_feat)  # (1,1,H,W)
        # HF features
        dwt_feats = model.dwt_module(pixel_values)  # (1,9,112,112)
        shallow_feats = vision_outputs.hidden_states[3][:, 1:, :]  # (1,196,768)
        shallow_feats = shallow_feats.permute(0,2,1).view(-1,768,H_feat,W_feat)  # (1,768,14,14)
        shallow_up = F.interpolate(shallow_feats, size=(112,112), mode='bilinear', align_corners=False)
        hf_features = torch.cat([shallow_up, dwt_feats], dim=1)  # (1,777,112,112)
        # Fusion (fine map)
        fine_map = model.fusion_block(hf_features, coarse_map)  # (1,1,112,112)
        # Upsample to original size (224x224)
        pred_mask = F.interpolate(fine_map, size=(224,224), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)  # (224,224)

    # Convert tensors to numpy for plotting
    img = Image.open(os.path.join('data', 'breast_tumors', 'test_images', img_name)).convert('RGB')
    img_np = np.array(img)
    gt_np = gt_mask.squeeze().cpu().numpy()
    pred_np = pred_mask.cpu().numpy()
    pred_binary = (torch.sigmoid(pred_mask) > 0.5).float().cpu().numpy()
    coarse_np = coarse_map.squeeze().cpu().numpy()
    # Resize coarse map to 224 for visualization
    coarse_resized = F.interpolate(coarse_map, size=(224,224), mode='bilinear', align_corners=False).squeeze().cpu().numpy()

    # ===== POSTPROCESSING (KMeans - như MedCLIP-SAMv2 gốc) =====
    # Normalize pred_np to [0, 1] for postprocessing
    pred_normalized = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)
    # Apply KMeans postprocessing
    pred_cleaned = postprocess_saliency_kmeans(pred_normalized, num_clusters=2, top_k_components=1)
    
    # Plotting (3x3 grid: add postprocessing results)
    fig, axs = plt.subplots(3, 3, figsize=(18,16))
    
    # Row 1: Original, GT, Coarse
    axs[0,0].imshow(img_np)
    axs[0,0].set_title('Original Image', fontsize=14, fontweight='bold')
    axs[0,0].axis('off')
    axs[0,1].imshow(gt_np, cmap='gray')
    axs[0,1].set_title('Ground Truth Mask', fontsize=14, fontweight='bold')
    axs[0,1].axis('off')
    axs[0,2].imshow(coarse_resized, cmap='viridis')
    axs[0,2].set_title('Coarse Map (Text-Guided)', fontsize=14, fontweight='bold')
    axs[0,2].axis('off')
    
    # Row 2: Fine Raw, Fine Binary, Fine Cleaned
    axs[1,0].imshow(pred_np, cmap='viridis')
    axs[1,0].set_title('Fine Saliency (Raw)', fontsize=14, fontweight='bold')
    axs[1,0].axis('off')
    axs[1,1].imshow(pred_binary, cmap='gray')
    axs[1,1].set_title('Fine Binary (Threshold=0.5)', fontsize=14, fontweight='bold')
    axs[1,1].axis('off')
    axs[1,2].imshow(pred_cleaned, cmap='gray')
    axs[1,2].set_title('Fine Cleaned (KMeans)', fontsize=14, fontweight='bold', color='red')
    axs[1,2].axis('off')
    
    # Row 3: Overlays for comparison
    # Overlay: GT on Image
    overlay_gt = img_np.copy()
    overlay_gt[gt_np > 0] = [255, 0, 0]  # Red for GT
    axs[2,0].imshow(overlay_gt)
    axs[2,0].set_title('GT Overlay', fontsize=14)
    axs[2,0].axis('off')
    
    # Overlay: Binary on Image
    overlay_binary = img_np.copy()
    overlay_binary[pred_binary > 0] = [0, 255, 0]  # Green for prediction
    axs[2,1].imshow(overlay_binary)
    axs[2,1].set_title('Binary Overlay', fontsize=14)
    axs[2,1].axis('off')
    
    # Overlay: Cleaned on Image
    overlay_cleaned = img_np.copy()
    overlay_cleaned[pred_cleaned > 0] = [0, 0, 255]  # Blue for cleaned
    axs[2,2].imshow(overlay_cleaned)
    axs[2,2].set_title('Cleaned Overlay (Recommended)', fontsize=14, fontweight='bold', color='blue')
    axs[2,2].axis('off')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"visual_{img_name.replace('.','_')}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Visualization saved: {out_path}")

if __name__ == '__main__':
    # Load BiomedCLIP processor and tokenizer for the dataset
    model_name = "chuhac/BiomedCLIP-vit-bert-hf"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Create test dataset
    dataset = FreqMedCLIPDataset('data', 'breast_tumors', processor, tokenizer, split='test')
    print(f"Generating visualizations for {len(dataset)} test samples...")
    for idx in range(len(dataset)):
        visualize_sample(sample_idx=idx)
    print("All visualizations saved in 'visualizations/'")

