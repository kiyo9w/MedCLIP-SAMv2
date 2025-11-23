"""
Based on code of https://github.com/bazingagin/IBA, https://github.com/BioroboticsLab/IBA

TGCAM Implementation:
- Replaces unidirectional IBA/M2IB bottleneck with symmetric attention (CAM)
- Adds iterative text enhancement (ITEM) for hierarchical feature refinement
"""

from scripts.iba import IBAInterpreter, Estimator
from scripts.tgcam_components import TGCAMPipeline
from scripts.utils import normalize
import numpy as np
# import clip
# import copy
import torch 
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast

# Feature Map is the output of a certain layer given X
def extract_feature_map(model, layer_idx, x):
    with torch.no_grad():
        states = model(x, output_hidden_states=True) 
        feature = states['hidden_states'][layer_idx+1] # +1 because the first output is embedding 
        return feature
    
def extract_text_feature_map(model, layer_idx, x):
    with torch.no_grad():
        states = model(x, output_hidden_states=True) 
        feature = states[0] # +1 because the first output is embedding 
        return feature

# Extract BERT Layer
def extract_bert_layer(model, layer_idx):
    desired_layer = ''
    for _, submodule in model.named_children():
        for n, s in submodule.named_children():
            if n == 'layers' or n == 'resblocks':
                for n2, s2 in s.named_children():
                    if n2 == str(layer_idx):
                        desired_layer = s2
                        return desired_layer

# Get an estimator for the compression term
def get_compression_estimator(var, layer, features):
    estimator = Estimator(layer)
    estimator.M = torch.zeros_like(features)
    estimator.S = var*np.ones(features.shape)
    estimator.N = 1
    estimator.layer = layer
    return estimator

def text_heatmap_iba(text_t, image_t, model, layer_idx, beta, var, lr=1, train_steps=10, progbar=True):
    # features = extract_feature_map(model.text_model, layer_idx, text_t)
    features = extract_text_feature_map(model.text_model,layer_idx, text_t)
    layer = extract_bert_layer(model.text_model, layer_idx)
    compression_estimator = get_compression_estimator(var, layer, features)
    reader = IBAInterpreter(model, compression_estimator, beta=beta, lr=lr, steps=train_steps, progbar=progbar)
    return reader.text_heatmap(text_t, image_t)

def vision_heatmap_iba(text_t, image_t, model, layer_idx, beta, var, lr=1, train_steps=10,ensemble=False, progbar=True):
    features = extract_feature_map(model.vision_model, layer_idx, image_t)
    layer = extract_bert_layer(model.vision_model, layer_idx)
    compression_estimator = get_compression_estimator(var, layer, features)
    reader = IBAInterpreter(model, compression_estimator, beta=beta, lr=lr, steps=train_steps, progbar=progbar,ensemble=ensemble)
    return reader.vision_heatmap(text_t, image_t)


def vision_heatmap_tgcam(text_t, image_t, model, layer_idx=7, tgcam_instance=None, device='cuda', **kwargs):
    """
    Args:
        text_t: Tokenized text input [B, T_seq]
        image_t: Preprocessed image tensor [B, C, H, W]
        model: BiomedCLIP model wrapper (ClipWrapper)
        layer_idx: Layer index for feature extraction (default: 7)
        tgcam_instance: Pre-initialized TGCAMPipeline object. 
                        IF NONE, it creates one (warning: inefficient/untrained).
        device: Device to run on (default: 'cuda')
        **kwargs: Additional arguments (visual_dim, text_dim, etc.)
    """
    if device is None or device == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    
    model.eval()
    
    with torch.no_grad():
        # Extract visual features from specified layer (same as IBA method)
        visual_features_raw = extract_feature_map(model.vision_model, layer_idx, image_t)
        # visual_features_raw shape: [B, V_seq, visual_dim]
        
        # Extract text features from specified layer (same as IBA method)
        text_features_raw = extract_text_feature_map(model.text_model, layer_idx, text_t)
        # text_features_raw shape: [B, T_seq, text_dim]
        
        # Auto-detect dimensions
        visual_dim = visual_features_raw.shape[-1]
        text_dim = text_features_raw.shape[-1]
        
        # Ensure features are on correct device
        visual_features = visual_features_raw.to(device)
        text_features = text_features_raw.to(device)
        
        # Visual features: [B, V_seq, visual_dim] where V_seq might include CLS token
        B, V_seq, V_dim = visual_features.shape
        
        # Auto-detect CLS token: If V_seq is NOT a perfect square, assume first token is CLS
        spatial_root = int(V_seq ** 0.5)
        if spatial_root * spatial_root == V_seq:
            # Already perfect square (e.g., 196), no CLS token
            visual_features_patches = visual_features
        else:
            # Has CLS token (e.g., 197 = 196 + 1), slice it off
            visual_features_patches = visual_features[:, 1:, :]
            print(f"Detected CLS token. Sliced from {V_seq} -> {visual_features_patches.shape[1]} patches.")
        
        # --- Engineering Fix: Use passed instance ---
        if tgcam_instance is None:
            # Fallback for backward compatibility, but log warning
            print("WARNING: Re-initializing TGCAM. Pass an instance for efficiency.")
            tgcam = TGCAMPipeline(
                visual_dim=visual_dim,
                text_dim=text_dim,
                mid_channels=kwargs.get('common_dim', 512),
                num_item_iterations=kwargs.get('num_item_iterations', 2)
            ).to(device)
        else:
            tgcam = tgcam_instance
        
        # Forward Pass
        saliency_map, fused_features = tgcam(visual_features_patches, text_features)
        
        # --- Post-Processing ---
        # Saliency map is [B, 1, H_patch, W_patch]. Upsample to original image size
        # CRITICAL: Use image_t's spatial dimensions, not hardcoded 224×224
        target_size = (image_t.shape[2], image_t.shape[3])  # [H, W]
        
        heatmap = torch.nn.functional.interpolate(
            saliency_map,
            size=target_size,
            mode='bicubic',
            align_corners=False
        )
        
        # Return numpy array [H, W]
        return heatmap.squeeze().cpu().detach().numpy()


# Alias for backward compatibility - can be used as drop-in replacement
def vision_heatmap(text_t, image_t, model, layer_idx=7, **kwargs):
    """
    Wrapper function that defaults to TGCAM but maintains backward compatibility.
    
    Usage:
        # Use TGCAM (new symmetric attention approach)
        saliency = vision_heatmap(text_t, image_t, model, layer_idx=7)
        
        # Or explicitly use TGCAM
        saliency = vision_heatmap_tgcam(text_t, image_t, model, layer_idx=7)
        
        # Or use old IBA approach
        saliency = vision_heatmap_iba(text_t, image_t, model, layer_idx, beta, var)
    """
    # Check if old IBA parameters are provided
    if 'beta' in kwargs and 'var' in kwargs:
        # Fall back to IBA if old parameters are provided
        return vision_heatmap_iba(
            text_t, image_t, model, layer_idx,
            beta=kwargs.get('beta', 0.1),
            var=kwargs.get('var', 1.0),
            lr=kwargs.get('lr', 1),
            train_steps=kwargs.get('train_steps', 10),
            ensemble=kwargs.get('ensemble', False),
            progbar=kwargs.get('progbar', False)
        )
    else:
        # Default to TGCAM
        return vision_heatmap_tgcam(
            text_t, image_t, model, layer_idx,
            visual_dim=kwargs.get('visual_dim', 768),
            text_dim=kwargs.get('text_dim', 768),
            common_dim=kwargs.get('common_dim', 512),
            num_item_iterations=kwargs.get('num_item_iterations', 2),
            device=kwargs.get('device', None),
            progbar=kwargs.get('progbar', False)
        )

