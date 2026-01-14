
import torch
import torch.nn as nn
import sys
import os

# Add parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock BiomedCLIP
class MockConfig:
    def __init__(self):
        self.vision_config = type('obj', (object,), {'image_size': 224})
        self.hidden_size = 768

class MockVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MockConfig()
        
    def forward(self, x, output_hidden_states=True, return_dict=True):
        B = x.shape[0]
        # Return list of 13 layers (0-12), shape (B, 197, 768)
        hidden_states = [torch.randn(B, 197, 768) for _ in range(13)]
        return type('obj', (object,), {'hidden_states': hidden_states})

class MockTextModel(nn.Module):
    def forward(self, input_ids, output_hidden_states=True, return_dict=True):
        B = input_ids.shape[0]
        # Return list of layers
        hidden_states = [torch.randn(B, 77, 768) for _ in range(13)]
        return type('obj', (object,), {'hidden_states': hidden_states})

class MockBiomedCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = MockVisionModel()
        self.text_model = MockTextModel()
        self.config = MockConfig()

# Import Model
from freqmedclip.train_freq_fusion import FrequencyMedCLIPSAMv2
from freqmedclip.scripts.spectr_components import RecouplingLoss, MorphologicalEdgeTarget

def test_model():
    print("Initializing Mock Model...")
    biomedclip = MockBiomedCLIP()
    args = type('obj', (object,), {})
    
    model = FrequencyMedCLIPSAMv2(biomedclip, args)
    model.eval()
    
    # Inputs
    B = 2
    pixel_values = torch.randn(B, 3, 224, 224)
    image_raw = torch.rand(B, 3, 224, 224) # [0,1]
    input_ids = torch.randint(0, 1000, (B, 77))
    masks = torch.randint(0, 2, (B, 224, 224)).float()
    
    print("Running Forward Pass...")
    preds, edge_logits, fused_feats, text_feats = model(pixel_values, input_ids, image_raw)
    
    print(f"Preds Shape: {preds.shape} (Expected: {B, 1, 224, 224})")
    print(f"Edge Logits Shape: {edge_logits.shape} (Expected: {B, 1, 112, 112})")
    print(f"Fused Feats Shape: {fused_feats.shape} (Expected: {B, 24, 224, 224})")
    print(f"Text Feats Shape: {text_feats.shape} (Expected: {B, 768})")
    
    assert preds.shape == (B, 1, 224, 224)
    assert edge_logits.shape == (B, 1, 112, 112)
    assert fused_feats.shape == (B, 24, 224, 224)
    
    # Test Losses
    print("Testing Losses...")
    recoupling_loss_fn = RecouplingLoss(in_channels=24, text_dim=768)
    edge_target_fn = MorphologicalEdgeTarget()
    
    loss_recouple = recoupling_loss_fn(fused_feats, preds, text_feats)
    print(f"Recoupling Loss: {loss_recouple.item()}")
    
    edge_targets = edge_target_fn(masks.unsqueeze(1)) # (B, 1, 224, 224)
    # Resize target to matches logits (112) or logits to target?
    # In train loop we did interpolate targets to logits size
    import torch.nn.functional as F
    edge_targets_aligned = F.interpolate(edge_targets, size=(112, 112), mode='nearest')
    
    print(f"Edge Targets Shape: {edge_targets_aligned.shape}")
    
    print("Verification Successful!")

if __name__ == "__main__":
    test_model()
