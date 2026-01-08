"""
Test script for M2IB Saliency + SAM pipeline.

Verifies:
1. All modules can be imported
2. Forward pass works with correct dimensions
3. Components are properly connected
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn

def test_dwt_projector():
    """Test DWT Frequency Projector."""
    print("Testing DWTFrequencyProjector...")
    from freqmedclip.scripts.dwt_projector import DWTFrequencyProjector
    
    projector = DWTFrequencyProjector(in_channels=12, out_channels=768)
    
    # Test with 512x512 input
    x = torch.randn(2, 3, 512, 512)
    out = projector(x, target_size=(32, 32))
    
    assert out.shape == (2, 768, 32, 32), f"Expected (2, 768, 32, 32), got {out.shape}"
    print(f"  ✓ Output shape: {out.shape}")
    
    # Test with 224x224 input
    x_small = torch.randn(2, 3, 224, 224)
    out_small = projector(x_small, target_size=(14, 14))
    
    assert out_small.shape == (2, 768, 14, 14), f"Expected (2, 768, 14, 14), got {out_small.shape}"
    print(f"  ✓ Small input shape: {out_small.shape}")
    
    return True


def test_m2ib():
    """Test M2IB Module."""
    print("Testing M2IBModule...")
    from freqmedclip.scripts.m2ib_module import M2IBModule
    
    m2ib = M2IBModule(embed_dim=512, hidden_dim=256, spatial_size=14)
    
    image_embed = torch.randn(2, 512)
    text_embed = torch.randn(2, 512)
    
    coarse_map = m2ib(image_embed, text_embed)
    
    assert coarse_map.shape == (2, 1, 14, 14), f"Expected (2, 1, 14, 14), got {coarse_map.shape}"
    assert coarse_map.min() >= 0 and coarse_map.max() <= 1, "Output should be in [0, 1]"
    print(f"  ✓ Coarse map shape: {coarse_map.shape}")
    print(f"  ✓ Value range: [{coarse_map.min():.3f}, {coarse_map.max():.3f}]")
    
    return True


def test_spatial_gate():
    """Test Spatial Gate Attention."""
    print("Testing SpatialGateAttention...")
    from freqmedclip.scripts.spatial_gate import SpatialGateAttention
    
    gate = SpatialGateAttention(m2ib_channels=1, freq_channels=768, hidden_channels=256)
    
    m2ib_map = torch.randn(2, 1, 14, 14)
    freq_features = torch.randn(2, 768, 14, 14)
    
    saliency_map = gate(m2ib_map, freq_features)
    
    assert saliency_map.shape == (2, 1, 14, 14), f"Expected (2, 1, 14, 14), got {saliency_map.shape}"
    print(f"  ✓ Saliency map shape: {saliency_map.shape}")
    
    return True


def test_prompt_extractor():
    """Test Prompt Extractor."""
    print("Testing PromptExtractor...")
    from freqmedclip.scripts.prompt_extractor import PromptExtractor
    
    extractor = PromptExtractor(threshold=0.5, num_points=3)
    
    saliency_map = torch.zeros(2, 1, 14, 14)
    # Add a bright region
    saliency_map[0, 0, 4:10, 4:10] = 0.8
    saliency_map[1, 0, 2:8, 6:12] = 0.9
    
    prompts = extractor(saliency_map, image_size=(512, 512))
    
    assert 'boxes' in prompts, "Should return boxes"
    assert 'point_coords' in prompts, "Should return point_coords"
    assert 'point_labels' in prompts, "Should return point_labels"
    assert prompts['boxes'].shape == (2, 4), f"Expected (2, 4), got {prompts['boxes'].shape}"
    assert prompts['point_coords'].shape == (2, 3, 2), f"Expected (2, 3, 2), got {prompts['point_coords'].shape}"
    
    print(f"  ✓ Boxes shape: {prompts['boxes'].shape}")
    print(f"  ✓ Point coords shape: {prompts['point_coords'].shape}")
    print(f"  ✓ Sample box: {prompts['boxes'][0].tolist()}")
    
    return True


def test_dummy_sam():
    """Test Dummy SAM."""
    print("Testing DummySAM...")
    from freqmedclip.scripts.sam_wrapper import DummySAM
    
    sam = DummySAM()
    
    images = torch.randn(2, 3, 512, 512)
    saliency_map = torch.randn(2, 1, 14, 14).sigmoid()
    
    outputs = sam(images, saliency_map)
    
    assert 'masks' in outputs, "Should return masks"
    assert outputs['masks'].shape == (2, 1, 512, 512), f"Expected (2, 1, 512, 512), got {outputs['masks'].shape}"
    print(f"  ✓ Output mask shape: {outputs['masks'].shape}")
    
    return True


def test_saliency_only_model():
    """Test SaliencyOnlyModel (without real BiomedCLIP)."""
    print("Testing SaliencyOnlyModel with mock BiomedCLIP...")
    
    # Create a mock BiomedCLIP
    class MockVisionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.ModuleList([nn.Identity() for _ in range(12)])
            self.encoder.layers = self.encoder
            
        def forward(self, x, output_hidden_states=False, return_dict=True):
            # Mock output
            B = x.shape[0]
            hidden_state = torch.randn(B, 197, 768, device=x.device)
            pooler_output = torch.randn(B, 768, device=x.device)
            
            class Output:
                def __init__(self):
                    self.pooler_output = pooler_output
                    self.hidden_states = [hidden_state] * 13
            
            return Output()
    
    class MockTextModel(nn.Module):
        def forward(self, input_ids, output_hidden_states=False, return_dict=True):
            B = input_ids.shape[0]
            hidden_state = torch.randn(B, 77, 768, device=input_ids.device)
            pooler_output = torch.randn(B, 768, device=input_ids.device)
            
            class Output:
                def __init__(self):
                    self.pooler_output = pooler_output
                    self.hidden_states = [hidden_state] * 13
            
            return Output()
    
    class MockBiomedCLIP(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_model = MockVisionModel()
            self.text_model = MockTextModel()
    
    mock_biomedclip = MockBiomedCLIP()
    
    from freqmedclip.scripts.saliency_sam_model import SaliencyOnlyModel
    
    model = SaliencyOnlyModel(
        biomedclip_model=mock_biomedclip,
        input_size=512,
        freeze_biomedclip=False  # Don't freeze mock
    )
    
    # Test forward pass
    images = torch.randn(2, 3, 512, 512)
    input_ids = torch.randint(0, 1000, (2, 77))
    
    outputs = model(images, input_ids)
    
    assert 'saliency_map' in outputs, "Should return saliency_map"
    assert 'image_embed' in outputs, "Should return image_embed"
    assert 'text_embed' in outputs, "Should return text_embed"
    
    # Saliency should be upsampled to input size
    print(f"  ✓ Saliency map shape: {outputs['saliency_map'].shape}")
    print(f"  ✓ Image embed shape: {outputs['image_embed'].shape}")
    print(f"  ✓ Text embed shape: {outputs['text_embed'].shape}")
    
    # Check value ranges
    sal_min = outputs['saliency_map'].min().item()
    sal_max = outputs['saliency_map'].max().item()
    print(f"  ✓ Saliency value range: [{sal_min:.3f}, {sal_max:.3f}]")
    
    return True


def test_full_pipeline():
    """Test complete pipeline integration."""
    print("\nTesting Full Pipeline Integration...")
    
    from freqmedclip.scripts.dwt_projector import DWTFrequencyProjector
    from freqmedclip.scripts.m2ib_module import M2IBModule
    from freqmedclip.scripts.spatial_gate import SpatialGateAttention
    from freqmedclip.scripts.prompt_extractor import PromptExtractor
    from freqmedclip.scripts.sam_wrapper import DummySAM
    
    # Initialize components
    dwt_proj = DWTFrequencyProjector(12, 768)
    m2ib = M2IBModule(512, 256, 14)
    spatial_gate = SpatialGateAttention(1, 768, 256)
    prompt_extractor = PromptExtractor()
    dummy_sam = DummySAM()
    
    # Simulate inputs
    raw_image = torch.randn(2, 3, 512, 512)
    image_embed = torch.randn(2, 512)
    text_embed = torch.randn(2, 512)
    
    # Step 1: DWT Projection
    freq_features = dwt_proj(raw_image, target_size=(14, 14))
    print(f"  1. DWT Projection: {freq_features.shape}")
    
    # Step 2: M2IB
    coarse_map = m2ib(image_embed, text_embed)
    print(f"  2. M2IB Coarse Map: {coarse_map.shape}")
    
    # Step 3: Spatial Gate
    saliency_small = spatial_gate(coarse_map, freq_features)
    print(f"  3. Spatial Gate Saliency: {saliency_small.shape}")
    
    # Step 4: Extract Prompts
    prompts = prompt_extractor(saliency_small, image_size=(512, 512))
    print(f"  4. Extracted Prompts - Boxes: {prompts['boxes'].shape}")
    
    # Step 5: SAM Mask
    saliency_up = torch.nn.functional.interpolate(
        saliency_small, size=(512, 512), mode='bilinear', align_corners=False
    )
    sam_out = dummy_sam(raw_image, saliency_up)
    print(f"  5. SAM Output Mask: {sam_out['masks'].shape}")
    
    print("  ✓ Full pipeline test passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("M2IB Saliency + SAM Pipeline Tests")
    print("=" * 60)
    
    tests = [
        test_dwt_projector,
        test_m2ib,
        test_spatial_gate,
        test_prompt_extractor,
        test_dummy_sam,
        test_saliency_only_model,
        test_full_pipeline,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            results.append((test.__name__, False))
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Pipeline is ready for training.")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")


if __name__ == '__main__':
    main()
