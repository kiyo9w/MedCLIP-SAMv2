#!/usr/bin/env python3
"""
Quick Test Script - Validate Inference Pipeline Setup
Tests that all components are working before running full inference
"""

import os
import sys
import torch
from pathlib import Path

def test_checkpoint():
    """Test checkpoint detection"""
    print("🔍 Testing checkpoint detection...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    import glob
    
    pth_files = glob.glob(os.path.join(script_dir, "*.pth"))
    pth_files += glob.glob(os.path.join(script_dir, "**", "*.pth"), recursive=True)
    
    if pth_files:
        latest = max(pth_files, key=os.path.getctime)
        print(f"   ✓ Found checkpoint: {os.path.basename(latest)}")
        return True
    else:
        print(f"   ✗ No checkpoint found")
        return False

def test_dataset():
    """Test dataset availability"""
    print("🔍 Testing dataset availability...")
    
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "data" / "brain_tumors"
    
    splits = ['train', 'val', 'test']
    all_exist = True
    
    for split in splits:
        img_dir = dataset_path / f"{split}_images"
        mask_dir = dataset_path / f"{split}_masks"
        
        if img_dir.exists() and mask_dir.exists():
            img_count = len(list(img_dir.glob("*.png")))
            mask_count = len(list(mask_dir.glob("*.png")))
            print(f"   ✓ {split:5s}: {img_count} images, {mask_count} masks")
        else:
            print(f"   ✗ {split:5s}: Directory not found")
            all_exist = False
    
    return all_exist

def test_dependencies():
    """Test required dependencies"""
    print("🔍 Testing dependencies...")
    
    required = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'tqdm': 'tqdm',
    }
    
    all_installed = True
    
    for module, name in required.items():
        try:
            __import__(module)
            print(f"   ✓ {name}")
        except ImportError:
            print(f"   ✗ {name} - NOT INSTALLED")
            all_installed = False
    
    return all_installed

def test_cuda():
    """Test CUDA availability"""
    print("🔍 Testing CUDA/GPU...")
    
    if torch.cuda.is_available():
        print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   ✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print(f"   ⚠ CUDA not available (will use CPU - slower)")
        return False

def test_output_dirs():
    """Test output directory creation"""
    print("🔍 Testing output directories...")
    
    script_dir = Path(__file__).parent
    test_dirs = [
        script_dir / "freqmedclip_results" / "brain_tumors",
        script_dir / "predictions_temp"
    ]
    
    for test_dir in test_dirs:
        try:
            test_dir.mkdir(parents=True, exist_ok=True)
            print(f"   ✓ Can create: {test_dir.name}")
        except Exception as e:
            print(f"   ✗ Cannot create {test_dir.name}: {e}")
            return False
    
    return True

def main():
    print("\n" + "="*70)
    print("FreqMedCLIP Inference - Pre-flight Check")
    print("="*70 + "\n")
    
    tests = [
        ("Dependencies", test_dependencies),
        ("CUDA/GPU", test_cuda),
        ("Checkpoint", test_checkpoint),
        ("Dataset", test_dataset),
        ("Output Directories", test_output_dirs),
    ]
    
    results = {}
    
    for name, test_func in tests:
        results[name] = test_func()
        print()
    
    print("="*70)
    print("Summary")
    print("="*70 + "\n")
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status:10s} {name}")
    
    print()
    
    all_passed = all(results.values())
    
    if all_passed:
        print("✅ All checks passed! Ready to run inference.")
        print("\nTo start inference, run:")
        print("   ./run_brain_tumors_inference.sh")
        print("or")
        print("   python inference_all_brain_tumors.py")
        return 0
    else:
        print("⚠️  Some checks failed. Please fix issues before running inference.")
        
        if not results["Dependencies"]:
            print("\nTo install dependencies:")
            print("   pip install torch torchvision transformers albumentations opencv-python numpy pillow tqdm")
        
        if not results["Checkpoint"]:
            print("\nTo specify checkpoint manually:")
            print("   python inference_all_brain_tumors.py --checkpoint path/to/checkpoint.pth")
        
        return 1

if __name__ == '__main__':
    sys.exit(main())
