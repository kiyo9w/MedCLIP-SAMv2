#!/usr/bin/env python3
"""
Prepare brain_tumors and breast_tumors datasets for FMISeg training.

FMISeg requires:
- Images_H (high-frequency wavelets)
- Images_L (low-frequency wavelets)
- GTs (ground truth masks)
- CSV with Image and Description/text columns

This script converts standard image/mask splits into FMISeg format.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pywt

def extract_wavelet_components(image_path, wavelet='db2'):
    """
    Extract high-pass (H) and low-pass (L) frequency components using 2D wavelet transform.
    Returns H and L as images (normalized to 0-255).
    """
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load {image_path}")
    
    # Ensure float32 for wavelet
    img = img.astype(np.float32)
    
    # 2D Wavelet decomposition (single level)
    coeffs = pywt.dwt2(img, wavelet)
    cA, (cH, cV, cD) = coeffs  # cA=low-pass, cH/cV/cD=high-pass
    
    # High frequency: combine high-pass components (edge detection)
    H = np.abs(cH) + np.abs(cV) + np.abs(cD)
    H = cv2.resize(H, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    H = cv2.normalize(H, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Low frequency: upscale approximation coefficients
    L = cv2.resize(cA, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    L = cv2.normalize(L, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return H, L

def prepare_dataset(dataset_name, data_root='./data'):
    """
    Prepare a single dataset (brain_tumors or breast_tumors) for FMISeg.
    """
    print(f"\n{'='*70}")
    print(f"Preparing {dataset_name} for FMISeg...")
    print(f"{'='*70}")
    
    dataset_path = os.path.join(data_root, dataset_name)
    
    # Create FMISeg structure for each split
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dataset_path, f"FMISeg_{split}")
        os.makedirs(split_path, exist_ok=True)
        
        images_h_dir = os.path.join(split_path, 'Images_H')
        images_l_dir = os.path.join(split_path, 'Images_L')
        gts_dir = os.path.join(split_path, 'GTs')
        
        os.makedirs(images_h_dir, exist_ok=True)
        os.makedirs(images_l_dir, exist_ok=True)
        os.makedirs(gts_dir, exist_ok=True)
        
        # Paths to original splits
        img_src_dir = os.path.join(dataset_path, f'{split}_images')
        mask_src_dir = os.path.join(dataset_path, f'{split}_masks')
        
        if not os.path.exists(img_src_dir):
            print(f"  ⚠ {split} images not found: {img_src_dir}")
            continue
        
        image_files = sorted([f for f in os.listdir(img_src_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        print(f"\n  [{split}] Processing {len(image_files)} images...")
        
        # Process each image
        for img_file in tqdm(image_files, desc=f"  {split} wavelet"):
            img_path = os.path.join(img_src_dir, img_file)
            mask_path = os.path.join(mask_src_dir, img_file)
            
            try:
                # Extract wavelet components
                H, L = extract_wavelet_components(img_path)
                
                # Save H and L
                cv2.imwrite(os.path.join(images_h_dir, img_file), H)
                cv2.imwrite(os.path.join(images_l_dir, img_file), L)
                
                # Copy and rename mask (FMISeg expects mask_<name> format for QaTa-COV19 style)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask_out_name = f"mask_{img_file}"
                    cv2.imwrite(os.path.join(gts_dir, mask_out_name), mask)
            except Exception as e:
                print(f"  ! Error processing {img_file}: {e}")
        
        print(f"  ✓ {split} images_H: {len(os.listdir(images_h_dir))}")
        print(f"  ✓ {split} images_L: {len(os.listdir(images_l_dir))}")
        print(f"  ✓ {split} GTs: {len(os.listdir(gts_dir))}")
    
    # Generate CSV prompts for train/val/test
    print(f"\n  Generating prompts CSV files...")
    
    # Define prompts for each dataset
    if 'brain' in dataset_name:
        prompts = [
            "A brain tumor magnetic resonance imaging scan showing abnormal tissue growth.",
            "MRI brain scan with visible tumor lesion.",
            "Brain tumor with clear tumor boundary on MR imaging.",
            "Glioma brain tumor visible on MRI.",
            "Meningioma brain tumor on magnetic resonance imaging.",
            "Pituitary brain tumor visible in MRI scan.",
        ]
    elif 'breast' in dataset_name:
        prompts = [
            "A breast mammography showing abnormal lesion or mass.",
            "Breast ultrasound image with visible tumor region.",
            "Breast cancer tumor visible on medical imaging.",
            "Malignant breast tumor on mammography.",
            "Benign breast tumor visible on imaging.",
            "Breast lesion region marked on diagnostic image.",
        ]
    else:
        prompts = ["A medical image with abnormal region visible."]
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dataset_path, f"FMISeg_{split}")
        gts_dir = os.path.join(split_path, 'GTs')
        csv_path = os.path.join(split_path, f'{split}.csv')
        
        # Get list of mask files (without mask_ prefix for CSV)
        mask_files = sorted(os.listdir(gts_dir))
        if not mask_files:
            print(f"  ⚠ No masks found in {gts_dir}")
            continue
        
        # Create CSV with Image and Description columns
        csv_data = []
        for i, mask_file in enumerate(mask_files):
            # For QaTa-COV19 format compatibility, store without mask_ prefix in CSV
            image_name = mask_file.replace('mask_', '')
            prompt = prompts[i % len(prompts)]
            csv_data.append({'Image': image_name, 'Description': prompt})
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        print(f"  ✓ CSV created: {csv_path} ({len(csv_data)} entries)")

def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("FMISeg Data Preparation Script")
    print("="*70)
    
    data_root = './data'
    
    # Prepare both datasets
    for dataset_name in ['brain_tumors', 'breast_tumors']:
        prepare_dataset(dataset_name, data_root)
    
    print("\n" + "="*70)
    print("✓ FMISeg data preparation complete!")
    print("="*70)
    print("\nDirectory structure created:")
    print("  data/")
    print("    brain_tumors/")
    print("      FMISeg_train/  {Images_H, Images_L, GTs, train.csv}")
    print("      FMISeg_val/    {Images_H, Images_L, GTs, val.csv}")
    print("      FMISeg_test/   {Images_H, Images_L, GTs, test.csv}")
    print("    breast_tumors/")
    print("      FMISeg_train/  {Images_H, Images_L, GTs, train.csv}")
    print("      FMISeg_val/    {Images_H, Images_L, GTs, val.csv}")
    print("      FMISeg_test/   {Images_H, Images_L, GTs, test.csv}")
    print("\nNext steps:")
    print("  1. Update FMISeg/config/train.yaml paths")
    print("  2. Run: cd FMISeg && python train.py")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
