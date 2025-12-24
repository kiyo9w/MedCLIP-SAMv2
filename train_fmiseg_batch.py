#!/usr/bin/env python3
"""
Batch training script for FMISeg on brain_tumors and breast_tumors.
Trains sequentially and logs results.
"""

import os
import sys
import subprocess
from datetime import datetime

def run_fmiseg_training(dataset_name, config_file):
    """Run FMISeg training for a single dataset."""
    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training FMISeg on {dataset_name}")
    print(f"{'='*70}")
    print(f"Config: {config_file}\n")
    
    cmd = f"python train.py --config {config_file} --dataset {dataset_name}"
    
    # Run training
    result = subprocess.run(cmd, shell=True, cwd='.')
    
    if result.returncode == 0:
        print(f"\n✓ {dataset_name} training completed successfully!")
        return True
    else:
        print(f"\n✗ {dataset_name} training failed!")
        return False

def main():
    """Main entry point."""
    print(f"\n{'#'*70}")
    print(f"# FMISeg Batch Training - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")
    
    os.chdir('FMISeg')
    
    datasets = [
        {'name': 'brain_tumors', 'config': './config/train_brain_tumors.yaml'},
        {'name': 'breast_tumors', 'config': './config/train_breast_tumors.yaml'},
    ]
    
    results = []
    
    for dataset_cfg in datasets:
        success = run_fmiseg_training(dataset_cfg['name'], dataset_cfg['config'])
        results.append((dataset_cfg['name'], success))
    
    # Summary
    print(f"\n{'='*70}")
    print("Training Summary:")
    print(f"{'='*70}")
    for dataset_name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {dataset_name}: {status}")
    
    print(f"\nCheckpoints saved in: FMISeg/save_model/")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
