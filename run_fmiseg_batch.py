"""
Batch training script for FMISeg on brain_tumors and breast_tumors.
Trains sequentially, evaluates, and saves results with logging.
"""

import os
import sys
import subprocess
from datetime import datetime
import shutil

def run_command(cmd, log_file, description):
    """Run command and log output."""
    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {description}")
    print(f"{'='*70}")
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {description}\n")
        f.write(f"{'='*70}\n")
        f.write(f"Command: {cmd}\n\n")
    
    # Run and stream output to both console and log
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    with open(log_file, 'a', encoding='utf-8') as f:
        for line in process.stdout:
            print(line, end='')
            f.write(line)
    
    process.wait()
    return process.returncode == 0

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = f"./results_fmiseg_{timestamp}"
    os.makedirs(base_results_dir, exist_ok=True)
    
    log_file = os.path.join(base_results_dir, "training_log.txt")
    summary_file = os.path.join(base_results_dir, "SUMMARY.txt")
    
    print(f"\n{'#'*70}")
    print(f"# FMISeg Batch Training & Evaluation - {timestamp}")
    print(f"# Results Directory: {base_results_dir}")
    print(f"# Log File: {log_file}")
    print(f"{'#'*70}\n")
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"FMISeg Training & Evaluation Log\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Results Directory: {base_results_dir}\n\n")
    
    datasets = [
        {'name': 'brain_tumors', 'config': './config/train_brain_tumors.yaml'},
        {'name': 'breast_tumors', 'config': './config/train_breast_tumors.yaml'},
    ]
    
    results = []
    
    # Change to FMISeg directory
    os.chdir('FMISeg')
    
    for dataset_cfg in datasets:
        dataset_name = dataset_cfg['name']
        config_path = dataset_cfg['config']
        
        print(f"\n{'*'*70}")
        print(f"* Processing: {dataset_name.upper()}")
        print(f"{'*'*70}\n")
        
        dataset_result_dir = os.path.join('..', base_results_dir, dataset_name)
        os.makedirs(dataset_result_dir, exist_ok=True)
        
        # Step 1: Train
        train_cmd = f"python train.py --config {config_path} --dataset {dataset_name}"
        train_success = run_command(
            train_cmd,
            os.path.join('..', log_file),
            f"TRAINING {dataset_name}"
        )
        
        if train_success:
            print(f"\n✓ Training completed for {dataset_name}")
            
            # Copy checkpoint
            ckpt_src = f"./save_model/{dataset_name}"
            if os.path.exists(ckpt_src):
                ckpt_dst = os.path.join(dataset_result_dir, 'checkpoint')
                if os.path.exists(ckpt_dst):
                    shutil.rmtree(ckpt_dst)
                shutil.copytree(ckpt_src, ckpt_dst)
                print(f"✓ Checkpoint saved to {ckpt_dst}")
                
                # Find checkpoint file
                ckpt_files = [f for f in os.listdir(ckpt_src) if f.endswith('.ckpt')]
                if ckpt_files:
                    ckpt_file = os.path.join(ckpt_src, ckpt_files[0])
                    
                    # Step 2: Evaluate
                    eval_cmd = f"python evaluate.py --config {config_path} --checkpoint {ckpt_file}"
                    eval_success = run_command(
                        eval_cmd,
                        os.path.join('..', log_file),
                        f"EVALUATING {dataset_name}"
                    )
                    
                    if eval_success:
                        print(f"✓ Evaluation completed for {dataset_name}")
                    else:
                        print(f"✗ Evaluation failed for {dataset_name}")
                    
                    results.append({
                        'dataset': dataset_name,
                        'train_success': train_success,
                        'eval_success': eval_success if ckpt_files else False,
                        'checkpoint': ckpt_file if ckpt_files else None
                    })
                else:
                    print(f"✗ No checkpoint found for {dataset_name}")
                    results.append({
                        'dataset': dataset_name,
                        'train_success': train_success,
                        'eval_success': False,
                        'checkpoint': None
                    })
            else:
                print(f"✗ Checkpoint directory not found for {dataset_name}")
                results.append({
                    'dataset': dataset_name,
                    'train_success': train_success,
                    'eval_success': False,
                    'checkpoint': None
                })
        else:
            print(f"✗ Training failed for {dataset_name}")
            results.append({
                'dataset': dataset_name,
                'train_success': False,
                'eval_success': False,
                'checkpoint': None
            })
    
    # Generate summary
    print(f"\n{'='*70}")
    print("Generating Summary Report...")
    print(f"{'='*70}\n")
    
    with open(os.path.join('..', summary_file), 'w', encoding='utf-8') as f:
        f.write(f"FMISeg Training & Evaluation Summary\n")
        f.write(f"{'='*70}\n")
        f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Results Directory: {base_results_dir}\n\n")
        
        for result in results:
            f.write(f"\n{'-'*70}\n")
            f.write(f"Dataset: {result['dataset'].upper()}\n")
            f.write(f"{'-'*70}\n")
            f.write(f"Training: {'✓ SUCCESS' if result['train_success'] else '✗ FAILED'}\n")
            f.write(f"Evaluation: {'✓ SUCCESS' if result['eval_success'] else '✗ FAILED'}\n")
            if result['checkpoint']:
                f.write(f"Checkpoint: {result['checkpoint']}\n")
            f.write("\n")
        
        f.write(f"\n{'='*70}\n")
        f.write("Summary:\n")
        f.write(f"{'='*70}\n")
        for result in results:
            status = "✓ COMPLETE" if result['train_success'] and result['eval_success'] else "✗ INCOMPLETE"
            f.write(f"  {result['dataset']}: {status}\n")
    
    print(f"\n{'='*70}")
    print("All tasks completed!")
    print(f"{'='*70}")
    print(f"Results saved in: {base_results_dir}")
    print(f"Log file: {log_file}")
    print(f"Summary: {summary_file}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
