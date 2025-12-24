"""
Batch Postprocessing for FreqMedCLIP Predictions
Applies KMeans clustering to clean raw saliency maps (remove noise).

Usage:
    python postprocess_freqmedclip_outputs.py --input predictions/breast_tumors --output predictions_cleaned/breast_tumors
"""
import os
import sys
import argparse
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from freqmedclip.scripts.postprocess import postprocess_saliency_kmeans, postprocess_saliency_threshold


def main():
    parser = argparse.ArgumentParser(description='Postprocess FreqMedCLIP raw saliency maps')
    parser.add_argument('--input', type=str, required=True, 
                        help='Input directory containing raw saliency maps (.png)')
    parser.add_argument('--output', type=str, required=True, 
                        help='Output directory for cleaned masks')
    parser.add_argument('--method', type=str, default='kmeans', 
                        choices=['kmeans', 'threshold'], 
                        help='Postprocessing method (default: kmeans)')
    parser.add_argument('--num-clusters', type=int, default=2, 
                        help='Number of clusters for KMeans (default: 2)')
    parser.add_argument('--top-k', type=int, default=1, 
                        help='Keep top K largest connected components (default: 1)')
    parser.add_argument('--threshold', type=float, default=0.3, 
                        help='Threshold value for threshold method (default: 0.3)')
    args = parser.parse_args()

    # Validate input
    if not os.path.isdir(args.input):
        print(f"❌ Error: Input directory not found: {args.input}")
        return
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Get all image files
    files = [f for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print(f"❌ No image files found in {args.input}")
        return
    
    print(f"📁 Input: {args.input}")
    print(f"📁 Output: {args.output}")
    print(f"🔧 Method: {args.method}")
    print(f"📊 Processing {len(files)} files...")
    
    success_count = 0
    for file in tqdm(files, desc="Postprocessing"):
        try:
            # Read saliency map
            input_path = os.path.join(args.input, file)
            saliency_map = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            
            if saliency_map is None:
                print(f"⚠️ Warning: Failed to read {file}, skipping...")
                continue
            
            # Apply postprocessing
            if args.method == 'kmeans':
                cleaned_mask = postprocess_saliency_kmeans(
                    saliency_map, 
                    num_clusters=args.num_clusters, 
                    top_k_components=args.top_k
                )
            else:  # threshold
                cleaned_mask = postprocess_saliency_threshold(
                    saliency_map, 
                    threshold=args.threshold, 
                    top_k_components=args.top_k
                )
            
            # Save cleaned mask
            output_path = os.path.join(args.output, file)
            cv2.imwrite(output_path, cleaned_mask)
            success_count += 1
            
        except Exception as e:
            print(f"❌ Error processing {file}: {e}")
    
    print(f"\n✅ Postprocessing complete!")
    print(f"   - Processed: {success_count}/{len(files)} files")
    print(f"   - Output: {args.output}")
    
    # Generate comparison report
    print("\n📊 Pipeline Recommendation:")
    print("   1. Use 'kmeans' for general cases (default)")
    print("   2. Use 'threshold' for simpler/faster processing")
    print("   3. Adjust --top-k for multi-object segmentation (e.g., lungs: --top-k 2)")


if __name__ == '__main__':
    main()
