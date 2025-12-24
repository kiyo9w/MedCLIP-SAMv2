# FreqMedCLIP (FMISeg Dual-Branch Version)

Kiến trúc mới: BiomedCLIP backbone + FrequencyEncoder + FPNAdapter + dual decoder (ViT branch & Frequency branch) với FFBI/LFFI. Không dùng SmartFusionBlock/SAM.

## Files chính
- `train_freq_fusion.py`: Train 1 dataset (dual-branch FMISeg-style).
- `batch_train_and_eval.py`: Train tuần tự brain_tumors + breast_tumors, auto-eval, sinh summary.
- `resume_training.py`: Tiếp tục train thêm 100 epochs từ checkpoint đã có.
- `evaluate_freqmedclip.py`: Evaluate test/val + visualize intermediate (freq features, FPN scales, 2 branches, overlay).
- `postprocess_freqmedclip_outputs.py`: Hậu xử lý mask (kmeans/threshold, top-k components).
- `scripts/freq_components.py`, `scripts/fmiseg_components.py`, `scripts/postprocess.py`: Core modules.
- `old/`: Toàn bộ script pipeline/SmartFusionBlock cũ (không còn dùng, giữ tham khảo).

## Quick start

```powershell
cd freqmedclip
conda activate medclipsamv2  # hoặc chạy ../activate_conda_medclipsamv2.ps1

# Train 1 dataset
python train_freq_fusion.py --dataset breast_tumors --epochs 100 --batch-size 4 --lr 1e-4

# Evaluate
python evaluate_freqmedclip.py --dataset breast_tumors --checkpoint ..\checkpoints\results_*/breast_tumors_checkpoints/*.pth
```

## Train hai dataset + auto eval
```powershell
cd freqmedclip
python batch_train_and_eval.py
# Kết quả + visualizations + summary trong results_{timestamp}/
```

## Resume thêm 100 epochs từ checkpoint cũ
```powershell
cd freqmedclip
python resume_training.py
# Input checkpoints: ..\checkpoints\results_20251205_011222\<dataset>_checkpoints\fusion_...pth
# Output: ..\checkpoints\results_resume_{timestamp}\
```

## Hậu xử lý mask (dùng cho weak SSL hoặc nộp kết quả)
```powershell
python postprocess_freqmedclip_outputs.py --input <pred_dir> --output <clean_dir> --method kmeans --top-k 1
```
- Input: raw mask (0-255) từ mô hình.
- Output: binary mask sạch hơn (top-k largest components).

## Chuẩn bị pseudo-labels cho weakly SSL (nnUNet)
1) Sinh predictions (dùng evaluate_freqmedclip hoặc script riêng) => thư mục raw/cleaned masks.
2) Chuyển sang định dạng nnUNet (theo `weak_segmentation/nnunetv2/dataset_conversion/our_datasets.py`):
```
data/nnUNet_raw/DatasetXXX_Name/
  imagesTr/ case_0000_0000.png
  labelsTr/ case_0000.png   # pseudo-label
  imagesTs/ ...
  labelsTs/ ... (nếu có)
```
3) Chạy:
```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID
nnUNetv2_train DATASET_ID 2d all --npz
```

## Legacy (không dùng với bản mới)
- `old/README.md`, `old/run_freqmedclip_pipeline.ps1`, `old/save_freqmedclip_predictions.py`, `old/train_and_eval.bat`, `old/train_both_clean.bat`, `old/train_both_datasets.bat`, `old/visualize_prediction.py`.

## Notes
- Checkpoints/results đều đã được ignore trong .gitignore; không push lên git.
- BiomedCLIP tải local từ `../saliency_maps/model`.
