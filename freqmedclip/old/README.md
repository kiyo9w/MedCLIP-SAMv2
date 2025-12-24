# FreqMedCLIP - Smart Single-Stream Architecture

Folder chứa code cho FreqMedCLIP (Frequency-domain Medical CLIP) với kiến trúc Smart Single-Stream.

## Cấu trúc

```
freqmedclip/
├── scripts/
│   ├── freq_components.py      # DWT + SmartFusionBlock
│   └── postprocess.py          # KMeans + Threshold postprocessing
│
├── train_freq_fusion.py        # Main training script
├── evaluate_model.py            # Evaluation + auto-visualization
├── save_freqmedclip_predictions.py   # Batch prediction
├── postprocess_freqmedclip_outputs.py # Batch postprocessing
├── visualize_prediction.py     # 3x3 grid visualization
│
├── train_both_clean.bat         # ⭐ RECOMMENDED: Train 2 datasets (best epoch only)
├── train_both_datasets.bat      # Old: Train all epochs
├── train_and_eval.bat           # Train + evaluate
└── run_freqmedclip_pipeline.ps1 # Complete pipeline
```

## Quick Start

### 1. Training (Chỉ lưu best epoch - Recommended)

```bash
cd freqmedclip
.\train_both_clean.bat
```

Sẽ train:
- breast_tumors (100 epochs)
- brain_tumors (100 epochs)
- Chỉ lưu checkpoint tốt nhất (tiết kiệm ~98% dung lượng)
- In Dice + IoU mỗi epoch
- Tự động evaluate + visualize sau khi train xong

### 2. Training từng dataset

```bash
cd freqmedclip
python train_freq_fusion.py --dataset breast_tumors --epochs 100 --batch-size 4 --lr 0.0001
```

### 3. Evaluation + Visualization

```bash
python evaluate_model.py --dataset breast_tumors --checkpoint ../checkpoints/fusion_breast_tumors_epoch50.pth
```

Output:
- `results_breast_tumors_epoch50.txt` - Metrics (Dice, IoU, Precision, Recall)
- `visualizations/breast_tumors_eval/` - Overlay images (GT vs Pred)

### 4. Complete Pipeline (Generate → Postprocess → Visualize)

```powershell
.\run_freqmedclip_pipeline.ps1 -Dataset breast_tumors -Checkpoint ../checkpoints/fusion_breast_tumors_epoch100.pth
```

## File Details

### Core Components
- **freq_components.py**: 
  - `DWTForward` - Discrete Wavelet Transform (Haar filters)
  - `SmartFusionBlock` - Coarse-to-Fine fusion với multi-scale gates

### Training
- **train_freq_fusion.py**: Main training với validation metrics
  - Automatically saves best checkpoint only
  - Prints Dice + IoU every epoch

### Evaluation
- **evaluate_model.py**: 
  - Calculate Dice, IoU, Precision, Recall
  - Auto-generate visualizations (every 10 samples)

### Postprocessing
- **postprocess.py**:
  - `postprocess_saliency_kmeans()` - KMeans clustering (default)
  - `postprocess_saliency_threshold()` - Simple threshold

## Import từ ngoài folder

Nếu cần import FreqMedCLIP từ root project:

```python
from freqmedclip.scripts.freq_components import SmartFusionBlock, DWTForward
from freqmedclip.train_freq_fusion import FreqMedCLIPDataset
```

## Dependencies

- PyTorch >= 1.13
- transformers >= 4.30
- PyWavelets (pywt)
- OpenCV (cv2)
- PIL, numpy, tqdm

Xem `../medclipsamv2_env.yml` để setup environment.
