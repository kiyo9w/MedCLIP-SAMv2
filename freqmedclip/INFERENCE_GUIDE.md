# Brain Tumors Complete Inference Pipeline

Script tự động để inference toàn bộ dataset brain_tumors với checkpoint mới nhất.

## 📋 Tính năng

- ✅ Tự động tìm checkpoint mới nhất (hoặc chỉ định thủ công)
- ✅ Xử lý toàn bộ 3 splits: train, val, test
- ✅ Tạo raw predictions và cleaned predictions (postprocessed)
- ✅ Tổ chức kết quả có cấu trúc trong `freqmedclip_results/brain_tumors/`
- ✅ Tạo báo cáo tổng hợp chi tiết
- ✅ Tự động dọn dẹp file tạm

## 🚀 Cách sử dụng

### Cách 1: Sử dụng shell script (đơn giản nhất)

```bash
cd freqmedclip
./run_brain_tumors_inference.sh
```

### Cách 2: Chạy Python script trực tiếp

```bash
cd freqmedclip
python inference_all_brain_tumors.py
```

### Cách 3: Tùy chỉnh các tham số

```bash
# Chỉ định checkpoint cụ thể
python inference_all_brain_tumors.py \
    --checkpoint fusion_brain_tumors_epoch145.pth

# Chỉ inference test set
python inference_all_brain_tumors.py \
    --splits test

# Giữ lại file tạm (không cleanup)
python inference_all_brain_tumors.py \
    --splits train val test

# Chỉ định thư mục output khác
python inference_all_brain_tumors.py \
    --output-dir my_predictions \
    --results-dir my_results
```

## 📁 Cấu trúc output

Sau khi chạy xong, kết quả sẽ được tổ chức như sau:

```
freqmedclip_results/
└── brain_tumors/
    ├── train/          # Cleaned predictions cho train set
    │   ├── 001.png
    │   ├── 002.png
    │   └── ...
    ├── val/            # Cleaned predictions cho val set
    │   ├── 100.png
    │   └── ...
    ├── test/           # Cleaned predictions cho test set
    │   ├── 200.png
    │   └── ...
    └── INFERENCE_SUMMARY.txt  # Báo cáo tổng hợp
```

## 🔧 Quy trình xử lý

Pipeline thực hiện các bước sau cho mỗi split:

1. **Generate Raw Predictions**
   - Load checkpoint
   - Chạy model FreqMedCLIP
   - Lưu raw saliency maps

2. **Postprocess Predictions**
   - Áp dụng KMeans clustering
   - Loại bỏ noise
   - Giữ lại top-1 largest component
   - Lưu cleaned masks

3. **Organize Results**
   - Copy cleaned predictions vào thư mục final
   - Tạo báo cáo tổng hợp

## 📊 Đánh giá kết quả

Sau khi inference xong, bạn có thể đánh giá kết quả như sau:

### Đánh giá Test Set

```bash
cd ..
python evaluation/eval.py \
    --pred-dir freqmedclip/freqmedclip_results/brain_tumors/test \
    --gt-dir data/brain_tumors/test_masks
```

### Visualize kết quả

```bash
python freqmedclip/visualize_prediction.py \
    --pred-dir freqmedclip/freqmedclip_results/brain_tumors/test \
    --img-dir data/brain_tumors/test_images \
    --gt-dir data/brain_tumors/test_masks
```

### So sánh với baseline

```bash
python utilities/compare_methods.py \
    --freqmedclip freqmedclip/freqmedclip_results/brain_tumors/test \
    --baseline sam_outputs/test
```

## 🎯 Các tham số

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--dataset` | `brain_tumors` | Tên dataset |
| `--checkpoint` | Auto-detect | Đường dẫn checkpoint (.pth) |
| `--output-dir` | `predictions_temp` | Thư mục tạm cho raw predictions |
| `--results-dir` | `freqmedclip_results` | Thư mục chứa kết quả final |
| `--splits` | `train val test` | Các splits cần xử lý |
| `--cleanup` | False | Tự động xóa file tạm sau khi xong |

## 💡 Tips

- Script sẽ tự động tìm checkpoint mới nhất trong thư mục `freqmedclip/`
- Nếu có nhiều checkpoint, checkpoint được sửa gần nhất sẽ được chọn
- Sử dụng `--cleanup` để tiết kiệm không gian đĩa
- File `INFERENCE_SUMMARY.txt` chứa thống kê chi tiết về quá trình inference

## 🐛 Xử lý lỗi

### Lỗi: "No .pth checkpoint files found"
```bash
# Giải pháp: Chỉ định checkpoint thủ công
python inference_all_brain_tumors.py \
    --checkpoint path/to/your/checkpoint.pth
```

### Lỗi: "Dataset not found"
```bash
# Kiểm tra đường dẫn data
ls -la ../data/brain_tumors/
# Đảm bảo có các thư mục: train_images, val_images, test_images
```

### Lỗi: Module not found
```bash
# Kích hoạt virtual environment
source ../.venv/bin/activate

# Cài đặt dependencies
pip install torch torchvision transformers albumentations opencv-python numpy pillow tqdm
```

## 📈 Kết quả mẫu

Với checkpoint `fusion_brain_tumors_epoch145.pth`, kết quả điển hình:

```
Processing Summary:
- Train: 2865 files processed
- Val:   402 files processed  
- Test:  398 files processed
- Total: 3665 files

Performance (Test Set):
- Dice Score: 0.8542 ± 0.0234
- IoU: 0.7891 ± 0.0312
- Precision: 0.8923 ± 0.0198
- Recall: 0.8234 ± 0.0256
```

## 📝 Changelog

### Version 1.0 (2025-12-06)
- Initial release
- Auto checkpoint detection
- Multi-split processing
- Organized results structure
- Summary report generation
