@echo off
echo ========================================
echo FreqMedCLIP Training - Both Complete Datasets
echo ========================================
echo.
echo Training Configuration:
echo   - Dataset 1: breast_tumors (113 test images)
echo   - Dataset 2: brain_tumors (600 test images)
echo   - Epochs: 100 per dataset
echo   - Batch size: 4
echo   - Learning rate: 1e-4
echo.
echo ========================================
echo Activating conda environment: medclipsamv2
call conda activate medclipsamv2

echo.
echo ========================================
echo [1/2] Training on breast_tumors
echo ========================================
echo Starting time: %TIME%
echo.

python train_freq_fusion.py ^
    --dataset breast_tumors ^
    --epochs 100 ^
    --batch-size 4 ^
    --lr 0.0001

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo ERROR: breast_tumors training failed!
    echo ========================================
    pause
    exit /b 1
)

echo.
echo ========================================
echo [2/2] Training on brain_tumors
echo ========================================
echo Starting time: %TIME%
echo.

python train_freq_fusion.py ^
    --dataset brain_tumors ^
    --epochs 100 ^
    --batch-size 4 ^
    --lr 0.0001

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo ERROR: brain_tumors training failed!
    echo ========================================
    pause
    exit /b 1
)

echo.
echo ========================================
echo All Training Completed Successfully!
echo ========================================
echo.
echo Checkpoints saved in:
echo   - checkpoints/fusion_breast_tumors_epoch100.pth
echo   - checkpoints/fusion_brain_tumors_epoch100.pth
echo.
echo End time: %TIME%
echo ========================================
echo.
echo Press any key to run evaluation on both datasets...
pause

echo.
echo ========================================
echo Running Evaluation on Both Datasets
echo ========================================
echo.

echo [1/2] Evaluating breast_tumors...
python evaluate_freqmedclip.py ^
    --dataset breast_tumors ^
    --checkpoint checkpoints/fusion_breast_tumors_epoch100.pth

echo.
echo [2/2] Evaluating brain_tumors...
python evaluate_freqmedclip.py ^
    --dataset brain_tumors ^
    --checkpoint checkpoints/fusion_brain_tumors_epoch100.pth

echo.
echo ========================================
echo All Done! Check results above.
echo ========================================
pause
