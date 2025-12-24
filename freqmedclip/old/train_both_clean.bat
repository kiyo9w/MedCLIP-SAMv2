@echo off
echo ========================================
echo FreqMedCLIP Training - Clean Restart
echo ========================================
echo.
echo IMPORTANT: This will DELETE old checkpoints to save space!
echo Training will only save BEST epoch per dataset.
echo.
pause
echo.

echo ========================================
echo Cleaning old checkpoints...
echo ========================================
del /Q checkpoints\fusion_breast_tumors_*.pth 2>nul
del /Q checkpoints\fusion_brain_tumors_*.pth 2>nul
echo ✓ Old checkpoints removed
echo.

echo ========================================
echo Activating conda environment: medclipsamv2
call conda activate medclipsamv2

echo.
echo ========================================
echo Training Configuration:
echo   - Dataset 1: breast_tumors (113 test)
echo   - Dataset 2: brain_tumors (600 test)
echo   - Epochs: 100 per dataset
echo   - Batch size: 4
echo   - Learning rate: 1e-4
echo   - Checkpoint: BEST epoch only
echo   - Metrics: Dice, IoU printed each epoch
echo ========================================
echo.

echo ========================================
echo [1/2] Training on breast_tumors
echo ========================================
echo Start time: %TIME%
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
echo ✓ breast_tumors training completed!
echo.

echo ========================================
echo [2/2] Training on brain_tumors
echo ========================================
echo Start time: %TIME%
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
echo ✓ brain_tumors training completed!
echo.

echo ========================================
echo All Training Completed!
echo ========================================
echo End time: %TIME%
echo.
echo Press any key to run evaluation with visualizations...
pause

echo.
echo ========================================
echo Evaluating + Visualizing Results
echo ========================================
echo.

echo [1/2] Evaluating breast_tumors...
for %%f in (checkpoints\fusion_breast_tumors_*.pth) do (
    python evaluate_model.py ^
        --dataset breast_tumors ^
        --checkpoint %%f
)

echo.
echo [2/2] Evaluating brain_tumors...
for %%f in (checkpoints\fusion_brain_tumors_*.pth) do (
    python evaluate_model.py ^
        --dataset brain_tumors ^
        --checkpoint %%f
)

echo.
echo ========================================
echo All Done!
echo ========================================
echo Check:
echo   - Results: results_*.txt
echo   - Visualizations: visualizations/
echo ========================================
pause
