@echo off
echo ========================================
echo FreqMedCLIP Phase 2 Training - 100 Epochs
echo ========================================
echo.
echo Activating conda environment: medclipsamv2
call conda activate medclipsamv2

echo.
echo Starting training...
echo Dataset: breast_tumors
echo Epochs: 100
echo Batch size: 4
echo Learning rate: 1e-4
echo.

python train_freq_fusion.py --dataset breast_tumors --epochs 100 --batch-size 4 --lr 0.0001

echo.
echo ========================================
echo Training completed!
echo Checkpoints saved in: checkpoints/breast_tumors/
echo ========================================
echo.
echo Press any key to start evaluation on test set...
pause

echo.
echo ========================================
echo Running Evaluation on Test Set
echo ========================================
python evaluate_model.py --dataset breast_tumors --checkpoint checkpoints/breast_tumors/fusion_breast_tumors_epoch100.pth

echo.
echo ========================================
echo All done! Check results above.
echo ========================================
pause
