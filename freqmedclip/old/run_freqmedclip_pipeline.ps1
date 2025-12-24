# ============================================================================
# FreqMedCLIP Complete Pipeline with Postprocessing
# ============================================================================
# This script runs the complete FreqMedCLIP pipeline:
# 1. Generate raw saliency maps
# 2. Postprocess with KMeans (like MedCLIP-SAMv2)
# 3. Visualize before/after comparison
#
# Usage:
#   .\run_freqmedclip_pipeline.ps1 -Dataset breast_tumors -Checkpoint checkpoints/breast_tumors/fusion_breast_tumors_epoch100.pth
# ============================================================================

param(
    [Parameter(Mandatory=$true)]
    [string]$Dataset,
    
    [Parameter(Mandatory=$true)]
    [string]$Checkpoint,
    
    [string]$Split = "test",
    
    [string]$PostprocessMethod = "kmeans",
    
    [int]$TopK = 1
)

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "FreqMedCLIP Pipeline with Postprocessing" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Dataset: $Dataset" -ForegroundColor Yellow
Write-Host "Checkpoint: $Checkpoint" -ForegroundColor Yellow
Write-Host "Split: $Split" -ForegroundColor Yellow
Write-Host "Postprocess Method: $PostprocessMethod" -ForegroundColor Yellow
Write-Host ""

# Define output directories
$RawPredDir = "predictions/${Dataset}_raw"
$CleanedPredDir = "predictions/${Dataset}_cleaned"
$VisualizationDir = "visualizations/${Dataset}"

# Step 1: Generate raw predictions
Write-Host "[Step 1/3] Generating raw saliency maps..." -ForegroundColor Green
python save_freqmedclip_predictions.py `
    --dataset $Dataset `
    --checkpoint $Checkpoint `
    --output $RawPredDir `
    --split $Split

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error in Step 1: Failed to generate predictions" -ForegroundColor Red
    exit 1
}

# Step 2: Postprocess predictions
Write-Host ""
Write-Host "[Step 2/3] Postprocessing with $PostprocessMethod..." -ForegroundColor Green
python postprocess_freqmedclip_outputs.py `
    --input $RawPredDir `
    --output $CleanedPredDir `
    --method $PostprocessMethod `
    --top-k $TopK

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error in Step 2: Failed to postprocess predictions" -ForegroundColor Red
    exit 1
}

# Step 3: Generate visualizations (optional)
Write-Host ""
Write-Host "[Step 3/3] Generating visualizations..." -ForegroundColor Green
Write-Host "(This may take a while...)" -ForegroundColor Gray
python visualize_prediction.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️ Warning: Visualization failed, but predictions are ready" -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "✅ Pipeline Complete!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📁 Outputs:" -ForegroundColor Yellow
Write-Host "   - Raw predictions: $RawPredDir" -ForegroundColor White
Write-Host "   - Cleaned predictions: $CleanedPredDir (← Use this for evaluation)" -ForegroundColor Green
Write-Host "   - Visualizations: $VisualizationDir" -ForegroundColor White
Write-Host ""
Write-Host "📊 Next steps:" -ForegroundColor Yellow
Write-Host "   1. Evaluate cleaned predictions:" -ForegroundColor White
Write-Host "      python evaluation/eval.py --pred-dir $CleanedPredDir --gt-dir data/${Dataset}/${Split}_masks" -ForegroundColor Gray
Write-Host ""
Write-Host "   2. Compare with MedCLIP-SAMv2 baseline:" -ForegroundColor White
Write-Host "      python compare_methods.py --freqmedclip $CleanedPredDir --baseline sam_outputs/${Dataset}/masks" -ForegroundColor Gray
Write-Host ""
