"""
Frequency-Aware Integration Module for MedCLIP-SAMv2

This package implements the Frequency-Aware Integration pipeline that enhances
MedCLIP-SAMv2 with wavelet-based boundary detection capabilities.

Components:
1. wavelet_preprocessor: Dual-stream preprocessing with wavelet transforms
2. feature_fusion: Feature fusion module for semantic + boundary information
3. saliency_generation: Refined saliency map generation
4. postprocessing: Post-processing, ROI extraction, and SAM integration
"""

from .wavelet_preprocessor import (
    WaveletTransform,
    DualStreamPreprocessor,
    process_batch_dual_stream
)

from .feature_fusion import (
    HighFreqProjection,
    FeatureFusionGate,
    FrequencyAwareVisionEncoder,
    make_frequency_aware
)

from .saliency_generation import (
    FrequencyAwareSaliencyGenerator,
    MultiScaleSaliencyGenerator,
    post_process_saliency_map
)

from .postprocessing import (
    ROIExtractor,
    SAMPromptGenerator,
    MaskRefinement,
    FrequencyAwarePipeline,
    SAMPrompt,
    draw_prompts,
    visualize_segmentation_results
)

__version__ = "1.0.0"
__author__ = "Frequency-Aware Integration Team"

__all__ = [
    # Wavelet Preprocessing
    'WaveletTransform',
    'DualStreamPreprocessor',
    'process_batch_dual_stream',
    
    # Feature Fusion
    'HighFreqProjection',
    'FeatureFusionGate',
    'FrequencyAwareVisionEncoder',
    'make_frequency_aware',
    
    # Saliency Generation
    'FrequencyAwareSaliencyGenerator',
    'MultiScaleSaliencyGenerator',
    'post_process_saliency_map',
    
    # Post-processing
    'ROIExtractor',
    'SAMPromptGenerator',
    'MaskRefinement',
    'FrequencyAwarePipeline',
    'SAMPrompt',
    'draw_prompts',
    'visualize_segmentation_results',
]
