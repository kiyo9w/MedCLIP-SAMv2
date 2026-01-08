# FreqMedCLIP Scripts - M2IB Saliency + SAM Architecture

# Core frequency components
from freqmedclip.scripts.freq_components import haar_dwt, FrequencyEncoder, FPNAdapter

# DWT Projector for frequency injection
from freqmedclip.scripts.dwt_projector import DWTFrequencyProjector, DWTHighFreqOnly

# M2IB Module for text-image fusion
from freqmedclip.scripts.m2ib_module import M2IBModule, M2IBModuleWithSpatialFeatures

# Spatial Gate Attention
from freqmedclip.scripts.spatial_gate import SpatialGateAttention, SpatialGateAttentionV2, SimpleSpatialGate

# Prompt Extractor for SAM
from freqmedclip.scripts.prompt_extractor import PromptExtractor, DifferentiablePromptExtractor

# SAM Wrapper
from freqmedclip.scripts.sam_wrapper import SAMWrapper, DummySAM

# Full Model - FMISeg-inspired with LFFI, FFBI, Dual Branch
from freqmedclip.scripts.saliency_sam_model import SaliencyModel

__all__ = [
    # Frequency components
    'haar_dwt',
    'FrequencyEncoder', 
    'FPNAdapter',
    'DWTFrequencyProjector',
    'DWTHighFreqOnly',
    # M2IB
    'M2IBModule',
    'M2IBModuleWithSpatialFeatures',
    # Spatial Gate
    'SpatialGateAttention',
    'SpatialGateAttentionV2',
    'SimpleSpatialGate',
    # Prompt Extraction
    'PromptExtractor',
    'DifferentiablePromptExtractor',
    # SAM
    'SAMWrapper',
    'DummySAM',
    # Full Models
    'SaliencyModel',
]
