"""Subpacote neural_eq: Treinamento, avaliação e exportação do Neural EQ."""

from neural_eq.constants import (
    GAIN_MAX_DB,
    GAIN_MIN_DB,
    NUM_ERB_BANDS,
    TOTAL_FFT_BINS,
    build_dct_basis,
    build_interpolation_matrix,
)
from neural_eq.dataset import (
    NeuralEqDataset,
    extract_eq_features,
    generate_synthetic_speech,
    load_speech_audio,
)
from neural_eq.export import (
    export_neural_eq_onnx,
    validate_onnx_parity,
    verify_tract_compatibility,
)
from neural_eq.loss import NeuralEqLoss
from neural_eq.losses import (
    MODEL_ERB_WIDTHS,
    NUM_FFT_BINS,
    SpectralEqLoss,
    build_erb_interpolation_matrix,
    curvature_regularizer,
    interpolate_erb_to_bins,
    neutrality_regularizer,
    spectral_huber_loss,
)
from neural_eq.metrics import evaluate_eq_metrics
from neural_eq.model import (
    BoundedNeutralActivation,
    NeuralEqConfig,
    NeuralEqModel,
    NeuralEqNet,
    build_cosine_basis,
)
from neural_eq.train import TrainConfig, train_neural_eq
from neural_eq.transforms import (
    ERB_WIDTHS,
    FFT_SIZE,
    HOP_SIZE,
    SAMPLE_RATE,
    DistortionCurve,
    MicrophoneDistortionGenerator,
    apply_distortion,
    bin_factors_from_erb,
    design_fir_filter,
    get_erb_center_frequencies,
    get_erb_centers,
    get_erb_interpolation_matrix,
)

__all__ = [
    "ERB_WIDTHS",
    "FFT_SIZE",
    "GAIN_MAX_DB",
    "GAIN_MIN_DB",
    "HOP_SIZE",
    "MODEL_ERB_WIDTHS",
    "NUM_ERB_BANDS",
    "NUM_FFT_BINS",
    "SAMPLE_RATE",
    "TOTAL_FFT_BINS",
    "BoundedNeutralActivation",
    "DistortionCurve",
    "MicrophoneDistortionGenerator",
    "NeuralEqConfig",
    "NeuralEqDataset",
    "NeuralEqLoss",
    "NeuralEqModel",
    "NeuralEqNet",
    "SpectralEqLoss",
    "TrainConfig",
    "apply_distortion",
    "bin_factors_from_erb",
    "build_cosine_basis",
    "build_dct_basis",
    "build_erb_interpolation_matrix",
    "build_interpolation_matrix",
    "curvature_regularizer",
    "design_fir_filter",
    "evaluate_eq_metrics",
    "export_neural_eq_onnx",
    "extract_eq_features",
    "generate_synthetic_speech",
    "get_erb_center_frequencies",
    "get_erb_centers",
    "get_erb_interpolation_matrix",
    "interpolate_erb_to_bins",
    "load_speech_audio",
    "neutrality_regularizer",
    "spectral_huber_loss",
    "train_neural_eq",
    "validate_onnx_parity",
    "verify_tract_compatibility",
]
