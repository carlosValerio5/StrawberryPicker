"""Run trained models on new images and return maturity, defects and grade per berry."""

from strawberry_picker.inference.pipeline import (
    BerryPrediction,
    LockedSetError,
    Predictor,
    UnsupportedImageError,
    annotate,
    ensure_not_locked,
    image_paths,
    load_image,
    load_predictor,
    predict_image,
    predict_paths,
    quality_head,
    resolve_policy,
    run_policy,
)

__all__ = [
    "BerryPrediction",
    "LockedSetError",
    "Predictor",
    "UnsupportedImageError",
    "annotate",
    "ensure_not_locked",
    "image_paths",
    "load_image",
    "load_predictor",
    "predict_image",
    "predict_paths",
    "quality_head",
    "resolve_policy",
    "run_policy",
]
