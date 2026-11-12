"""Run trained models on new images and return maturity predictions."""

from strawberry_picker.inference.predict import (
    FruitPrediction,
    MaturityRun,
    annotate_image,
    latest_maturity_run,
    load_maturity_run,
    predict_image,
    predictions_table,
)

__all__ = [
    "FruitPrediction",
    "MaturityRun",
    "annotate_image",
    "latest_maturity_run",
    "load_maturity_run",
    "predict_image",
    "predictions_table",
]
