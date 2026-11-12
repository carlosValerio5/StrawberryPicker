"""Gradio demo: uv run --with gradio python scripts/demo_maturity.py"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

from strawberry_picker.inference.predict import (
    annotate_image,
    load_maturity_run,
    predict_image,
    predictions_table,
)

_MAX_SIDE = 640


def _fit(image: Image.Image, max_side: int = _MAX_SIDE) -> Image.Image:
    rgb = image.convert("RGB")
    w, h = rgb.size
    scale = min(1.0, max_side / max(w, h))
    if scale >= 1.0:
        return rgb
    return rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)


@lru_cache(maxsize=1)
def _embedder(backbone: str):
    from strawberry_picker.audit.embeddings import dinov2_embedder

    return dinov2_embedder(backbone)


@lru_cache(maxsize=1)
def _detector():
    from strawberry_picker.labeling.proposals import owlv2_detector

    return owlv2_detector()


def _build_app(run_dir: Path | None, share: bool, server_port: int) -> Any:
    import gradio as gr

    run = load_maturity_run(run_dir)
    spanish = {
        "unripe": "inmadura (unripe)",
        "ripe": "madura (ripe)",
        "overripe": "sobremadura (overripe)",
    }

    def predict(photo: Image.Image | None, detect: bool, min_score: float):
        if photo is None:
            raise gr.Error("Sube una foto de fresa.")
        photo = _fit(photo)
        preds = predict_image(
            photo,
            run.probe,
            _embedder(run.backbone),
            run.classes,
            detect_fn=_detector() if detect else None,
            min_score=float(min_score),
        )
        annotated = annotate_image(photo, preds)
        table = predictions_table(preds)
        summary = " · ".join(
            f"#{i} {spanish.get(p.label, p.label)} {p.confidence:.0%}"
            for i, p in enumerate(preds, start=1)
        )
        return summary or "sin predicciones", table, annotated

    with gr.Blocks(title="StrawberryCheck") as app:
        gr.Markdown("# StrawberryCheck — demo de madurez")
        with gr.Row():
            with gr.Column(scale=1):
                photo = gr.Image(
                    type="pil",
                    label="Foto",
                    sources=["upload"],
                    height=280,
                )
                detect = gr.Checkbox(value=False, label="Detectar fresas (OWLv2)")
                min_score = gr.Slider(0.05, 0.6, value=0.2, step=0.05, label="Umbral detección")
                go = gr.Button("Clasificar", variant="primary")
            with gr.Column(scale=1):
                out_summary = gr.Textbox(label="Resumen", interactive=False)
                out_table = gr.Dataframe(
                    headers=["#", "clase", "confianza", "det", "probs"],
                    label="Detalle por fresa",
                    interactive=False,
                )
                out_img = gr.Image(type="pil", label="Predicción", height=280)
        outputs = [out_summary, out_table, out_img]
        go.click(predict, inputs=[photo, detect, min_score], outputs=outputs)
        photo.upload(predict, inputs=[photo, detect, min_score], outputs=outputs)
    return app.launch(share=share, server_port=server_port)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="model run folder (default: newest)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Gradio public link (optional)")
    args = parser.parse_args()
    _build_app(args.run, args.share, args.port)


if __name__ == "__main__":
    main()
