# Models

Trained weights and checkpoints. **Git-ignored**. Only this README is tracked.

```
models/
└── <task>/                 # e.g. maturity_cls
    └── <run_id>/           # YYYYMMDD-HHMM_<short-description>
        ├── config.yaml     # fully resolved config used for the run
        ├── weights.*       # .pt / .safetensors / .onnx
        └── metrics.json    # val/test metrics + gold dataset version
```

Override the location with `STRAWBERRY_MODELS_DIR`. Pretrained backbones are
downloaded to the Hugging Face / torch caches, not here.
