# ResNet50 severity classifier (whole, seed 814)

Input: whole data3a images from the filtered manifest. Training and validation splits only; no test image was loaded.

ImageNet ResNet50 (IMAGENET1K_V2), 3-class head. Phase 1 trains the head for 5 epochs with the backbone frozen; phase 2 fine-tunes all layers (backbone lr 1e-4, head 1e-3, cosine) with early stopping. best.pt is the epoch with the highest validation macro-F1, so validation metrics are optimistic for model selection.

val_metrics.json also splits validation results into images where YOLO detected damage and fallback images. Same-image groups are identical across variants, so the breakdown is comparable.
