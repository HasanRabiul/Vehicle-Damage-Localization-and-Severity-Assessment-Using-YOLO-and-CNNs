# YOLO crops for severity classification

The yolo11n_initial checkpoint was run on every included image of the data3a manifest (imgsz 640, confidence > 0.25, NMS IoU 0.7). Each image produces one crop: the union of its detections, padded by 10% of the union size per side and clipped. Images with no detection fall back to the whole image and are flagged in crops.csv; they are kept, not discarded.

Crops for the test split are created so the pipeline can be evaluated later; no metrics were computed on any split. crop_samples.png shows training/validation examples only. Original images and the checkpoint are unchanged.
