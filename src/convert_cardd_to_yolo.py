
import json
import shutil
from pathlib import Path
from collections import defaultdict

import yaml

# Project paths
ROOT = Path.cwd()
SOURCE = ROOT / "data/cardd/CarDD/CarDD_COCO"
OUTPUT = ROOT / "data/processed/cardd_yolo"

SPLITS = {
    "train2017": "train",
    "val2017": "val",
    "test2017": "test",
}

# All six CarDD categories become one detection class
CLASS_ID = 0
CLASS_NAME = "damage"


def coco_to_yolo(bbox, image_width, image_height):
    """Convert a COCO xywh box to normalized YOLO xywh."""
    x, y, w, h = map(float, bbox)

    if w <= 0 or h <= 0:
        return None

    # Clip box to image boundaries
    x1 = max(0.0, min(x, image_width))
    y1 = max(0.0, min(y, image_height))
    x2 = max(0.0, min(x + w, image_width))
    y2 = max(0.0, min(y + h, image_height))

    if x2 <= x1 or y2 <= y1:
        return None

    xc = ((x1 + x2) / 2) / image_width
    yc = ((y1 + y2) / 2) / image_height
    bw = (x2 - x1) / image_width
    bh = (y2 - y1) / image_height

    return f"{CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def convert_split(source_split, target_split):
    annotation_file = (
        SOURCE / "annotations" /
        f"instances_{source_split}.json"
    )

    image_source = SOURCE / source_split
    image_output = OUTPUT / "images" / target_split
    label_output = OUTPUT / "labels" / target_split

    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    with open(annotation_file, "r") as f:
        coco = json.load(f)

    image_info = {
        image["id"]: image
        for image in coco["images"]
    }

    annotations_by_image = defaultdict(list)

    for annotation in coco["annotations"]:
        annotations_by_image[annotation["image_id"]].append(
            annotation
        )

    converted_images = 0
    converted_boxes = 0
    skipped_boxes = 0
    missing_images = 0

    for image_id, info in image_info.items():
        filename = Path(info["file_name"]).name
        source_path = image_source / info["file_name"]

        if not source_path.is_file():
            print(f"Missing image: {source_path}")
            missing_images += 1
            continue

        width = info["width"]
        height = info["height"]

        destination = image_output / filename
        shutil.copy2(source_path, destination)

        lines = []

        for annotation in annotations_by_image[image_id]:
            result = coco_to_yolo(
                annotation["bbox"], width, height
            )

            if result is not None:
                lines.append(result)
                converted_boxes += 1
            else:
                skipped_boxes += 1

        label_path = label_output / (
            Path(filename).stem + ".txt"
        )

        # Empty labels are valid for images without damage
        label_path.write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )

        converted_images += 1

    print(f"\n{target_split.upper()}")
    print(f"  Images converted: {converted_images}")
    print(f"  Bounding boxes:   {converted_boxes}")
    print(f"  Skipped boxes:    {skipped_boxes}")
    print(f"  Missing images:   {missing_images}")

    return missing_images


def main():
    if not SOURCE.is_dir():
        raise FileNotFoundError(
            f"CarDD folder not found: {SOURCE}"
        )

    total_missing = 0

    for source_split, target_split in SPLITS.items():
        total_missing += convert_split(
            source_split, target_split
        )

    config = {
        "path": str(OUTPUT.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": {0: CLASS_NAME},
    }

    with open(OUTPUT / "data.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print("\nYOLO dataset created:")
    print(OUTPUT)
    print("\nConfiguration:")
    print(OUTPUT / "data.yaml")

    if total_missing:
        print(
            f"\nWARNING: {total_missing} images were missing."
        )


if __name__ == "__main__":
    main()

