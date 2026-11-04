
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

ROOT = Path.cwd()
DATA = ROOT / "data/processed/cardd_yolo"

images_dir = DATA / "images/train"
labels_dir = DATA / "labels/train"

images = [
    p for p in images_dir.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
]

random.seed(42)
samples = random.sample(images, min(6, len(images)))

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for ax, image_path in zip(axes.flat, samples):
    image = cv2.imread(str(image_path))

    if image is None:
        ax.set_title("Image could not be loaded")
        ax.axis("off")
        continue

    height, width = image.shape[:2]
    label_path = labels_dir / (image_path.stem + ".txt")

    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()

            if len(parts) != 5:
                continue

            _, xc, yc, bw, bh = map(float, parts)

            x1 = int((xc - bw / 2) * width)
            y1 = int((yc - bh / 2) * height)
            x2 = int((xc + bw / 2) * width)
            y2 = int((yc + bh / 2) * height)

            cv2.rectangle(
                image, (x1, y1), (x2, y2),
                (0, 255, 0), 2
            )

            cv2.putText(
                image, "damage", (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2
            )

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax.imshow(image_rgb)
    ax.set_title(image_path.name)
    ax.axis("off")

plt.tight_layout()

output = ROOT / "outputs/figures/cardd_annotations.png"
output.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output, dpi=200, bbox_inches="tight")
plt.show()

print(f"Saved visualization: {output}")

