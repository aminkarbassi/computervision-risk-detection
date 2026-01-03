import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches


def visualize(image_path: Path, anno_path: Path):
    image = Image.open(image_path).convert("RGB")

    with open(anno_path) as f:
        anno = json.load(f)

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)

    for box in anno["boxes"]:
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)

    ax.set_axis_off()
    plt.show()


def main():
    img_dir = Path("data/detection/train/images")
    anno_dir = Path("data/detection/train/annotations")

    images = sorted(img_dir.glob("*.jpg"))[:3]

    for img_path in images:
        anno_path = anno_dir / f"{img_path.stem}.json"
        print(f"Visualizing {img_path.name}")
        visualize(img_path, anno_path)


if __name__ == "__main__":
    main()
