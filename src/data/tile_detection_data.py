import json
from pathlib import Path

import numpy as np
from PIL import Image


TILE_SIZE = 512


def clip_box(box, x0, y0, x1, y1):
    xmin, ymin, xmax, ymax = box
    xmin = max(xmin, x0)
    ymin = max(ymin, y0)
    xmax = min(xmax, x1)
    ymax = min(ymax, y1)
    if xmax <= xmin or ymax <= ymin:
        return None
    return [xmin - x0, ymin - y0, xmax - x0, ymax - y0]


def tile_image(
    image_path: Path,
    anno_path: Path,
    out_img_dir: Path,
    out_anno_dir: Path,
):
    image = Image.open(image_path).convert("RGB")
    W, H = image.size

    with open(anno_path) as f:
        anno = json.load(f)

    boxes = anno["boxes"]

    tile_id = 0
    for y0 in range(0, H, TILE_SIZE):
        for x0 in range(0, W, TILE_SIZE):
            x1 = min(x0 + TILE_SIZE, W)
            y1 = min(y0 + TILE_SIZE, H)

            tile = image.crop((x0, y0, x1, y1))

            tile_boxes = []
            for box in boxes:
                clipped = clip_box(box, x0, y0, x1, y1)
                if clipped is not None:
                    tile_boxes.append(clipped)

            if len(tile_boxes) == 0:
                continue

            tile_name = f"{image_path.stem}_tile_{tile_id}"
            tile.save(out_img_dir / f"{tile_name}.jpg")

            tile_anno = {
                "boxes": tile_boxes,
                "labels": [1] * len(tile_boxes),
            }

            with open(out_anno_dir / f"{tile_name}.json", "w") as f:
                json.dump(tile_anno, f)

            tile_id += 1


def process_split(split: str):
    in_img_dir = Path(f"data/detection/{split}/images")
    in_anno_dir = Path(f"data/detection/{split}/annotations")

    out_img_dir = Path(f"data/detection_tiled/{split}/images")
    out_anno_dir = Path(f"data/detection_tiled/{split}/annotations")

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_anno_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(in_img_dir.glob("*.jpg")):
        anno_path = in_anno_dir / f"{img_path.stem}.json"
        tile_image(img_path, anno_path, out_img_dir, out_anno_dir)


def main():
    process_split("train")
    process_split("val")


if __name__ == "__main__":
    main()
