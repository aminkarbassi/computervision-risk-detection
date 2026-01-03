import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import measure


def mask_to_boxes(mask: np.ndarray):
    """
    Convert a binary mask to bounding boxes using connected components.
    Returns list of [xmin, ymin, xmax, ymax].
    """
    boxes = []
    labeled = measure.label(mask, connectivity=2)

    for region in measure.regionprops(labeled):
        min_row, min_col, max_row, max_col = region.bbox
        boxes.append([min_col, min_row, max_col, max_row])

    return boxes


def convert_split(
    image_dir: Path,
    mask_dir: Path,
    out_image_dir: Path,
    out_anno_dir: Path,
):
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_anno_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(image_dir.glob("*.tif")):
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            continue

        image = Image.open(img_path)
        mask = np.array(Image.open(mask_path))

        # Inria masks: buildings are white (255)
        binary_mask = mask > 0

        boxes = mask_to_boxes(binary_mask)

        if len(boxes) == 0:
            continue

        # Save image as JPG for simplicity
        out_img_path = out_image_dir / f"{img_path.stem}.jpg"
        image.convert("RGB").save(out_img_path)

        annotation = {
            "boxes": boxes,
            "labels": [1] * len(boxes),
        }

        out_anno_path = out_anno_dir / f"{img_path.stem}.json"
        with open(out_anno_path, "w") as f:
            json.dump(annotation, f)


def main():
    raw_images = Path("data/raw/inria/images")
    raw_masks = Path("data/raw/inria/masks")

    # VERY SMALL SPLIT for now
    all_images = sorted(raw_images.glob("*.tif"))
    train_imgs = all_images[:10]
    val_imgs = all_images[10:15]

    # Temporary directories
    tmp_train_img = Path("data/tmp/train/images")
    tmp_train_mask = Path("data/tmp/train/masks")
    tmp_val_img = Path("data/tmp/val/images")
    tmp_val_mask = Path("data/tmp/val/masks")

    for p in [tmp_train_img, tmp_train_mask, tmp_val_img, tmp_val_mask]:
        p.mkdir(parents=True, exist_ok=True)

    for img in train_imgs:
        (tmp_train_img / img.name).symlink_to(img.resolve())
        (tmp_train_mask / img.name).symlink_to((raw_masks / img.name).resolve())

    for img in val_imgs:
        (tmp_val_img / img.name).symlink_to(img.resolve())
        (tmp_val_mask / img.name).symlink_to((raw_masks / img.name).resolve())

    convert_split(
        tmp_train_img,
        tmp_train_mask,
        Path("data/detection/train/images"),
        Path("data/detection/train/annotations"),
    )

    convert_split(
        tmp_val_img,
        tmp_val_mask,
        Path("data/detection/val/images"),
        Path("data/detection/val/annotations"),
    )


if __name__ == "__main__":
    main()
