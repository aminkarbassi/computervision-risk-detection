from src.data.detection_dataset import DetectionDataset
from torchvision.transforms import ToTensor

ds = DetectionDataset(
    image_dir="data/detection_tiled/train/images",
    annotation_dir="data/detection_tiled/train/annotations",
    transforms=ToTensor(),
)

print("Dataset size:", len(ds))

img, target = ds[0]
print("Image shape:", img.shape)
print("Boxes:", target["boxes"].shape)
print("Labels:", target["labels"])
