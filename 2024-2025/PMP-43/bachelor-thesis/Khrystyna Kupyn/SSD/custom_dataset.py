import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transforms=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        ann_path = os.path.join(self.annotation_dir, img_name.replace(".jpg", ".json"))

        img = Image.open(img_path).convert("RGB")

        with open(ann_path) as f:
            data = json.load(f)

        boxes = torch.tensor(data["boxes"], dtype=torch.float32)
        labels = torch.tensor(data["labels"], dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.images)