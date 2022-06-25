import os
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset


class GWDataset(Dataset):
    def __init__(
            self,
            path_images: str,
            dataset: pd.DataFrame,
            original_img_size: int,
            transforms: torchvision.transforms = None,
    ) -> None:
        """
        Init the dataset.

        Args:
            path_images: the path to the folder containing the images.
            dataset: pandas dataframe with the annotations.
            transforms: the transformation to apply to the dataset.
        """
        self.path_images = path_images
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, idx):
        image_id = self.dataset.iloc[idx]['image_id']
        path_image = os.path.join(self.path_images, f'{image_id}.jpg')
        image = Image.open(path_image).convert("RGB")

        # Convert everything into a torch tensor
        # switch to yxyx be ware!
        boxes = torch.tensor(self.dataset[self.dataset['image_id'] == image_id]
                             [['y_min', 'x_min', 'y_max', 'x_max']].values, dtype=torch.float32)

        # normalize coordinates (as in Albumentations library for PASCAL VOC format)
        boxes /= self.original_img_size

        # There is only one class
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

        # need also to record image size and scale, needed during evaluation to get
        # detector prediction in the right format

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'img_size': image.size,
            'img_scale': torch.tensor([1.0])
        }

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

    def __len__(self) -> int:
        return self.dataset.shape[0]
