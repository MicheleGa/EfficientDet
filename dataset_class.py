import os
import numpy as np
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
        self.original_img_size = original_img_size
        self.transforms = transforms

    def __getitem__(self, idx):

        # get image as a PIL object
        image_id = self.dataset.iloc[idx]['image_id']
        path_image = os.path.join(self.path_images, f'{image_id}.jpg')
        image = Image.open(path_image).convert("RGB")
        height, width = image.size

        # get boxes from dataframe rows
        rows = self.dataset[self.dataset['image_id'] == image_id]
        boxes = rows[["x_min", "y_min", "x_max", "y_max"]].values

        # there is only one class
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

        sample = {
            'image': np.array(image, dtype=np.float32),
            'bboxes': boxes,
            'labels': labels,
        }

        if self.transforms is not None:
            sample = self.transforms(**sample)

        # input for the model
        image = sample['image']

        # Process annotations.
        # Be ware that Pytorch implementation of EfficientDet, require data in
        # format YXYX and not XYXY!
        # Need also to record image size and scale, needed during evaluation to get
        # detector prediction in the right format.
        sample['bboxes'][:, [0, 1, 2, 3]] = np.array(sample['bboxes'][:, [1, 0, 3, 2]])

        target = {
            'bboxes': torch.as_tensor(sample['bboxes'], dtype=torch.float32),
            'labels': torch.as_tensor(sample['labels']),
            'image_id': torch.tensor([idx]),
            'img_size': (height, width),
            'img_scale': torch.tensor([1.0])
        }

        return image, target

    def __len__(self) -> int:
        return self.dataset.shape[0]
