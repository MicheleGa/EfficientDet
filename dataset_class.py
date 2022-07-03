import os
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


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

        # get boxes from dataframe rows and
        # be ware that Pytorch implementation of EfficientDet, require data in
        # format YXYX and not XYXY!
        rows = self.dataset[self.dataset['image_id'] == image_id]
        boxes = rows[['x_min', 'y_min', 'x_max', 'y_max']].values

        # there is only one class
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

        sample = {
            'image': np.array(image, dtype=np.float32),
            'bboxes': boxes,
            'labels': labels,
        }

        target = {}

        for i in range(10):
            sample = self.transforms(**sample)
            if len(sample['bboxes']) > 0:
                target['bboxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                # yxyx switch
                target['bboxes'][:, [0, 1, 2, 3]] = target['bboxes'][:, [1, 0, 3, 2]]
                break

        # image normalized
        image = sample['image']
        labels = sample['labels']

        # img_size and img_scale needed by the model during evaluation
        target['labels'] = torch.as_tensor(labels)
        target['image_id'] = torch.tensor([idx])
        target['img_size'] = (height, width)
        target['img_scale'] = torch.tensor([1.0])

        return image, target, f'{image_id}.jpg'

    def __len__(self) -> int:
        return self.dataset.shape[0]


def get_data_transforms(input_img_size=1024, img_size=512):
    """
    Return transformations to be applied to train and validation data in roder to perform data
    augmentation, using Albumentations library.

    Args:
        input_img_size: image shape before transformations (1024,1024).
        img_size: image shape before entering EfficientDet d0 model (512, 512).

    Return
        data_transforms: dictionary with train and val keys, with associate pipeline with transformations.
    """

    # common datasets hyperparameters
    mean_image_net = [0.485, 0.456, 0.406]
    std_image_net = [0.229, 0.224, 0.225]

    # Albumentations library has a Normalize with default mean and std = to ImageNet values,
    # moreover, bbox_params of Compose allows to Normalize coordinates by dividing for input
    # image size (input_img_size) in order to work with small floats instead with big integers
    # in range 0 - 1024
    data_transforms = {'train': A.Compose([A.HorizontalFlip(p=0.5),
                                           A.VerticalFlip(p=0.5),
                                           A.Resize(height=img_size,
                                                    width=img_size,
                                                    p=1),
                                           A.Normalize(mean=mean_image_net,
                                                       std=std_image_net),
                                           ToTensorV2(p=1)],
                                          p=1,
                                          bbox_params=A.BboxParams(format='pascal_voc',
                                                                   min_area=0,
                                                                   min_visibility=0,
                                                                   label_fields=['labels'])),

                       'val': A.Compose([A.Resize(height=img_size,
                                                  width=img_size,
                                                  p=1),
                                         A.Normalize(mean=mean_image_net,
                                                     std=std_image_net),
                                         ToTensorV2(p=1)],
                                        p=1,
                                        bbox_params=A.BboxParams(format='pascal_voc',
                                                                 min_area=0,
                                                                 min_visibility=0,
                                                                 label_fields=['labels']))
                       }

    return data_transforms
