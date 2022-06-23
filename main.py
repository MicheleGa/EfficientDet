import os
import time
import copy
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from PIL import Image, ImageDraw, ImageFont
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Tuple, Union
import tqdm
import datetime
from sklearn.model_selection import train_test_split
import timm
from timm.utils.metrics import accuracy
from ensemble_boxes import nms, weighted_boxes_fusion
from dataset_class import GWDataset
from preprocessing import unzip_dataset, pre_process_annotations
from utils import fix_random, show_images


def data_preparation() -> [DataLoader, DataLoader]:

    # unzip dataset
    unzip_dataset()

    # process file with annotations
    annotations = pd.read_csv('./dataset/GlobalWheatDetection/train.csv')
    print(annotations.head())

    annotations = pre_process_annotations(annotations)
    print(annotations.head())

    # splitting the dataset
    train_set, val_set = train_test_split(annotations, test_size=0.2, random_state=42)

    # get train and validation dataloaders
    train_dataset = GWDataset(path_images='/content/GlobalWheatDetection/train',
                              dataset=train_set,
                              transforms=data_transforms['train'])

    validation_dataset = GWDataset(path_images='/content/GlobalWheatDetection/train',
                                   dataset=val_set,
                                   transforms=data_transforms['val'])
    # show a data sample with bbox
    idx = 100
    show_images(annotations, idx, '/content/GlobalWheatDetection/train', 'Wheat Head Example', 'red')

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(validation_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              collate_fn=collate_fn)

    return train_loader, valid_loader

if __name__ == '__main__':

    # experiments reproducibility
    magic_seed = 42
    fix_random(seed=magic_seed)

    # device
    device = "cpu"
    if torch.cuda.is_available:
        print('Gpu available')
        device = torch.device("cuda:0")
    else:
        print('Please set GPU via Edit -> Notebook Settings.')

    # common datasets hyperparameters
    mean_image_net = [0.485, 0.456, 0.406]
    std_image_net = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean_image_net, std_image_net)

    # image sizes for efficient det d0
    resize_crop = 512
    image_crop = 512

    # quite standard augmentations
    data_transforms = {'train': transforms.Compose([transforms.RandomResizedCrop(resize_crop),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    normalize]),

                       'val': transforms.Compose([transforms.Resize(image_crop),
                                                  transforms.ToTensor(),
                                                  normalize])}

    # hyperparameters adapted to resource availability :(
    num_workers = 2
    batch_size = 8

    # background + white head classes
    num_classes = 2

    train_loader, valida_loader = data_preparation()





