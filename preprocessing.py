import os
import re
from zipfile import ZipFile
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset_class import GWDataset
from utils import show_images


def unzip_dataset(dataset_path='./dataset', name='GlobalWheatDetection/') -> None:
    """
    Extract zipped dataset in order to get a folder with train/test images and a
    csv file with annotations.

    Args:
        dataset_path: path to the dataset directory.
        name: name of the specific dataset.

    """

    path = os.path.join(dataset_path, name)

    train_dir = os.path.join(path, 'train')
    os.makedirs(train_dir)

    test_dir = os.path.join(path, 'test')
    os.makedirs(test_dir)

    for file in os.listdir(path):

        file_path = os.path.join(path, file)

        if file == 'train.zip':
            with ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(train_dir)
        elif file == 'test.zip':
            with ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(test_dir)
        elif file == 'train.csv.zip':
            with ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(path)


def pre_process_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust pandas dataframe with annotations in order to get for each bounding box its image id and its
    coordinates in the image.

    Args:
        df: annotations for training samples.

    Return:
        df: dataframe without correctly formatted annotations.

    """

    # Adjust bbox column format
    df['x_min'] = df['bbox'].apply(lambda x: float(re.findall(r'[0-9.]+', x.split(',')[0])[0]))
    df['y_min'] = df['bbox'].apply(lambda x: float(re.findall(r'[0-9.]+', x.split(',')[1])[0]))
    df['box_width'] = df['bbox'].apply(lambda x: float(re.findall(r'[0-9.]+', x.split(',')[2])[0]))
    df['box_height'] = df['bbox'].apply(lambda x: float(re.findall(r'[0-9.]+', x.split(',')[3])[0]))
    df.drop('bbox', axis=1, inplace=True)

    # switch to PASCAL VOC annotation format
    df['x_max'] = df['x_min'] + df['box_width']
    df['y_max'] = df['y_min'] + df['box_height']

    return df[['image_id', 'x_min', 'y_min', 'x_max', 'y_max']]


def data_preparation(data_transforms,
                     original_img_size,
                     num_workers=1,
                     batch_size=16,
                     train_path='./dataset',
                     name='GlobalWheatDetection') -> [DataLoader, DataLoader]:
    """
    Data pre-processing step where raw images and annotations are extracted from the
    zipped dataset and then organized into a torch Dataset class.

    Args:
        data_transforms: dictionary with train/val transformation.
        original_img_size: input size of the images extracted (1024,1024).
        num_workers: number of parallel threads to load data.
        batch_size: batch size.
        train_path: where zip file are stored.
        name: name of the dataset directory.

    Return:
        pair composed by train data loader and validation data loader.

    """

    # unzip dataset
    unzip_dataset()

    # process file with annotations
    annotations = pd.read_csv(os.path.join(train_path, name, 'train.csv'))
    print('Dataset head before preprocessing:')
    print(annotations.head())

    annotations = pre_process_annotations(annotations)
    print('Dataset head after preprocessing:')
    print(annotations.head())

    # splitting the dataset
    train_set, val_set = train_test_split(annotations, test_size=0.2, random_state=42)

    # get train and validation data loaders
    path_images = os.path.join(train_path, name, 'train')
    train_dataset = GWDataset(path_images=path_images,
                              dataset=train_set,
                              original_img_size=original_img_size,
                              transforms=data_transforms['train'])

    validation_dataset = GWDataset(path_images=path_images,
                                   dataset=val_set,
                                   original_img_size=original_img_size,
                                   transforms=data_transforms['val'])
    # show a data sample with bbox
    idx = 100
    show_images(annotations, idx, path_images, 'Wheat Head Example', 'red')

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=num_workers,
                              collate_fn=collate_fn)

    valid_loader = DataLoader(validation_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              drop_last=True,
                              num_workers=num_workers,
                              collate_fn=collate_fn)

    return train_loader, valid_loader
