import os
from zipfile import ZipFile
import pandas as pd


def unzip_dataset(dataset_path='./dataset', name='GlobalWheatDetection') -> None:
    """
    Extract zipped dataset in order to get a folder with train/test images and a
    csv file with annotations.

    Args:
        dataset_path: path to the dataset directory
        name: name of the specific dataset
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
        else:
            with ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(path)


def pre_process_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust pandas dataframe with annotations in order to get for each bounding box its image id and its
    coordinates in the image.

    Args:
        df: annotations for training samples
    Return:
        df: dataframe without correctly formatted annotations
    """

    # Adjust bbox column format
    df['x_min'] = df['bbox'].apply(lambda x: float(re.findall(r'[0-9.]+', x.split(',')[0])[0]))
    df['y_min'] = df['bbox'].apply(lambda x: float(re.findall(r'[0-9.]+', x.split(',')[1])[0]))
    df['box_width'] = df['bbox'].apply(lambda x: float(re.findall(r'[0-9.]+', x.split(',')[2])[0]))
    df['box_height'] = df['bbox'].apply(lambda x: float(re.findall(r'[0-9.]+', x.split(',')[3])[0]))
    df.drop('bbox', axis=1, inplace=True)
    df['x_max'] = df['x_min'] + df['box_width']
    df['y_max'] = df['y_min'] + df['box_height']

    return df[['image_id', 'x_min', 'y_min', 'x_max', 'y_max']]