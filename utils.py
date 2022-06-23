import os
import numpy as np
import torch
import random
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def fix_random(seed: int) -> None:
    """
    Fix all the possible sources of randomness.

    Args:
        seed: the seed to use.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def show_images(df: pd.DataFrame,
                idx: int,
                folder: str,
                title: str,
                linecolor: str) -> None:
    """
    Function to show images with detected wheat heads.

    Args:
        df: annotations dataframe.
        idx: id of the image to display.
        folder: specify folder to the images (train/test).
        title: title of the matplotlib image.
        linecolor: color of the bbox.
    """
    fig, axs = plt.figsize = (10, 10)

    image_id = df.iloc[idx, :].image_id

    path = os.path.join(folder, image_id + '.jpg')
    image = Image.open(path)

    objects = df[df['image_id'] == image_id][['x_min', 'x_max', 'y_min', 'y_max']].values

    # drawing on the Image
    draw = ImageDraw.Draw(image)

    for box in objects:
        draw.rectangle([box[0], box[2], box[1], box[3]], width=10, outline=linecolor)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.suptitle(title)
    plt.savefig(f'./{title} - idx {idx}.jpg')
    plt.clf()


def count_parameters(model: torch.nn.Module) -> int:
    """
    Counts parameters of the object detector.

    Args:
        model: object detector.
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_requires_grad_for_layer(model: torch.nn.Module, train: bool) -> None:
    """
    Sets the attribute requires_grad to True or False for each parameter.

    Args:
        model: object detector.
        train: if true train the model parameter.
    """
    for p in model.parameters():
        p.requires_grad = train
