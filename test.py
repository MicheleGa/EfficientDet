import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader
from model import model_definition
from postprocessing import get_test_images, get_predicted_df
from utils import show_images


def test_model(save_path: str,
               annotations: pd.DataFrame,
               efficient_det_model_name: str,
               data_transforms: DataLoader,
               device: str):
    efficient_det_model = model_definition(inference=True,
                                           checkpoint_path=save_path,
                                           model_name=efficient_det_model_name,
                                           pretrained_backbone=False)
    efficient_det_model = efficient_det_model.to(device)

    # get unseen data
    images_tensor, test_images_ids, test_image_sizes = get_test_images(data_transforms=data_transforms)
    num_images = len(test_images_ids)

    images_tensor = images_tensor.float().to(device)
    img_info = {
        'img_size': torch.tensor(test_image_sizes).float().to(device),
        'img_scale': torch.ones(num_images).float().to(device)
    }

    # test model
    efficient_det_model.eval()
    with torch.no_grad():
        detections = efficient_det_model(images_tensor.to(device),
                                         img_info=img_info)

        detections = detections.cpu().numpy()

    predicted_data = get_predicted_df(num_images=num_images,
                                      detections=detections,
                                      test_image_sizes=test_image_sizes,
                                      test_images_ids=test_images_ids)

    # show predicted bboxes on unseen data
    # idx is 0...9
    idx = 0
    show_images(df=predicted_data,
                idx=idx,
                folder=os.path.join('./dataset/GlobalWheatDetection', 'test'),
                title='Wheat Head Prediction On Unseen Data',
                linecolor='red')

    # compare with a train sample
    idx = 100
    image_id = annotations.iloc[idx, :].image_id
    path = os.path.join('./dataset/GlobalWheatDetection/train', image_id + '.jpg')
    image = Image.open(path)
    h, w = image.size

    image_tensor = data_transforms['val'](
        image=np.array(image, dtype=np.float32),
        labels=np.ones(1),
        bboxes=np.array([[0, 0, 1, 1]])  # fake, just need to make image transformation work
    )['image']

    image_tensor = torch.stack([image_tensor]).float().to(device)
    img_info = {
        'img_size': torch.tensor([(h, w)]).float().to(device),
        'img_scale': torch.ones(1).float().to(device)
    }

    with torch.no_grad():
        detections = efficient_det_model(image_tensor,
                                         img_info=img_info)
        detections = detections.cpu().numpy()

    predicted_data = get_predicted_df(num_images=1,
                                      detections=detections,
                                      test_image_sizes=[(h, w)],
                                      test_images_ids=image_id)

    fig, axs = plt.figsize = (10, 10)

    image = Image.open(path)

    objects = predicted_data[['x_min', 'x_max', 'y_min', 'y_max']].values
    # Drawing on the Image
    draw = ImageDraw.Draw(image)

    for box in objects:
        draw.rectangle([box[0], box[2], box[1], box[3]], width=2, outline='red')

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')

    plt.suptitle('Wheat Head Prediction on a Train Example')
    plt.savefig(f'./Wheat Head Train Example - idx {idx}.jpg')
    plt.clf()
