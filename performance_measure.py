import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from ensemble_boxes import weighted_boxes_fusion


def get_test_images(dataset_path='./dataset',
                    name='GlobalWheatDetection',
                    data_transforms=None):
    """
    Collect the 10 test images in a list of PIL objects

    Args:
        dataset_path: dataset folder.
        name: specific dataset.
        data_transforms: validation transformation.
    Return:
        images: torch tensor with the 10 PIL images.
        images_id: identifier of each image.
    """
    # gather test images in a list of PIL objects
    test_path = os.path.join(dataset_path, name, 'test')

    test_image_ids = []
    images = []
    for image_id in os.listdir(test_path):
        path_image = os.path.join(test_path, f'{image_id}')
        images.append(Image.open(path_image).convert('RGB'))
        test_image_ids.append(image_id[:-4])

    # as in validation during training, size needed
    test_image_sizes = [(image.size[1], image.size[0]) for image in images]

    images_tensor = []
    for image in images:
        images_tensor.append(
            data_transforms['val'](image)
        )

    return torch.stack(images_tensor), test_image_ids, test_image_sizes


def post_process(detections,
                 image_sizes,
                 test_image_ids,
                 confidence_threshold=0.2,
                 iou_threshold=0.3,
                 image_crop=512) -> pd.DataFrame:
    """
    Process prediction of the efficient det, considering that each box have to be rescaled in
    order to be displayed over the test image in order to see its effectiveness.

    Args:
        detections: detected bboxes by the fine-tuned efficient det.
        image_sizes: list of pair (height, width) for each image in order to rescale bboxes correctly.
        test_image_ids: image ids to compose the final df.
        confidence_threshold: confidence threshold.
        iou_threshold: intersection over union threshold.
        image_crop: as for image_sizes is needed to rescale bboxes.
    Return:
        df: dataframe that has bboxes on the row, for each bbox image id and coordinates of the bbox are reported.
    """

    # model's output processing to display predicted boxes
    prediction_boxes = []
    prediction_scores = []
    prediction_classes = []

    for i in range(len(detections)):
        boxes = detections[i][:, :4]
        scores = detections[i][:, 4]
        classes = detections[i][:, 5]

        indexes = np.where(scores > confidence_threshold)[0]
        boxes = boxes[indexes] / image_crop
        scores = scores[indexes]
        classes = classes[indexes]

        boxes = [boxes.tolist()]
        scores = [scores.tolist()]
        classes = [classes.tolist()]

        boxes, scores, classes = weighted_boxes_fusion(
            boxes,
            scores,
            classes,
            iou_thr=iou_threshold
        )
        boxes = boxes * (image_crop - 1)

        prediction_boxes.append(boxes.tolist())
        prediction_scores.append(scores.tolist())
        prediction_classes.append(classes.tolist())

    scaled_bboxes = []
    for bboxes, img_dims in zip(prediction_boxes, image_sizes):
        im_h, im_w = img_dims

        if len(bboxes) > 0:
            scaled_bboxes.append(
                (
                        np.array(bboxes)
                        * [
                            im_w / image_crop,
                            im_h / image_crop,
                            im_w / image_crop,
                            im_h / image_crop,
                        ]
                ).tolist()
            )
        else:
            scaled_bboxes.append(bboxes)

    predicted_data = {'image_id': [],
                      'x_min': [],
                      'y_min': [],
                      'x_max': [],
                      'y_max': []
                      }

    for i in range(len(scaled_bboxes)):
        for box in scaled_bboxes[i]:
            predicted_data['image_id'].append(test_image_ids[i])
            predicted_data['x_min'].append(box[0])
            predicted_data['y_min'].append(box[1])
            predicted_data['x_max'].append(box[2])
            predicted_data['y_max'].append(box[3])

    return pd.DataFrame(predicted_data)
