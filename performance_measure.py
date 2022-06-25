import os
from typing import List
import numpy as np
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

    # we will predict boxes in images that are 512*512, so rescale to 1024*1024,
    # i.e. original resolution, must be done
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
                 iou_threshold=0.44,
                 skip_box_thr=0.43,
                 img_size=512) -> [List, List, List]:
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
        # get predicted bbox coordinates for a image
        boxes = detections[i][:, :4]
        # get scores for a image bboxes
        scores = detections[i][:, 4]
        # get classes for a image bboxes
        classes = detections[i][:, 5]

        # np.where results are wrapped in a numpy array
        indexes = np.where(scores > confidence_threshold)[0]
        boxes = boxes[indexes]
        scores = scores[indexes]
        classes = classes[indexes]

        boxes = [(boxes / img_size).tolist()]
        scores = [scores.tolist()]
        classes = [classes.tolist()]

        boxes, scores, classes = weighted_boxes_fusion(
            boxes,
            scores,
            classes,
            iou_thr=iou_threshold,
            skip_box_thr=skip_box_thr
        )
        boxes = boxes * (img_size - 1)

        prediction_boxes.append(boxes.tolist())
        prediction_scores.append(scores.tolist())
        prediction_classes.append(classes.tolist())

    # rescale boxes in order to be put on a (1024, 1024) image
    scaled_bboxes = []
    for bboxes, img_dims in zip(prediction_boxes, image_sizes):
        im_h, im_w = img_dims  # (1024, 1024)

        if len(bboxes) > 0:
            scaled_bboxes.append(
                (np.array(bboxes) * [im_w / img_size, im_h / img_size, im_w / img_size, im_h / img_size]).tolist())
        else:
            scaled_bboxes.append(bboxes)

    return scaled_bboxes, prediction_scores, prediction_classes
