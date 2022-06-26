import os
import torch
import pandas as pd
import gc
from torchvision import transforms
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import timm
from effdet.config.model_config import efficientdet_model_param_dict
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from effdet.efficientdet import HeadNet
from dataset_class import GWDataset
from preprocessing import unzip_dataset, pre_process_annotations
from train import train
from performance_measure import get_test_images, post_process
from utils import fix_random, show_images, count_parameters, set_requires_grad_for_layer, build_output_dataframe


def data_preparation(original_img_size,
                     num_workers=1,
                     batch_size=16,
                     train_path='./dataset',
                     name='GlobalWheatDetection') -> [DataLoader, DataLoader]:
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


def model_definition(inference=False,
                     checkpoint_path='',
                     num_classes=2,
                     model_name='efficientdet_d0',
                     pretrained_backbone=False,
                     freeze_backbone=False):
    """
    Define a configuration to choose a detector from EfficientDet family.

    Args:
        inference: false to get model to train.
        checkpoint_path: path to the pre-trained/fine-tuned model.
        num_classes: output classes of the classifier from the detector.
        model_name: specifies the architecture to select
        pretrained_backbone: false to load model checkpoints from models dir
        freeze_backbone: true to avoid changing also weights of the backbone during training
    Return:
        model wrapped in a specific class for training or inference
    """
    # get parameter dictionary and modify configuration for the specific GWDataset
    config = get_efficientdet_config(model_name)
    net = EfficientDet(config, pretrained_backbone=pretrained_backbone)

    # fine-tune, load tf_efficient_det_d0 or tf_efficient_det_d0_ap checkpoints
    if not inference:
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint)

    # reset_head will change config to meet our problem number of classes
    if num_classes is not None and num_classes != config.num_classes:
        net.reset_head(num_classes=num_classes)

    # change model architecture with the new config
    net.class_net = HeadNet(config, num_outputs=config.num_classes)

    if inference:
        # load just trained model
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])

        # free memory
        del checkpoint
        gc.collect()

        return DetBenchPredict(net)

    else:
        if freeze_backbone:
            set_requires_grad_for_layer(net.backbone, False)

        return DetBenchTrain(net, config)


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

    # width/height of all input img is 1024
    input_img_size = 1024

    # resize images to EfficientDet d0 input resolution, i.e. 512*512
    img_size = 512

    # Albumentation library has a Normalize with default mean and std = to ImageNet values,
    # moreover, bbox_params of Compose allows to Normalize coordinates by dividing for input
    # image size (input_img_size) in order to work with small floats instead with big integers
    # in range 0 - 1024
    data_transforms = {'train': A.Compose([A.RandomSizedCrop(min_max_height=(800, 800),
                                                             height=input_img_size,
                                                             width=input_img_size,
                                                             p=0.5),
                                           A.HorizontalFlip(p=0.5),
                                           A.VerticalFlip(p=0.5),
                                           A.Resize(height=img_size,
                                                    width=img_size,
                                                    p=1),
                                           A.Normalize(mean=mean_image_net, std=std_image_net),
                                           ToTensorV2(p=1)],
                                          p=1,
                                          bbox_params=A.BboxParams(format='pascal_voc',
                                                                   min_area=0,
                                                                   min_visibility=0,
                                                                   label_fields=['labels'])),

                       'val': A.Compose([A.Resize(height=img_size,
                                                  width=img_size,
                                                  p=1),
                                         A.Normalize(mean=mean_image_net, std=std_image_net),
                                         ToTensorV2(p=1)],
                                        p=1,
                                        bbox_params=A.BboxParams(format='pascal_voc',
                                                                 min_area=0,
                                                                 min_visibility=0,
                                                                 label_fields=['labels']))
                       }

    # adapted to resource availability :(
    num_workers = 1
    batch_size = 16

    # background already accounted by the EfficientDet model
    # hence the number of classes of the task is 1
    num_classes = 1

    train_loader, valid_loader = data_preparation(input_img_size, num_workers, batch_size)

    # model definition and training
    print('Available efficient det models:')
    print(list(efficientdet_model_param_dict.keys()))

    print('Available efficient net models:')
    print(timm.list_models('*efficientnet*'))

    # get model and print infos
    checkpoint_path = './models'
    efficient_det_model_name = 'tf_efficientdet_d0'
    efficient_det_checkpoint = ''
    print(f'Model: {efficient_det_model_name}')

    if efficient_det_model_name == 'tf_efficientdet_d0':
        efficient_det_checkpoint = 'tf_efficientdet_d0_34-f153e0cf.pth'
    elif efficient_det_model_name == 'tf_efficientdet_d0_ap':
        efficient_det_checkpoint = 'tf_efficientdet_d0_ap-d0cdbd0a.pth'
    else:
        raise Exception('Invalid model ...')

    efficient_det_model = model_definition(checkpoint_path=os.path.join(checkpoint_path,
                                                                        efficient_det_checkpoint),
                                           model_name=efficient_det_model_name,
                                           pretrained_backbone=False)

    print(f'Parameters: {count_parameters(efficient_det_model)}')
    print('EfficientDet configuration:')
    for k, v in efficient_det_model.config.items():
        print(f'- {k}: {str(v)}')

    print(efficient_det_model)
    efficient_det_model = efficient_det_model.to(device)

    # model training
    epochs = 5

    save_path = os.path.join(checkpoint_path, efficient_det_model_name + '_gwd.pth')
    train(model_name=efficient_det_model_name,
          efficient_det_model=efficient_det_model,
          epochs=epochs,
          train_loader=train_loader,
          valid_loader=valid_loader,
          device=device,
          save_path=save_path)

    # performance measure
    # restore model
    efficient_det_model = model_definition(inference=True,
                                           checkpoint_path=save_path,
                                           model_name=efficient_det_model_name,
                                           pretrained_backbone=False)
    efficient_det_model = efficient_det_model.to(device)

    # get test data
    images_tensor, test_images_ids, test_image_sizes = get_test_images(data_transforms=data_transforms)
    num_images = len(test_images_ids)

    # test model
    efficient_det_model.eval()
    with torch.no_grad():
        detections = efficient_det_model(images_tensor.to(device),
                                         img_info={'img_size': torch.tensor(test_image_sizes, device=device).float(),
                                                   'img_scale': torch.ones(num_images, device=device).float()})

    detections = detections.cpu().numpy()
    confidence_threshold = 0.2
    print('Number of test images: ', num_images)
    print('Confidence threshold: ', confidence_threshold)
    # 4 floats for bbox, confidence score, class
    print('Detected bounding box format: ', detections[0][0])

    scaled_bboxes, prediction_scores, prediction_classes = post_process(detections=detections,
                                                                        image_sizes=test_image_sizes,
                                                                        test_image_ids=test_images_ids)
    predicted_data = build_output_dataframe(scaled_bboxes=scaled_bboxes, test_image_ids=test_images_ids)

    # show predicted bboxes on unseen data
    # idx is 0...9
    idx = 0
    show_images(df=predicted_data,
                idx=idx,
                folder=os.path.join('./dataset', 'GlobalWheatDetection', 'test'),
                title='Wheat Head Test',
                linecolor='red')
