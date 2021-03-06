import os
import torch
import timm
from effdet.config.model_config import efficientdet_model_param_dict
from dataset_class import get_data_transforms
from model import model_definition
from preprocessing import data_preparation
from train import train
from utils import fix_random, count_parameters
from test import test_model

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

    # width/height of all input img is 1024
    input_img_size = 1024

    # resize images to EfficientDet d0 input resolution, i.e. 512*512
    img_size = 512

    data_transforms = get_data_transforms(input_img_size, img_size)

    # adapted to resource availability :(
    num_workers = 0
    batch_size = 16

    # background already accounted by the EfficientDet model
    # hence the number of classes of the task is 1
    num_classes = 1

    train_loader, valid_loader, annotations = data_preparation(data_transforms,
                                                               input_img_size,
                                                               device,
                                                               num_workers,
                                                               batch_size)

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
    elif efficient_det_model_name == 'tf_efficientnetv2_b0':
        efficient_det_checkpoint = ''
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
    lr = 0.0002

    save_path = os.path.join(checkpoint_path, efficient_det_model_name + '_gwd.pth')
    train(model_name=efficient_det_model_name,
          efficient_det_model=efficient_det_model,
          epochs=epochs,
          lr=lr,
          train_loader=train_loader,
          valid_loader=valid_loader,
          save_path=save_path)

    # restore model and test it on unseen and train data
    # notice that the model is a DetBenchPredict
    test_model(save_path=save_path,
               annotations=annotations,
               data_transforms=data_transforms,
               efficient_det_model_name=efficient_det_model_name,
               device=device)
