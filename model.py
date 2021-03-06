import gc
import torch
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict, DetBenchTrain
from effdet.config.model_config import efficientdet_model_param_dict
from utils import set_requires_grad_for_layer


def model_definition(inference=False,
                     checkpoint_path='',
                     num_classes=1,
                     model_name='efficientdet_d0',
                     v2_backbone=True,
                     pretrained_backbone=False,
                     freeze_backbone=False):
    """
    Define a configuration to choose a detector from EfficientDet family.

    Args:
        inference: false to get model to train.
        checkpoint_path: path to the pre-trained/fine-tuned model.
        num_classes: output classes of the classifier from the detector.
        model_name: specifies the architecture to select.
        v2_backbone: wheter to use or not new version of efficientnet's backbone.
        pretrained_backbone: false to load model checkpoints from models dir.
        freeze_backbone: true to avoid changing also weights of the backbone during training.
    Return:
        model wrapped in a specific class for training or inference.
    """
    if v2_backbone:
        # register backbone to a dictionary to use it in get_efficientdet_config
        # v2 backbone's are present in timm package
        efficientdet_model_param_dict[model_name] = dict(
            name=model_name,
            backbone_name=model_name,
            backbone_args=dict(drop_path_rate=0.2),
            num_classes=num_classes,
            image_size=(512, 512),
            url='')


    # get parameter dictionary and modify configuration for the specific GWDataset
    config = get_efficientdet_config(model_name)
    net = EfficientDet(config, pretrained_backbone=pretrained_backbone)

    # fine-tune, load tf_efficient_det_d0 or tf_efficient_det_d0_ap checkpoints
    if not inference and not v2_backbone:
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint)

    # reset_head will change config to meet our problem number of classes
    if num_classes != config.num_classes:
        net.reset_head(num_classes=num_classes)

    if inference:
        # load just trained model
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint)

        # free memory
        del checkpoint
        gc.collect()

        return DetBenchPredict(net)

    else:
        if freeze_backbone:
            set_requires_grad_for_layer(net.backbone, False)

        return DetBenchTrain(net, config)
