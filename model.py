import gc
import torch
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict, DetBenchTrain
from utils import set_requires_grad_for_layer


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
        model wrapped in a specific class for training or inference.
    """
    # get parameter dictionary and modify configuration for the specific GWDataset
    config = get_efficientdet_config(model_name)
    net = EfficientDet(config, pretrained_backbone=pretrained_backbone)

    # fine-tune, load tf_efficient_det_d0 or tf_efficient_det_d0_ap checkpoints
    if not inference:
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
