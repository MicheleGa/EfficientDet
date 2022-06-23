from typing import List

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train_loop(model: torch.nn.Module,
               trainloader: DataLoader,
               validloader: DataLoader,
               epochs: int,
               optimizer: torch.optim,
               lr_scheduler: torch.optim.lr_scheduler,
               device: str,
               writer: SummaryWriter) -> [List[float], List[float]]:
    """
    Args:
        model: object detector
        trainloader: train data
        validloader: validation data
        epochs: number of epochs to train the model
        optimizer: compute gradients
        lr_scheduler: simple scheduler that decrease lr during training
        device: cpu/cuda
        writer: log statistics to Tensorboard
    Return:
         train_losses: train loss for all epochs
         valid_losses: valid loss for all epochs
    """

    train_losses, valid_losses = [], []
    for epoch in range(epochs):
        print("---- Train Epoch %s ----" % (epoch + 1))
        model.train()
        size_ds_train = len(trainloader.dataset)
        num_batches = len(trainloader)
        running_train_loss = 0.0
        log_interval = num_batches // 5
        train_batches = 0

        for batch_index, (images, targets) in tqdm.tqdm(enumerate(trainloader), desc="Training on train set",
                                                        total=len(trainloader)):
            # acc samples
            train_batches += 1

            # get inputs (may move this part to collate fn)
            inputs = torch.stack(list(image.to(device) for image in images)).float()
            annotations = {
                'bbox': [target['boxes'].float().to(device) for target in targets],
                'cls': [target['labels'].float().to(device) for target in targets]
            }

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs, annotations)

            loss_class = outputs['class_loss']
            loss_boxes_regr = outputs['box_loss']
            losses = outputs['loss']

            losses.backward()
            optimizer.step()

            # learing rate scheduler step
            if lr_scheduler:
                lr_scheduler.step()

            # print statistics
            running_train_loss += losses.item()

            if log_interval > 0:
                if batch_index % log_interval == 0:
                    print(
                        f'\n[{batch_index + 1:5d} "/" {len(trainloader)}] train loss: {running_train_loss / train_batches}')
                    global_step = batch_index + (epoch * num_batches)
                    writer.add_scalar('Metrics/Loss_Train_IT_Sum', losses, global_step)
                    writer.add_scalar('Metrics/Loss_Train_IT_Boxes', loss_boxes_regr, global_step)
                    writer.add_scalar('Metrics/Loss_Train_IT_Classification', loss_class, global_step)

        train_losses.append(running_train_loss / train_batches)

        print()
        with torch.no_grad():
            print("---- Val Epoch %s ----" % (epoch + 1))
            model.eval()
            running_valid_loss = 0.0
            valid_batches = 0

            for batch_index, (images, targets) in tqdm.tqdm(enumerate(validloader), desc="Evaluating on validation set",
                                                            total=len(validloader)):
                # acc samples
                valid_batches += 1

                # get inputs (may move this part to collate fn)
                inputs = torch.stack(list(image.to(device) for image in images)).float()
                annotations = {
                    'bbox': [target['boxes'].float().to(device) for target in targets],
                    'cls': [target['labels'].float().to(device) for target in targets],
                    'img_size': torch.tensor([target["img_size"] for target in targets]).float().to(device),
                    'img_scale': torch.tensor([target["img_scale"] for target in targets]).float().to(device)
                }

                # forward
                outputs = model(inputs, annotations)
                losses = outputs['loss']

                # print statistics
                running_valid_loss += losses.item()

            valid_losses.append(running_valid_loss / valid_batches)

            print(f'Validation loss: {running_valid_loss / valid_batches}')
            writer.add_scalar("Metrics/Loss_validation", running_valid_loss / train_batches, epoch + 1)

    return train_losses, valid_losses