import datetime
import time
import tqdm
from typing import List
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train_loop(model: torch.nn.Module,
               train_loader: DataLoader,
               valid_loader: DataLoader,
               epochs: int,
               optimizer: torch.optim,
               lr_scheduler: torch.optim.lr_scheduler,
               writer: SummaryWriter) -> [List[float], List[float]]:
    """
    Train/Validation loop.

    Args:
        model: object detector.
        train_loader: train data.
        valid_loader: validation data.
        epochs: number of epochs to train the model.
        optimizer: compute gradients.
        lr_scheduler: simple scheduler that decrease lr during training.
        writer: log statistics to Tensorboard.

    Return:
         train_losses: train loss for all epochs.
         valid_losses: valid loss for all epochs.

    """

    train_losses, valid_losses, valid_outputs = [], [], []
    for epoch in range(epochs):
        print("---- Train Epoch %s ----" % (epoch + 1))
        model.train()
        num_batches = len(train_loader)
        running_train_loss = 0.0
        log_interval = num_batches // 5
        train_batches = 0

        for batch_index, (images, annotations) in tqdm.tqdm(enumerate(train_loader),
                                                            desc="Training on train set",
                                                            total=len(train_loader)):
            # acc samples
            train_batches += 1

            # inputs are accordingly formatted by collate fn

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images, annotations)

            loss_class = outputs['class_loss']
            loss_boxes_regr = outputs['box_loss']
            losses = outputs['loss']

            losses.backward()
            optimizer.step()

            # learning rate scheduler step
            if lr_scheduler:
                lr_scheduler.step()

            # print statistics
            running_train_loss += losses.item()

            if log_interval > 0:
                if batch_index % log_interval == 0:
                    print(
                        f'\n[{batch_index + 1:5d} "/" {len(train_loader)}] train loss: {running_train_loss / train_batches}')
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

            for batch_index, (images, annotations) in tqdm.tqdm(enumerate(valid_loader),
                                                                desc="Evaluating on validation set",
                                                                total=len(valid_loader)):
                # acc samples
                valid_batches += 1
                # get inputs: also img_size, img_scale and img_ids are needed for evaluation and metrics recording

                # forward
                output = model(images, annotations)
                losses = output['loss']

                # print statistics
                running_valid_loss += losses.item()

            valid_losses.append(running_valid_loss / valid_batches)
            print(f'Validation loss: {running_valid_loss / valid_batches}')
            writer.add_scalar('Metrics/Loss_running_val', running_valid_loss / valid_batches, epoch + 1)

    return train_losses, valid_losses


def train(model_name: str,
          efficient_det_model: torch.nn.Module,
          epochs: int,
          lr: float,
          train_loader: torch.utils.data.DataLoader,
          valid_loader: torch.utils.data.DataLoader,
          save_path: str
          ) -> None:
    """
    Define parameters for training and perform fine-tuning of the object detector.

    Args:
        model_name: specify the type of EfficientDet.
        efficient_det_model: torch.nn.Module to fine-tune.
        epochs: number of training loops.
        lr: starting learning rate.
        train_loader: training data.
        valid_loader: validation data.
        save_path: location where model will be saved.

    """

    optimizer = optim.AdamW(efficient_det_model.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    scheduler = None
    run_dir = f'runs/{model_name}_{str(datetime.datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").replace(".", "_")}'

    # track experiments
    writer = SummaryWriter(log_dir=run_dir)

    start_time = time.time()
    _, _ = train_loop(efficient_det_model,
                      train_loader,
                      valid_loader,
                      epochs,
                      optimizer,
                      scheduler,
                      writer)
    duration = time.time() - start_time

    print('Execution time: ', duration)

    writer.flush()
    writer.close()

    # save model for inference
    torch.save(efficient_det_model.model.state_dict(), save_path)
