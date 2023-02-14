# torch imports
import torch
import torch.nn as nn

# native imports
import copy
import os
import pandas as pd
import numpy as np
import gc


# local imports
from models.device import get_device
from misc import Timer, BatchCounter


def train_wrapper(
    model: nn.Module,
    loaders: dict,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    path_out: str,
) -> None:

    model.float().to(device)
    # model.check_param()

    # metrics
    val_acc_history = []
    train_acc_history = []
    best_model_wts = None
    best_loss = 10e10

    # epochs
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # loader
            dataloader = loaders[phase]
            counter = BatchCounter(num_batches=len(dataloader))
            running_loss = 0

            RATE = 4
            optimizer.zero_grad()
            for i, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.float().to(device)
                labels = labels.float().to(device)

                # gradient decent
                with torch.set_grad_enabled(phase == "train"):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        if (i + 1) % RATE == 0:
                            optimizer.step()
                            optimizer.zero_grad()

                # print which batch is being processed
                loss = loss.item()
                counter.report(loss=loss)
                running_loss += loss * inputs.size(0)

            epoch_loss = running_loss / len(dataloader.dataset)
            print("  | %s loss: %.3f" % (phase, epoch_loss))

            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_loss)
            elif phase == "train":
                train_acc_history.append(epoch_loss)

    history = dict({"train": train_acc_history, "val": val_acc_history})
    plot_curve(history, name=os.path.join(path_out, "loss_%.3f.png" % best_loss))
    print("-" * 10)
    print(" Best val loss: %.3f" % (best_loss))
    print("-" * 10)

    model.load_state_dict(best_model_wts)
    return model


def test_wrapper(
    model: nn.Module,
    loaders: dict,
    criterion: nn.Module,
    device: torch.device,
    path_out: str,
) -> None:

    model.float().to(device)
    model.eval()
    dataloader = loaders["test"]
    counter = BatchCounter(num_batches=len(dataloader))
    pred = []
    running_loss = 0
    for inputs, labels in dataloader:
        inputs = inputs.float().to(device)
        labels = labels.float().to(device)

        # forward only
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.requires_grad = False
            # Get model outputs and calculate loss
            outputs = model(inputs)
            (inputs, labels) = lazy(inputs, labels, batch=0)
        loss = criterion(outputs, labels)
        pred.append(outputs.cpu().detach().numpy())
        # GC
        del outputs
        torch.cuda.empty_cache()

        # print which batch is being processed
        loss = loss.item()
        counter.report(loss=loss)
        running_loss += loss * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    print("  | test loss: %.3f" % (epoch_loss))
    print("-" * 10)
    print()

    # save prediction
    pred_array = pred[0]  # list of list, concatenate it to a single list
    for p in pred[1:]:
        pred_array = np.concatenate([pred_array, p])
    df_pred = dataloader.dataset.img_labels.copy()

    # create new columns
    n_col = np.array(pred_array).shape[1]
    for i in range(n_col):
        df_pred.loc[:, "pred_%d" % i] = 0
        df_pred["pred_%d" % i] = pred_array[:, i]
    # suffix with error
    df_pred.to_csv(os.path.join(path_out, "pred_%.3f.csv" % epoch_loss), index=False)
    torch.save(
        model.state_dict(), os.path.join(path_out, "model_%.3f.pt" % (epoch_loss))
    )

    # return loss
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return epoch_loss


def plot_curve(history: dict, name: str = "loss.png") -> None:
    import matplotlib.pyplot as plt

    # set boundary
    plt.ylim(0, 50)
    plt.plot(history["train"], label="train")
    plt.plot(history["val"], label="val")
    plt.legend()
    plt.savefig(name)
    plt.close()
