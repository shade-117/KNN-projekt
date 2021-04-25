# stdlib
import shutil
from datetime import datetime
from math import floor
from unittest.mock import patch
import os
import sys
import time

# external
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# fix for local import problems - add all local directories
sys_path_extension = [os.getcwd()]  # + [d for d in os.listdir() if os.path.isdir(d)]
sys.path.extend(sys_path_extension)

# local
from geopose.dataset import get_dataset_loaders
from geopose.losses import rmse_loss, gradient_loss
from geopose.util import running_mean

from megadepth.options.train_options import TrainOptions
from megadepth.models.hourglass_model import HourglassModel

# configuration flags
running_in_colab = False
quiet = None

# paths, file names
drive_outputs_path = ''
training_run_id = ''
outputs_dir = ''

# model and evaluation settings
scale_invariancy = None
batch_size = None


def plot_training_loss(train_loss_history, show=True, save=True):
    fig, ax = plt.subplots()

    ax.plot(train_loss_history)
    ax.plot(running_mean(train_loss_history, 100, pad_start=True))
    ax.set_xlabel('batch (size = {})'.format(batch_size))
    ax.set_ylabel('RMSE loss')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.suptitle('Training loss \n(scale {}invariant)'.format('' if scale_invariancy else 'non-'))
    fig.legend(['per-batch', 'running-mean'])

    if save:
        file_name = 'training_loss.png'
        fig_location = os.path.join(outputs_dir, file_name)
        os.makedirs(os.path.dirname(fig_location), exist_ok=True)
        fig.savefig(fig_location)
        if running_in_colab:
            shutil.copy(fig_location, os.path.join(drive_outputs_path, file_name))
    if show:
        fig.show()


def plot_val_losses(data_history, data_si_history, grad_history, show=True, save=True):
    fig, ax = plt.subplots()

    ax.plot(data_history)
    ax.plot(data_si_history)
    ax.plot(grad_history)
    ax.set_xlabel('batch (size = {})'.format(batch_size))
    ax.set_ylabel('Losses')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.suptitle('Validation \n(trained on scale {}invariant)'.format('' if scale_invariancy else 'non-'))
    fig.legend(['RMSE', 'si-RMSE', 'gradient'])
    if save:
        file_name = 'val_loss.png'
        fig_location = os.path.join(outputs_dir, file_name)
        os.makedirs(os.path.dirname(fig_location), exist_ok=True)
        fig.savefig(fig_location)

        if running_in_colab:
            shutil.copy(fig_location, os.path.join(drive_outputs_path, file_name))
    if show:
        fig.show()


def save_weights(model, epoch, epoch_mean_loss, weights_dir):
    weights_file = f'weights_{epoch}_{epoch_mean_loss:.0f}.pth'
    weights_path = os.path.join(weights_dir, weights_file)
    torch.save(model.state_dict(), weights_path)

    if running_in_colab:
        drive_weights_path = os.path.join(drive_outputs_path + weights_file)
        shutil.copy(weights_path, drive_weights_path)
        if not quiet:
            print('saved weights to drive at:', drive_weights_path)


if __name__ == '__main__':
    # globals - technically everything is global but only the following vars are treated so:
    # drive_outputs_path, batch_size, scale_invariancy, quiet, training_run_id, outputs_dir

    """
    todo:

    - augmentace:
        - left-right flip
        - random crop 384x512
        - rotate 10 degrees
        - 
    
    - vyzkoušet overfittnout malý dataset
    - vyzkoušet overfittnout se scale-invariant loss
    
    - načítání FOV z datasetu
    - druhá hlava výstupu sítě - scaling factor
    
    """
    try:
        from IPython import get_ipython

        running_in_colab = 'google.colab' in str(get_ipython())
    except:
        running_in_colab = False

    curr_script_path = os.path.join(os.getcwd(), 'geopose', 'train.py')
    training_run_id = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    print('run ID:', training_run_id)

    if running_in_colab:
        dataset_path = '/content/drive/MyDrive/geoPose3K_final_publish'
        outputs_dir = os.path.join('/content', 'model_outputs', training_run_id)

        drive_outputs_path = '/content/drive/MyDrive/knn_outputs/' + training_run_id
        os.makedirs(drive_outputs_path, exist_ok=True)

    else:
        dataset_path = os.path.join('datasets', 'geoPose3K_final_publish')
        outputs_dir = os.path.join('geopose', 'model_outputs', training_run_id)

    os.makedirs(outputs_dir, exist_ok=True)

    writer = SummaryWriter(outputs_dir)

    # clear_dataset_dir(ds_dir)
    # rotate_images(ds_dir)

    """ Dataset """
    batch_size = 8 if running_in_colab else 2
    train_loader, val_loader = get_dataset_loaders(dataset_path, batch_size, workers=4, fraction=0.01)

    """ Model """
    megadepth_checkpoints_path = './megadepth/checkpoints/'

    with patch.object(sys, 'argv', [curr_script_path]):
        # fix for colab interpreter arguments
        opt = TrainOptions().parse(quiet=True)  # set CUDA_VISIBLE_DEVICES before import torch
    hourglass = HourglassModel(opt)

    """ Training """
    # torch.autograd.set_detect_anomaly(True)  # debugging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # better performance
    scaler = torch.cuda.amp.GradScaler()

    lr = opt.lr * 100
    optimizer = torch.optim.Adam(hourglass.model.parameters(), lr=lr, betas=(opt.beta1, 0.999))
    epochs = 20

    epochs_trained = 0
    train_loss_history = []  # combined loss
    val_loss_data_si_history = []
    val_loss_data_history = []
    val_loss_grad_history = []
    scale_invariancy = False
    stop_training = False  # break training loop flag
    quiet = False
    epoch_mean_loss = -1

    step_total = 0
    for epoch in range(epochs_trained, epochs_trained + epochs):
        epoch_train_loss_history = []
        epoch_train_data_loss_history = []
        epoch_train_grad_loss_history = []
        epoch_val_loss_data_history = []
        epoch_val_loss_data_si_history = []
        epoch_val_loss_grad_history = []
        epoch_val_loss_history = []

        epoch_start = time.time()
        hourglass.model.train()
        print("epoch:", epoch)
        try:
            for i, batch in enumerate(train_loader):
                step_total += 1
                # zero gradient
                for param in hourglass.model.parameters():
                    param.grad = None

                with torch.cuda.amp.autocast():
                    imgs = batch['img'].to('cuda:0')

                    depths = batch['depth'].cuda()
                    masks = batch['mask'].cuda()
                    paths = batch['path']

                    # batch prediction
                    preds = hourglass.model.forward(imgs)
                    preds = torch.squeeze(preds, dim=1)

                    # # pridane pre logaritmovanie
                    # preds = torch.squeeze(torch.exp(preds), dim=0)
                    # preds_t = torch.log(preds + 2)
                    # depths_t = torch.log(depths + 2)
                    # batch_loss = rmse_loss(preds_t, depths_t, masks, scale_invariant=scale_invariancy)

                    data_loss = rmse_loss(preds, depths, masks, scale_invariant=scale_invariancy)
                    grad_loss = gradient_loss(preds, depths, masks)
                    batch_loss = (data_loss + 0.5 * grad_loss)

                scaler.scale(batch_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # logging
                train_loss_history.append(batch_loss.item())
                epoch_train_loss_history.append(batch_loss.item())
                epoch_train_data_loss_history.append(data_loss.item())
                epoch_train_grad_loss_history.append(grad_loss.item())
                writer.add_scalars('step_loss', {'train': batch_loss}, step_total)
                writer.add_scalars('step_train_losses', {'joint': batch_loss,
                                                         'data': data_loss,
                                                         'gradient': grad_loss, }, step_total)

                if not quiet:
                    print("\t{:>4}/{} : d={:<9.2f} g={:<9.2f} t={:.2f}s/sample "
                          .format(i + 1, len(train_loader), batch_loss.item(), grad_loss.item(),
                                  (time.time() - epoch_start) / ((i + 1) * batch_size)))


        except KeyboardInterrupt:
            print('stopped training')
            stop_training = True
            # stops after evaluating on validation set

        epoch_mean_loss = np.mean(epoch_train_loss_history)
        writer.add_scalars('epoch_loss', {'train': epoch_mean_loss}, epoch)
        writer.add_scalars('epoch_train_losses', {'joint': epoch_mean_loss,
                                                  'data': np.mean(epoch_train_data_loss_history),
                                                  'gradient': np.mean(epoch_train_grad_loss_history), }, epoch)

        """Save weights and loss plot"""
        if epoch % 20 == 19:
            save_weights(hourglass.model, epoch, epoch_mean_loss, outputs_dir)
            plot_training_loss(train_loss_history, show=running_in_colab, save=running_in_colab)

        """Validation set evaluation"""
        hourglass.model.eval()
        with torch.no_grad():
            if not quiet:
                print('val:')
            batch_start = time.time()
            for i, batch in enumerate(val_loader):
                imgs = batch['img'].to('cuda:0')

                depths = batch['depth'].cuda()
                masks = batch['mask'].cuda()
                paths = batch['path']

                preds = hourglass.model.forward(imgs)
                preds = torch.squeeze(preds, dim=1)

                data_loss = rmse_loss(preds, depths, masks, scale_invariant=False)
                data_si_loss = rmse_loss(preds, depths, masks, scale_invariant=True)
                grad_loss = gradient_loss(preds, depths, masks)
                batch_loss = (data_loss + 0.5 * grad_loss)

                if not quiet:
                    print("\t{:>4}/{} : d={:<9.2f} g={:<9.2f} t={:.2f}s "
                          .format(i + 1, len(val_loader), batch_loss.item(), grad_loss.item(),
                                  time.time() - batch_start))

                epoch_val_loss_history.append(batch_loss.item())
                val_loss_data_history.append(data_loss.item())
                val_loss_data_si_history.append(data_si_loss.item())
                val_loss_grad_history.append(grad_loss.item())
                epoch_val_loss_data_history.append(data_loss.item())
                epoch_val_loss_data_si_history.append(data_si_loss.item())
                epoch_val_loss_grad_history.append(grad_loss.item())
                batch_start = time.time()

        writer.add_scalars('epoch_loss', {'val': np.mean(epoch_val_loss_history)}, epoch)
        writer.add_scalars('epoch_val_losses', {'joint': np.mean(epoch_val_loss_history),
                                                'data': np.mean(epoch_val_loss_data_history),
                                                'data_si': np.mean(epoch_val_loss_data_si_history),
                                                'gradient': np.mean(epoch_val_loss_grad_history), }, epoch)

        plot_val_losses(val_loss_data_history, val_loss_data_si_history, val_loss_grad_history)

        if stop_training:
            break

    epochs_trained += floor(len(train_loss_history) / len(train_loader.dataset))

    writer.add_hparams({'lr': lr, 'epochs': epochs_trained},
                       {'hparam/loss': epoch_mean_loss})
