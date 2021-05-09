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

from geopose.options.train_options import TrainOptions
from geopose.model.hourglass_model import HourglassModel

# configuration flags
running_in_colab = False
quiet = None

# paths, file names
drive_outputs_path = ''
training_run_id = ''
outputs_dir = ''

# model and evaluation settings
scale_invariance = None
batch_size = None


def plot_training_loss(train_loss_history, show=True, save=True):
    if not show and not save:
        return

    fig, ax = plt.subplots()

    ax.plot(train_loss_history)
    ax.plot(running_mean(train_loss_history, 100, pad_start=True))
    ax.set_xlabel('batch (size = {})'.format(batch_size))
    ax.set_ylabel('RMSE loss')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.suptitle('Training loss \n(scale {}invariant)'.format('' if scale_invariance else 'non-'))
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
    if not show and not save:
        return

    fig, ax = plt.subplots()

    ax.plot(data_history)
    ax.plot(data_si_history)
    ax.plot(grad_history)
    ax.set_xlabel('batch (size = {})'.format(batch_size))
    ax.set_ylabel('Losses')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.suptitle('Validation \n(trained on scale {}invariant)'.format('' if scale_invariance else 'non-'))
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
    quiet = False
    running_on_metacentrum = True

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
        batch_size = 8

    elif running_on_metacentrum:
        dataset_path = '/storage/brno3-cerit/home/xmojzi08/geoPose3K_final_publish'
        outputs_dir = os.path.join('geopose', 'model_outputs', training_run_id)
        batch_size = 8
    else:
        dataset_path = 'datasets/geoPose3K_final_publish'
        batch_size = 2

    os.makedirs(outputs_dir, exist_ok=True)

    writer = SummaryWriter(outputs_dir)

    """ Dataset """
    train_loader, val_loader, test_loader = get_dataset_loaders(dataset_path, batch_size, workers=8, fraction=1,
                                                   validation_split=0.2)

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not quiet:
        print(f'Using device: {device}\n')

        # Additional Info when using cuda
        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

    """ Model """
    weights_path = 'geopose/checkpoints/best_generalization_net_G.pth'
    with patch.object(sys, 'argv', [curr_script_path]):
        # fix for colab interpreter arguments
        opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
    hourglass = HourglassModel(opt, weights_path=weights_path)

    """ Training """
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # better performance
    scaler = torch.cuda.amp.GradScaler()

    lr = opt.lr
    optimizer = torch.optim.Adam(hourglass.model.parameters(), lr=lr, betas=(opt.beta1, 0.999))
    epochs = 10

    epochs_trained = 0
    train_loss_history = []  # combined loss
    val_loss_data_si_history = []
    val_loss_data_history = []
    val_loss_grad_history = []
    scale_invariance = False
    stop_training = False  # break training loop flag
    epoch_mean_loss = -1

    step_total = 0

    print("\t<current_batch>/<batches_total> : b= <total_batch_loss> g= <gradient_loss> t= <time_per_sample>")
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
                    fovs = batch['fov'].cuda()
                    """
                    # fov_embed.fov_id = batch['fov'].cuda()
                    # todo fov: načíst v rámci batche i field of view data a přiřadit je jako atribut custom vrstvě FOV
                    """

                    # make prediction
                    preds = hourglass.model.forward(imgs)

                    preds = preds.squeeze(dim=1)
                    depths = depths.squeeze(dim=1)
                    masks = masks.squeeze(dim=1)

                    # reshape fovs
                    fovs = fovs.reshape(batch_size, 1, 1)
                    # multiply batch by corresponding fov
                    preds = preds * (1/fovs)

                    data_loss = rmse_loss(preds, depths, masks, scale_invariant=scale_invariance)
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
                    print("\t{:>4}/{} : d={:<9.0f} g={:<9.0f} t={:.2f}s/sample "
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
        # if epoch % 20 == 19:
        # plot_training_loss(train_loss_history, show=running_in_colab, save=running_in_colab)
        save_weights(hourglass.model, epoch, epoch_mean_loss, outputs_dir)

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
                fovs = batch['fov'].cuda()

                preds = hourglass.model.forward(imgs)
                preds = preds.squeeze(dim=1)
                depths = depths.squeeze(dim=1)
                masks = masks.squeeze(dim=1)

                # reshape fovs
                fovs = fovs.reshape(batch_size, 1, 1)
                # multiply batch by corresponding fov
                preds = preds * (1 / fovs)

                data_loss = rmse_loss(preds, depths, masks, scale_invariant=False)
                data_si_loss = rmse_loss(preds, depths, masks, scale_invariant=True)
                grad_loss = gradient_loss(preds, depths, masks)
                batch_loss = (data_loss + 0.5 * grad_loss)

                if not quiet:
                    print("\t{:>4}/{} : d={:<9.0f} g={:<9.0f} t={:.2f}s "
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

        # plot_val_losses(val_loss_data_history, val_loss_data_si_history, val_loss_grad_history)

        if stop_training:
            break

    """Test set evaluation"""
    test_loss_history = []
    test_data_loss_history = []
    test_grad_loss_history = []

    hourglass.model.eval()
    with torch.no_grad():
        if not quiet:
            print('test:')
        for i, batch in enumerate(test_loader):
            imgs = batch['img'].to('cuda:0')

            depths = batch['depth'].cuda()
            masks = batch['mask'].cuda()
            paths = batch['path']
            fovs = batch['fov'].cuda()

            preds = hourglass.model.forward(imgs)
            preds = preds.squeeze(dim=1)
            depths = depths.squeeze(dim=1)
            masks = masks.squeeze(dim=1)

            # reshape fovs
            fovs = fovs.reshape(batch_size, 1, 1)
            # multiply batch by corresponding fov
            preds = preds * (1 / fovs)

            data_loss = rmse_loss(preds, depths, masks, scale_invariant=False)
            data_si_loss = rmse_loss(preds, depths, masks, scale_invariant=True)
            grad_loss = gradient_loss(preds, depths, masks)
            batch_loss = (data_loss + 0.5 * grad_loss)

            test_loss_history.append(batch_loss.item())
            test_data_loss_history.append(data_loss.item())
            test_grad_loss_history.append(grad_loss.item())

            if not quiet:
                print("\t{:>4}/{} : d={:<9.0f} g={:<9.0f}"
                      .format(i + 1, len(test_loader), batch_loss.item(), grad_loss.item()))

    mean_test_loss = np.nanmean(test_loss_history)
    mean_test_data_loss = np.nanmean(test_data_loss_history)
    mean_test_grad_loss = np.nanmean(test_grad_loss_history)
    print('Test set results:'
          f'\tTotal loss: {mean_test_loss}'
          f'\tData loss: {mean_test_data_loss}'
          f'\tGradient loss: {mean_test_grad_loss}'
          f'Scale invariance: {scale_invariance}'
          )

    epochs_trained += epochs

    writer.add_hparams({'lr': lr, 'epochs': epochs_trained},
                       {'hparam/loss': epoch_mean_loss})
