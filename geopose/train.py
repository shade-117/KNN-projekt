# stdlib
import shutil
from math import floor
from unittest.mock import patch
import os
import sys
import time

# external
import matplotlib.pyplot as plt
import numpy as np
import torch

# fix for local import problems - add all local directories
sys_path_extension = [os.getcwd()]  # + [d for d in os.listdir() if os.path.isdir(d)]
sys.path.extend(sys_path_extension)


# local

from geopose.dataset import get_dataset_loaders
from geopose.losses import rmse_loss, gradient_loss
from geopose.util import running_mean

from megadepth.models.models import create_model
from megadepth.options.train_options import TrainOptions

if __name__ == '__main__':

    """
    todo:

    augmentace

    """
    try:
        from IPython import get_ipython
        running_in_colab = 'google.colab' in str(get_ipython())
    except:
        running_in_colab = False

    if running_in_colab:
        saved_weights_path = '/content'
        dataset_path = '/content/drive/MyDrive/geoPose3K_final_publish'
        drive_saved_weights_path = '/content/drive/MyDrive/saved_models/'
        os.makedirs(drive_saved_weights_path, exist_ok=True)
    else:
        saved_weights_path = os.path.join('geopose', 'saved_models')
        os.makedirs(saved_weights_path, exist_ok=True)
        dataset_path = os.path.join('datasets', 'geoPose3K_final_publish')
        drive_saved_weights_path = None

    # clear_dataset_dir(ds_dir)
    # rotate_images(ds_dir)

    """ Dataset """
    batch_size = 4 if running_in_colab else 2
    train_loader, val_loader = get_dataset_loaders(dataset_path, batch_size, workers=4)

    """ Model """
    megadepth_checkpoints_path = './megadepth/checkpoints/'

    curr_script_path = os.path.join(os.getcwd(), 'geopose', 'train.py')
    with patch.object(sys, 'argv', [curr_script_path]):
        # fix for colab interpreter arguments
        opt = TrainOptions().parse(quiet=True)  # set CUDA_VISIBLE_DEVICES before import torch
    model = create_model(opt)

    """ Training """
    # torch.autograd.set_detect_anomaly(True)  # debugging
    torch.backends.cudnn.benchmark = True  # better performance
    scaler = torch.cuda.amp.GradScaler()

    optimizer = torch.optim.Adam(model.netG.parameters(), lr=opt.lr * 100, betas=(opt.beta1, 0.999))
    epochs = 20

    epochs_trained = 0
    i = 0
    running_loss = 0.0
    train_loss_history = []
    val_loss_history = []
    scale_invariancy = False
    stop_training = False  # break training loop flag
    quiet = False

    for epoch in range(epochs_trained, epochs_trained + epochs):

        epoch_start = time.time()
        model.netG.train()
        print("epoch:", epoch)
        try:
            batch_start = time.time()
            for i, batch in enumerate(train_loader):

                # zero gradient
                for param in model.netG.parameters():
                    param.grad = None

                with torch.cuda.amp.autocast():
                    imgs = batch['img'].type(torch.FloatTensor).permute(0, 3, 1, 2)  # from NHWC to NCHW
                    # todo imgs transformations could be a part of transforms

                    depths = batch['depth'].cuda()
                    masks = batch['mask'].cuda()
                    paths = batch['path']

                    # batch prediction
                    preds = model.netG.forward(imgs)
                    preds = torch.squeeze(preds, dim=1)

                    # # pridane pre logaritmovanie
                    # preds = torch.squeeze(torch.exp(preds), dim=0)
                    # preds_t = torch.log(preds + 2)
                    # depths_t = torch.log(depths + 2)
                    # batch_loss = rmse_loss(preds_t, depths_t, masks, scale_invariant=scale_invariancy)

                    data_loss = rmse_loss(preds, depths, masks, scale_invariant=scale_invariancy)
                    grad_loss = gradient_loss(preds, depths, masks)
                    batch_loss = (data_loss + 0.5 * grad_loss) / batch_size

                train_loss_history.append(batch_loss.item())

                scaler.scale(batch_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if not quiet:
                    print("\t{:>4}/{} : d={:<9.2f} g={:<9.2f} t={:.2f}s/sample "
                          .format(i + 1, len(train_loader), batch_loss.item(), grad_loss.item(),
                                  (time.time() - epoch_start) / ((i + 1) * batch_size)))
                batch_start = time.time()  # next batch timer

        except KeyboardInterrupt:
            print('stopped training')
            stop_training = True
            # stops after evaluating on validation set

        epoch_mean_loss = np.mean(train_loss_history)
        save_path = f'saved_{epoch}_{epoch_mean_loss:.4f}_net_G.pth'
        # save_path = os.path.join(saved_weights_dir, save_path)
        torch.save(model.netG.state_dict(), save_path)

        if running_in_colab:
            shutil.copy(save_path, drive_saved_weights_path + save_path)
            if not quiet:
                print('saved weights to drive at:', drive_saved_weights_path + save_path)

        model.netG.eval()
        with torch.no_grad():
            if not quiet:
                print('val:')
            batch_start = time.time()
            for i, batch in enumerate(val_loader):
                imgs = batch['img'].type(torch.FloatTensor).permute(0, 3, 1, 2)  # from NHWC to NCHW

                depths = batch['depth'].cuda()
                masks = batch['mask'].cuda()
                paths = batch['path']

                preds = model.netG.forward(imgs)
                preds = torch.squeeze(preds, dim=1)

                data_loss = rmse_loss(preds, depths, masks, scale_invariant=scale_invariancy)
                grad_loss = gradient_loss(preds, depths, masks)
                batch_loss = (data_loss + 0.5 * grad_loss) / batch_size

                if not quiet:
                    print("\t{:>4}/{} : d={:<9.2f} g={:<9.2f} t={:.2f}s "
                          .format(i + 1, len(val_loader), batch_loss.item(), grad_loss.item(), time.time() - batch_start))

                val_loss_history.append(batch_loss.item())
                batch_start = time.time()

        if stop_training:
            break

    epochs_trained += floor(len(train_loss_history) / len(train_loader.dataset))

    # Results

    plt.plot(train_loss_history)
    plt.plot(running_mean(train_loss_history, 100, pad_start=True))
    plt.title('Training loss \n(scale {}invariant)'.format('' if scale_invariancy else 'non-'))
    plt.xlabel('batch (size = {}, dataset_batches = {})'.format(batch_size, len(train_loader)))
    plt.ylabel('RMSE loss')
    plt.legend(['train', 'train-mean'])
    plt.show()

    plt.plot(val_loss_history)
    plt.title('Validation loss \n(scale {}invariant)'.format('' if scale_invariancy else 'non-'))
    plt.show()

    # todo uncomment in colab for model saving
    # save_path = '/content/saved_net_G.pth'
    # torch.save(model.netG.state_dict(), save_path)
