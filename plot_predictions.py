import time
import os

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import torch
import torchvision.transforms
from torchvision import transforms

from geopose.losses import rmse_loss, gradient_loss
import geopose.dataset as dataset

from geopose.model.builder import Hourglass


def load_semseg():
    """ uncomment for semseg - not used """
    # Network Builders

    # input_height = 384
    # input_width = 512

    # download weights: http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth
    # download http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth
    # put them into semseg/checkpoints/ade20k-resnet50dilated-ppm_deepsup
    # semseg_checkpoints_path = './semseg/checkpoints/ade20k-resnet50dilated-ppm_deepsup'
    #
    # net_encoder = ModelBuilder.build_encoder(
    #     arch='resnet50dilated',
    #     fc_dim=2048,
    #     weights=semseg_checkpoints_path + '/encoder_epoch_20.pth')
    # net_decoder = ModelBuilder.build_decoder(
    #     arch='ppm_deepsup',
    #     fc_dim=2048,
    #     num_class=150,
    #     weights=semseg_checkpoints_path + '/decoder_epoch_20.pth',
    #     use_softmax=True)
    #
    # crit = torch.nn.NLLLoss(ignore_index=-1)
    # segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    # segmentation_module.eval()
    # segmentation_module.cuda()
    # return model, segmentation_module
    return None


if __name__ == '__main__':
    # default - MegaDepth pretrained model
    # weights_path = 'geopose/checkpoints/best_generalization_net_G.pth'  # ugly, dp
    weights_path = 'geopose/checkpoints/weights_40_3549-base-plat.pth'
    # weights_path = 'geopose/checkpoints/weights_52_3871_scratch_normie.pth'  # Petr, arch='nice'
    # weights_path = 'geopose/checkpoints/weights_99_3377_scratch_nice.pth'  # Petr, arch='nice'

    megadepth_model = Hourglass(arch='nice', weights=weights_path)

    megadepth_model.model.eval()

    # semseg_model = load_semseg()

    """ Input sizes """
    input_height = 384
    input_width = 512
    # ds_dir = '/storage/brno3-cerit/home/xmojzi08/geoPose3K_final_publish'
    ds_dir = 'datasets/geoPose3K_final_publish/'
    # dataset.clear_dataset_dir(ds_dir)

    data_transform = transforms.Compose([transforms.ToTensor(), ])

    ds = dataset.GeoPoseDataset(ds_dir=ds_dir, transforms=data_transform)

    np.random.seed(1234)
    indices = np.random.randint(0, len(ds), 5)  # for random photos from dataset
    with torch.no_grad():
        for i, sample in enumerate(ds[indices]):
            start = time.time()
            input_image = sample['img'].cuda()
            mask_img = sample['mask']
            depth_img = sample['depth']
            dir_path = sample['path']
            fov = torch.zeros((1,)) + sample['fov']

            img = torch.unsqueeze(input_image, dim=0)
            # prediction for single sample
            pred = megadepth_model.model.forward(img, fov)
            pred = pred.detach().cpu()[0]
            """ Exp the output - was used by megadepth """
            # pred = torch.exp(pred)

            #pred = pred * 1 / fov

            data_loss = rmse_loss(pred, depth_img, mask_img, scale_invariant=False)
            data_si_loss = rmse_loss(pred, depth_img, mask_img, scale_invariant=True)
            grad_loss = gradient_loss(pred, depth_img, mask_img)
            print(f'Data loss: {data_loss.item()}\n'
                  f'Data si-loss: {data_si_loss.item()}\n'
                  f'Grad loss: {grad_loss.item()}\n'
                  f'{i}: {dir_path}')

            depth_img = depth_img[0]

            """ Segment image using semseg - not used """
            # img_for_semseg, _ = transform_image_for_semseg(input_image, input_height, input_width)
            # semseg_pred = semseg_predict(semseg_model, img_for_semseg)

            """ Get sky mask from ground truth """
            sky_mask = depth_img == -1
            pred_masked = np.copy(pred)
            pred_masked[0, sky_mask] = -1

            """ Get diff GT - pred """
            diff = pred - depth_img.numpy()

            """ Get sky mask from semseg - not used """
            # sky_mask = get_sky_mask(megadepth_pred_backup)
            # visualize_result(original_resized, pred)
            # print(sky_mask.shape)
            # applies it in place
            # apply_sky_mask(megadepth_pred.squeeze(), sky_mask)

            """ 
            Plot prediction 
                a) scale-invariant
                    (use for generalization weights)
                b) non-si
                    (use for trained model evaluation)
                    Note: this makes the colorbar range apply to the GT image only
            """
            plot_pred_si = True

            if plot_pred_si:
                value_range_args = {}
            else:
                # get value ranges for accurate color mapping
                pred_low, pred_high = pred.min(), pred.max()
                gt_low, gt_high = depth_img.min(), depth_img.max()
                vmin, vmax = min(gt_low, pred_low), max(gt_high, pred_high)
                value_range_args = {'vmin': vmin, 'vmax': vmax}

            """ show 4 subplots: original image, GT, prediction/GT difference, prediction """
            fig = plt.figure()  # constrained_layout=True
            widths = [1, 1]
            gs = fig.add_gridspec(2, 2, width_ratios=widths)
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.set_title('Input Image')
            ax1.imshow(sample['img'].permute(1, 2, 0).numpy())  # CHW to WHC

            ax2 = fig.add_subplot(gs[0, 1])
            colorbar_ax = ax2.imshow(depth_img, **value_range_args)
            ax2.set_title('Depth GT')

            ax3 = fig.add_subplot(gs[1, 0])
            ax3.imshow(diff[0], cmap=plt.get_cmap('RdBu'))
            ax3.set_title('Prediction/GT Difference')
            # GT subtracted from Prediction
            # => red == predicted more, blue == predicted less

            ax4 = fig.add_subplot(gs[1, 1])
            ax4.imshow(pred[0], **value_range_args)
            ax4.set_title('Depth Prediction')

            # ax5 = fig.add_subplot(gs[:, 2])
            # fig.colorbar(colorbar_ax, cax=ax5, orientation='vertical')
            fig.tight_layout(pad=0.5)

            for ax in fig.axes:
                # axis labels kept only for last axis (colormap)
                ax.axis('off')

            diff[0, sky_mask] = 0  # don't calculate mean and max diff from sky
            # ^ also, don't move this before plotting the diff

            # maximum absolute difference (sign is kept)
            diff_abs_max = diff.min() if np.abs(diff.min()) > np.abs(diff.max()) else diff.max()

            fig.text(0.038, 0.015,
                     f'abs-mean: {np.abs(diff).mean():0.2g}, max: {diff_abs_max:0.2g}',
                     color='black')

            fig.tight_layout(pad=0.7)

            plt.show()
