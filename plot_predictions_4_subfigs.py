import sys
import time
import csv
import os

import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from skimage import io
from skimage.transform import resize
import torch
import torchvision.transforms
from torch.autograd import Variable
from torchvision import transforms

from geopose.options.eval_options import EvalOptions
from geopose.model.hourglass_model import HourglassModel
# from semseg.models.models import ModelBuilder, SegmentationModule # for semseg - not used
# from semseg.utils import colorEncode # for semseg - not used
from geopose.losses import rmse_loss, gradient_loss
import geopose.dataset as dataset

from utils.process_images import get_sky_mask, transform_image_for_megadepth, megadepth_predict, \
    transform_image_for_semseg, semseg_predict, apply_sky_mask
from utils.semseg import visualize_result


def load_models(weights_path):
    opt = EvalOptions().parse()
    # weights_path = 'megadepth/checkpoints/test_local/best_generalization_net_G.pth'
    model = HourglassModel(opt, weights_path=weights_path)

    # input_height = 384
    # input_width = 512
    model.switch_to_eval()

    """ uncomment for semseg - not used """
    # Network Builders
    # todo download weights: http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth
    # todo  and http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth
    # todo and put them into semseg/checkpoints/ade20k-resnet50dilated-ppm_deepsup
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
    return model, None


if __name__ == '__main__':
    # weights_path = 'geopose/checkpoints/saved_9_1207.7374_net_G.pth'
    weights_path = 'geopose/checkpoints/weights_29_5214.pth'
    megadepth_model, semseg_model = load_models(weights_path)
    megadepth_model.switch_to_eval()

    """ Input sizes """
    input_height = 384
    input_width = 512
    # ds_dir = '/storage/brno3-cerit/home/xmojzi08/geoPose3K_final_publish'
    ds_dir = 'datasets/geoPose3K_final_publish/'
    # dataset.clear_dataset_dir(ds_dir)

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         # transforms.CenterCrop((384, 512)),
                                         # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         #                      std=[0.229, 0.224, 0.225])  # todo fix broadcasting error
                                         ])

    ds = dataset.GeoPoseDataset(ds_dir=ds_dir, transforms=data_transform)

    # lze iterovat stejně jako přes ds, jen pracuje s batches místo samples
    # loader = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=4, collate_fn=ds.collate)
    # train_loader, val_loader = dataset.get_dataset_loaders(ds_dir)
    np.random.seed(1234)
    indices = np.random.randint(0, len(ds), 50)  # for random photos from dataset
    with torch.no_grad():
        for i, sample in enumerate(ds[indices]):
            start = time.time()
            input_image = sample['img'].cuda()
            mask_img = sample['mask']
            depth_img = sample['depth']
            dir_path = sample['path']
            fov = float(sample['fov'])

            img = torch.unsqueeze(input_image, dim=0)
            # prediction for single sample
            pred = megadepth_model.model.forward(img).detach().cpu()[0]
            """ Exp the output - was used by megadepth """
            # pred = torch.exp(pred)

            # pred = pred * 1/fov

            depth = depth_img[None, ...]
            mask = mask_img[None, ...]

            data_loss = rmse_loss(pred, depth, mask, scale_invariant=False)
            data_si_loss = rmse_loss(pred, depth, mask, scale_invariant=True)
            grad_loss = gradient_loss(pred, depth, mask)
            print(f'Data loss: {data_loss.item()},\nData si-loss: {data_si_loss.item()},\nGrad loss: {grad_loss.item()}')
            print(f'{i}: {dir_path}')

            megadepth_pred = np.copy(pred)
            megadepth_pred_raw = megadepth_pred.copy()

            """ Segment image using semseg - not used """
            # img_for_semseg, _ = transform_image_for_semseg(input_image, input_height, input_width)
            # semseg_pred = semseg_predict(semseg_model, img_for_semseg)

            """ Get sky mask from ground truth """
            sky_mask = depth_img == -1
            idx = (sky_mask == True)
            megadepth_pred[idx] = -1

            """ Get sky mask from semseg - not used """
            # sky_mask = get_sky_mask(megadepth_pred_backup)
            # visualize_result(original_resized, pred)
            # print(sky_mask.shape)
            # applies it in place
            # apply_sky_mask(megadepth_pred.squeeze(), sky_mask)

            """ show 4 subplots: original image, GT, depth map, depth map no sky """
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
            ax1.imshow(sample['img'].permute(1, 2, 0).numpy())
            ax2.imshow(depth_img[0])
            ax3.imshow(megadepth_pred_raw.squeeze())
            ax4.imshow(megadepth_pred.squeeze())
            # fig.show()
            plt.show()

            """ save the plots """
            # figure_location = f'./figs/baseline/{i}.png'
            # os.makedirs(os.path.dirname(figure_location), exist_ok=True)
            # fig.savefig(figure_location, dpi=110)
            # if i == 50:
            #     break

            """ Save predictions as numpy array """
            # print(dir_path)
            # np.save(dir_path + '/depth_map', megadepth_pred_raw)
            # # megadepth_pred is with masked sky
            # np.save(dir_path + '/depth_map_no_sky', megadepth_pred)
            # end = time.time()
            # took = end - start
            # # print(dir_path)
            # print(f'{i}/{len(ds)}, last one took: {took:.3f}s', sep=' ', end='\r', flush=True)
            # sys.stdout.flush()
            # if i == 10:
            #     break