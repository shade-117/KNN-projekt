from megadepth.models.models import create_model
import scipy.io
import numpy as np
from skimage import io
import csv
import os

from megadepth.options.eval_options import EvalOptions
from megadepth.options.train_options import TrainOptions
from semseg.models.models import ModelBuilder, SegmentationModule
from semseg.utils import colorEncode
import torch
import geopose.dataset as dataset
import matplotlib.pyplot as plt
from skimage.transform import resize
import torchvision.transforms
from torch.autograd import Variable

from utils.process_images import get_sky_mask, transform_image_for_megadepth, megadepth_predict, \
    transform_image_for_semseg, semseg_predict, apply_sky_mask
from utils.semseg import visualize_result


def load_models():
    megadepth_checkpoints_path = './megadepth/checkpoints/'
    opt = EvalOptions().parse(megadepth_checkpoints_path)
    model = create_model(opt)
    # input_height = 384
    # input_width = 512
    model.switch_to_eval()

    # Network Builders
    # todo download weights: http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth
    # todo  and http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth
    # todo and put them into semseg/checkpoints/ade20k-resnet50dilated-ppm_deepsup
    semseg_checkpoints_path = './semseg/checkpoints/ade20k-resnet50dilated-ppm_deepsup'

    net_encoder = ModelBuilder.build_encoder(
        arch='resnet50dilated',
        fc_dim=2048,
        weights=semseg_checkpoints_path + '/encoder_epoch_20.pth')
    net_decoder = ModelBuilder.build_decoder(
        arch='ppm_deepsup',
        fc_dim=2048,
        num_class=150,
        weights=semseg_checkpoints_path + '/decoder_epoch_20.pth',
        use_softmax=True)

    crit = torch.nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module.eval()
    segmentation_module.cuda()
    return model, segmentation_module


if __name__ == '__main__':
    megadepth_model, semseg_model = load_models()
    megadepth_model.switch_to_eval()
    # todo input size for megadepth
    input_height = 384
    input_width = 512
    ds_dir = './datasets/geoPose3K_final_publish_500mb_part/'
    dataset.clear_dataset_dir(ds_dir)
    ds = dataset.GeoPoseDataset(data_dir=ds_dir)

    for i, sample in enumerate(ds):
        # sample = next(iter(ds))

        # f, ax = plt.subplots(1, 3)
        # ax[0].imshow(sample['img'])
        # ax[1].imshow(sample['depth1'])
        # ax[2].imshow(sample['depth2'])
        # f.show()

        input_image = sample['img']
        megadepth_input = transform_image_for_megadepth(input_image, input_height, input_width)
        megadepth_pred = megadepth_predict(megadepth_model, megadepth_input)
        megadepth_pred_backup = np.copy(megadepth_pred)

        # todo show megadepth
        # plt.imshow(megadepth_pred)
        # plt.colorbar()
        # plt.show()

        img_for_semseg, _ = transform_image_for_semseg(input_image, input_height, input_width)
        semseg_pred = semseg_predict(semseg_model, img_for_semseg)

        sky_mask = get_sky_mask(semseg_pred)
        # visualize_result(original_resized, pred)
        # print(sky_mask.shape)
        # applies it in place
        apply_sky_mask(megadepth_pred, sky_mask)

        # todo show with mask
        # plt.imshow(no_sky_image)
        # plt.colorbar()
        # plt.show()
        # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
        # ax1.imshow(sample['img'])
        # ax2.imshow(sample['depth1'])
        # ax3.imshow(megadepth_pred_backup)
        # ax4.imshow(megadepth_pred)
        # fig.show()
        # fig.savefig('./figs/' + str(i) + '.png', dpi=110)
        # if i == 50:
        #     break

        dir_path = os.path.dirname(os.path.realpath(sample['path']))
        # print(dir_path)
        np.save(dir_path + '/depth_map', megadepth_pred_backup)
        np.save(dir_path + '/depth_map_no_sky', megadepth_pred)
        print(f'{i} / {len(ds)}')
        break

