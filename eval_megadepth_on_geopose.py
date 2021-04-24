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

from megadepth.options.eval_options import EvalOptions
from megadepth.options.train_options import TrainOptions
from megadepth.models.hourglass_model import HourglassModel
from semseg.models.models import ModelBuilder, SegmentationModule
from semseg.utils import colorEncode
import geopose.dataset as dataset

from utils.process_images import get_sky_mask, transform_image_for_megadepth, megadepth_predict, \
    transform_image_for_semseg, semseg_predict, apply_sky_mask
from utils.semseg import visualize_result


def load_models():
    megadepth_checkpoints_path = './megadepth/checkpoints/'
    opt = EvalOptions().parse(megadepth_checkpoints_path)
    model = HourglassModel(opt)
    # input_height = 384
    # input_width = 512
    model.switch_to_eval()


    # todo uncomment for semseg
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
    megadepth_model, semseg_model = load_models()
    megadepth_model.switch_to_eval()
    # todo input size for megadepth
    input_height = 384
    input_width = 512
    ds_dir = './datasets/geoPose3K_final_publish/'
    # dataset.clear_dataset_dir(ds_dir)

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         # transforms.CenterCrop((384, 512)),
                                         # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         #                      std=[0.229, 0.224, 0.225])  # todo fix broadcasting error
                                         ])

    ds = dataset.GeoPoseDataset(ds_dir=ds_dir, transforms=data_transform)

    # lze iterovat stejně jako přes ds, jen pracuje s batches místo samples
    loader = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=4, collate_fn=ds.collate)

    indices = np.random.randint(0, len(ds), 50) # for random photos from dataset
    for i, sample in enumerate(ds[indices]):
        start = time.time()
        input_image_base = sample['img']
        input_image = sample['img'].type(torch.FloatTensor)
        depth_img = sample['depth']
        mask_img = sample['mask']
        dir_path = sample['path']


        img = torch.unsqueeze(input_image, dim=0)

        # prediction for single sample
        pred = megadepth_model.hg_model.forward(img).cpu()
        # megadepth_pred = torch.squeeze(torch.exp(pred), dim=0)
        megadepth_pred = torch.squeeze(pred, dim=0)


        # megadepth_input = transform_image_for_megadepth(input_image, input_height, input_width)
        # megadepth_pred = megadepth_predict(megadepth_model, megadepth_input)
        # megadepth_pred = megadepth_predict(megadepth_model, Variable(input_image.cuda()))
        megadepth_pred_backup = np.copy(megadepth_pred.detach().numpy())

        # todo show megadepth
        # plt.imshow(megadepth_pred)
        # plt.colorbar()
        # plt.show()

        # todo semseg is not working now
        # img_for_semseg, _ = transform_image_for_semseg(input_image, input_height, input_width)
        # semseg_pred = semseg_predict(semseg_model, img_for_semseg)

        # todo get sky mask from GT
        sky_mask = depth_img == -1
        idx = (sky_mask == True)
        megadepth_pred[idx] = -1

        # todo get sky mask from semseg
        # sky_mask = get_sky_mask(megadepth_pred_backup)
        # visualize_result(original_resized, pred)
        # print(sky_mask.shape)
        # applies it in place
        # apply_sky_mask(megadepth_pred.squeeze(), sky_mask)


        # todo show with mask
        # plt.imshow(no_sky_image)
        # plt.colorbar()
        # plt.show()

        # # todo show 4 subplots: original image, GT, depth map, depth map no sky
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
        ax1.imshow(torch.transpose(input_image_base.T, 0, 1))
        ax2.imshow(depth_img.squeeze())
        ax3.imshow(megadepth_pred_backup.squeeze())
        ax4.imshow(megadepth_pred.detach().numpy().squeeze())
        # fig.show()
        # plt.show()
        # todo save some figures
        print(i)
        fig.savefig('./figs/baseline/' + str(i) + '.png', dpi=110)
        if i == 50:
            break

        # todo save predicted depths as .npy
        # print(dir_path)
        # np.save(dir_path + '/depth_map', megadepth_pred_backup)
        # np.save(dir_path + '/depth_map_no_sky', megadepth_pred)
        # end = time.time()
        # took = end - start
        # # print(dir_path)
        # print(f'{i}/{len(ds)}, last one took: {took:.3f}s', sep=' ', end='\r', flush=True)
        # sys.stdout.flush()
        # if i == 10:
        #     break
