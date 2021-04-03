from megadepth.models.models import create_model
import scipy.io
import numpy as np
from skimage import io
import csv
from megadepth.options.train_options import TrainOptions
from semseg.models.models import ModelBuilder, SegmentationModule
from semseg.utils import colorEncode
import torch
import geopose.dataset as dataset
import matplotlib.pyplot as plt
from skimage.transform import resize
import torchvision.transforms
from torch.autograd import Variable

from utils.process_images import get_sky_mask
from utils.semseg import visualize_result


def load_models():
    opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
    model = create_model(opt)
    # input_height = 384
    # input_width = 512
    # model.switch_to_eval()

    # Network Builders
    # todo download weights: http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth
    # todo  and http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth
    # todo and put them into semseg/checkpoints/ade20k-resnet50dilated-ppm_deepsup
    path = './semseg/checkpoints/ade20k-resnet50dilated-ppm_deepsup'

    net_encoder = ModelBuilder.build_encoder(
        arch='resnet50dilated',
        fc_dim=2048,
        weights=path + '/encoder_epoch_20.pth')
    net_decoder = ModelBuilder.build_decoder(
        arch='ppm_deepsup',
        fc_dim=2048,
        num_class=150,
        weights=path + '/decoder_epoch_20.pth',
        use_softmax=True)

    crit = torch.nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module.eval()
    segmentation_module.cuda()
    return model, segmentation_module


if __name__ == '__main__':
    megadepth_model, semseg_model = load_models()
    megadepth_model.switch_to_eval()

    ds_dir = './datasets/geoPose3K_final_publish/'
    dataset.clear_dataset_dir(ds_dir)
    ds = dataset.GeoPoseDataset(data_dir=ds_dir)
    # for sample in d:
    sample = next(iter(ds))

    # f, ax = plt.subplots(1, 3)
    # ax[0].imshow(sample['img'])
    # ax[1].imshow(sample['depth1'])
    # ax[2].imshow(sample['depth2'])
    # f.show()

    # todo input size for megadepth
    input_height = 384
    input_width = 512

    input_image = sample['img']
    # todo transform images for megadepth
    img_for_megadepth = np.float32(input_image) / 255.0
    # print(img_for_megadepth.shape)
    img_megadepth = resize(img_for_megadepth, (input_height, input_width), order=1)
    input_img_megadepth = torch.from_numpy(np.transpose(img_megadepth, (2, 0, 1))).contiguous().float()
    input_img_megadepth = input_img_megadepth.unsqueeze(0)
    # print(input_img_megadepth.size())

    # todo megadepth prediction
    input_images = Variable(input_img_megadepth.cuda())
    pred_log_depth = megadepth_model.netG.forward(input_images)
    pred_log_depth = torch.squeeze(pred_log_depth)
    pred_depth = torch.exp(pred_log_depth)
    # pred_inv_depth = 1 / pred_depth
    pred_inv_depth = pred_depth.data.cpu().numpy()

    # todo show megadepth
    # plt.imshow(pred_inv_depth)
    # plt.colorbar()
    # plt.show()

    # todo transform images for semseg
    img_for_semseg = np.array(input_image)
    original_resized = np.uint8(resize(input_image, (input_height, input_width), preserve_range=True))
    img_for_semseg = np.float32(resize(img_for_semseg, (input_height, input_width)))

    pil_to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # These are RGB mean+std values
            std=[0.229, 0.224, 0.225])  # across a large photo dataset.
    ])

    # todo semseg prediction
    img_data = pil_to_tensor(img_for_semseg)
    singleton_batch = {'img_data': img_data[None].cuda()}
    output_size = img_data.shape[1:]
    with torch.no_grad():
        scores = semseg_model(singleton_batch, segSize=output_size)
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()

    sky_mask = get_sky_mask(pred)
    # visualize_result(original_resized, pred)
    # print(sky_mask.shape)

    # todo apply mask
    idx = (sky_mask == 2)
    pred_inv_depth[idx] = -1

    # todo show with mask
    plt.imshow(pred_inv_depth)
    plt.colorbar()
    plt.show()