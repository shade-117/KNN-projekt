import torch
import sys

import torchvision
from torch.autograd import Variable
import numpy as np
from options.train_options import TrainOptions

opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
from data.data_loader import CreateDataLoader
from models.models import create_model
from skimage import io
from skimage.transform import resize
import matplotlib.pylab as plt
import os, csv, torch, numpy, scipy.io, PIL.Image, torchvision.transforms
# Our libs
from megadepth.models.semseg.models import ModelBuilder, SegmentationModule
from megadepth.models.semseg.utils import colorEncode

img_path = 'image.jpg'

model = create_model(opt)

input_height = 384
input_width = 512

colors = scipy.io.loadmat('data/color150.mat')['colors']
names = {}
with open('data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]

    # Network Builders
    # todo download weights: http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth
    # todo  and http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth
    # todo and put them into megadepth/checkpoints/semseg/ade20k-resnet50dilated-ppm_deepsup
net_encoder = ModelBuilder.build_encoder(
    arch='resnet50dilated',
    fc_dim=2048,
    weights='checkpoints/semseg/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth')
net_decoder = ModelBuilder.build_decoder(
    arch='ppm_deepsup',
    fc_dim=2048,
    num_class=150,
    weights='checkpoints/semseg/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth',
    use_softmax=True)

crit = torch.nn.NLLLoss(ignore_index=-1)
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
segmentation_module.eval()
segmentation_module.cuda()


def visualize_result(img, pred, index=None):
    # filter prediction class if requested
    if index is not None:
        pred = pred.copy()
        pred[pred != index] = -1
        print(f'{names[index + 1]}:')

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(numpy.uint8)

    # aggregate images and save
    im_vis = numpy.concatenate((img, pred_color), axis=1)
    plt.imshow(PIL.Image.fromarray(im_vis))
    plt.show()


def test_simple(model):
    total_loss = 0
    toal_count = 0
    print("============================= TEST ============================")
    model.switch_to_eval()

    img = np.float32(io.imread(img_path)) / 255.0
    img = resize(img, (input_height, input_width), order=1)
    input_img = torch.from_numpy(np.transpose(img, (2, 0, 1))).contiguous().float()
    input_img = input_img.unsqueeze(0)




    pil_to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # These are RGB mean+std values
            std=[0.229, 0.224, 0.225])  # across a large photo dataset.
    ])

    pil_image = PIL.Image.open('image.jpg').convert('RGB')
    img_original = numpy.array(pil_image)
    img_data = pil_to_tensor(pil_image)
    singleton_batch = {'img_data': img_data[None].cuda()}
    output_size = img_data.shape[1:]
    with torch.no_grad():
        scores = segmentation_module(singleton_batch, segSize=output_size)
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()

    visualize_result(img_original, pred, index=2)

    input_images = Variable(input_img.cuda())
    pred_log_depth = model.netG.forward(input_images)
    pred_log_depth = torch.squeeze(pred_log_depth)

    pred_depth = torch.exp(pred_log_depth)

    # visualize prediction using inverse depth, so that we don't need sky segmentation (if you want to use RGB map for visualization, \
    # you have to run semantic segmentation to mask the sky first since the depth of sky is random from CNN)
    # pred_inv_depth = 1 / pred_depth
    pred_inv_depth = pred_depth.data.cpu().numpy()
    # you might also use percentile for better visualization
    # pred_inv_depth = pred_inv_depth / np.amax(pred_inv_depth)
    # pred_inv_depth = np.where(pred_inv_depth > 3, -1, pred_inv_depth)
    plt.imshow(pred_inv_depth)
    plt.colorbar()
    plt.show()

    io.imsave('image2.png', pred_inv_depth)
    # print(pred_inv_depth.shape)


test_simple(model)
print("We are done")
