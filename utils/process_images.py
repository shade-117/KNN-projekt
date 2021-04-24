import numpy as np
import torch
from skimage.transform import resize
import torchvision.transforms
from torch.autograd import Variable


def get_sky_mask(pred, index=2):
    # index 2 is sky
    if index is not None:
        pred = pred.copy()
        pred[pred != index] = -1
        # print(f'{names[index + 1]}:')
    return pred


def transform_image_for_megadepth(input_image, input_height, input_width):
    img_for_megadepth = np.float32(input_image) / 255.0
    # print(img_for_megadepth.shape)
    img_megadepth = resize(img_for_megadepth, (input_height, input_width), order=1)
    input_img_megadepth = torch.from_numpy(np.transpose(img_megadepth, (2, 0, 1))).contiguous().float()
    input_img_megadepth = input_img_megadepth.unsqueeze(0)
    return input_img_megadepth


def transform_image_for_semseg(input_image, input_height, input_width):
    img_for_semseg = np.array(input_image)
    original_resized = np.uint8(resize(input_image, (input_height, input_width), preserve_range=True))
    img_for_semseg = np.float32(resize(img_for_semseg, (input_height, input_width)))

    pil_to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # These are RGB mean+std values
            std=[0.229, 0.224, 0.225])  # across a large photo dataset.
    ])
    img_data = pil_to_tensor(img_for_semseg)
    # return img_data
    return img_data, original_resized


def apply_sky_mask(pred, sky_mask):
    idx = (sky_mask == 2)
    pred[idx] = -1
    return pred


def megadepth_predict(model, input_img):
    input_images = Variable(input_img.cuda())
    pred_log_depth = model.hg_model.forward(input_images)
    pred_log_depth = torch.squeeze(pred_log_depth)
    pred_depth = torch.exp(pred_log_depth)
    pred_depth = pred_depth.data.cpu().numpy()
    return pred_depth


def semseg_predict(model, input_img):
    singleton_batch = {'img_data': input_img[None].cuda()}
    output_size = input_img.shape[1:]
    with torch.no_grad():
        scores = model(singleton_batch, segSize=output_size)
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()
    return pred
