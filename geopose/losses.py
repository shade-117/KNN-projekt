import torch
import numpy as np


def rmse_loss(pred, gt, mask=None, scale_invariant=True):
    if mask is None:
        mask = torch.zeros(pred.shape) + 1

    n = mask.sum(dim=2).sum(dim=1) + np.finfo(float).eps

    diff = pred - gt
    diff = torch.mul(diff, mask)

    s1 = torch.pow(diff, 2).sum(dim=2).sum(dim=1) / n

    if scale_invariant:
        s2 = torch.pow(diff.sum(dim=2).sum(dim=1), 2) / (n * n)
        data_loss = s1 - s2
    else:
        data_loss = s1

    data_loss = torch.sqrt(torch.abs(data_loss))

    return data_loss.mean()


def gradient_loss(pred, gt, mask=None):
    """Gradient loss

    Forces the output relief/texture to resemble output
    Calculated at multiple scales

    """
    # adapted from:
    # https://github.com/ArnaudFickinger/MegaDepth-Training/blob/master/models/networks.py#L231
    # ^ origin of different targets_1['mask_X']
    # https://github.com/ArnaudFickinger/MegaDepth-Training/blob/master/data/image_folder.py#L268

    def gradient_loss_inner(p, g, mask):
        n = mask.sum(dim=2).sum(dim=1) + np.finfo(float).eps
        diff = p - g
        diff = torch.mul(diff, mask)

        v_gradient = torch.abs(diff[:, 0:-2, :] - diff[:, 2:, :])
        v_mask = torch.mul(mask[:, 0:-2, :], mask[:, 2:, :])
        v_gradient = torch.mul(v_gradient, v_mask)

        h_gradient = torch.abs(diff[:, :, 0:-2] - diff[:, :, 2:])
        h_mask = torch.mul(mask[:, :, 0:-2], mask[:, :, 2:])
        h_gradient = torch.mul(h_gradient, h_mask)

        loss = h_gradient.sum(dim=2).sum(dim=1) + v_gradient.sum(dim=2).sum(dim=1)
        loss = loss / n
        return loss

    if mask is None:
        mask = torch.zeros(pred.shape) + 1

    pred_div2 = pred[:, ::2, ::2]
    pred_div4 = pred_div2[:, ::2, ::2]
    pred_div8 = pred_div4[:, ::2, ::2]

    # downsampling can be calculated during loading dataset
    # but then we would have to apply the same augmentations to these
    gt_div2 = gt[:, ::2, ::2]
    gt_div4 = gt_div2[:, ::2, ::2]
    gt_div8 = gt_div4[:, ::2, ::2]

    mask_div2 = mask[:, ::2, ::2]
    mask_div4 = mask_div2[:, ::2, ::2]
    mask_div8 = mask_div4[:, ::2, ::2]

    loss = gradient_loss_inner(pred, gt, mask)
    loss += gradient_loss_inner(pred_div2, gt_div2, mask_div2)
    loss += gradient_loss_inner(pred_div4, gt_div4, mask_div4)
    loss += gradient_loss_inner(pred_div8, gt_div8, mask_div8)

    return loss.mean()
