import torch


def rmse_loss(pred, gt, mask=None, scale_invariant=True):
    # from rmse_error_main.py
    assert gt.shape == pred.shape, \
        'Loss tensor shapes invalid: {} x {}'.format(gt.shape, pred.shape)

    if mask is None:
        mask = torch.zeros(pred.shape) + 1

    n = torch.sum(mask)

    diff = pred - gt
    diff = torch.mul(diff, mask)

    s1 = torch.sum(torch.pow(diff, 2)) / n

    if scale_invariant:
        s2 = torch.pow(torch.sum(diff), 2) / (n * n)
        data_loss = s1 - s2
    else:
        data_loss = s1

    data_loss = torch.sqrt(data_loss)

    return data_loss


def gradient_loss(pred, gt, mask):
    # adapted from:
    # https://github.com/ArnaudFickinger/MegaDepth-Training/blob/master/models/networks.py#L231
    # ^ origin of different targets_1['mask_X']
    # https://github.com/ArnaudFickinger/MegaDepth-Training/blob/master/data/image_folder.py#L268

    def gradient_loss_inner(pred, gt, mask):
        n = torch.sum(mask)
        diff = pred - gt
        diff = torch.mul(diff, mask)

        v_gradient = torch.abs(diff[0:-2, :] - diff[2:, :])
        v_mask = torch.mul(mask[0:-2, :], mask[2:, :])
        v_gradient = torch.mul(v_gradient, v_mask)

        h_gradient = torch.abs(diff[:, 0:-2] - diff[:, 2:])
        h_mask = torch.mul(mask[:, 0:-2], mask[:, 2:])
        h_gradient = torch.mul(h_gradient, h_mask)

        loss = torch.sum(h_gradient) + torch.sum(v_gradient)
        loss = loss / n
        return loss

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

    return loss
