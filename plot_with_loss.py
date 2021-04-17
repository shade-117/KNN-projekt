# stdlib
import time

# external
from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

# local
from geopose.dataset import GeoPoseDataset, clear_dataset_dir, rotate_images


def rmse_loss(log_prediction_d, log_gt, mask=None, scale_invariant=True):
    # from rmse_error_main.py
    assert log_gt.shape == log_prediction_d.shape
    if mask is None:
        mask = torch.Tensor(np.zeros_like(log_prediction_d) + 1)

    n = torch.sum(mask)

    log_d_diff = log_prediction_d - log_gt
    log_d_diff = torch.mul(log_d_diff, mask)

    s1 = torch.sum(torch.pow(log_d_diff, 2)) / n
    s2 = torch.pow(torch.sum(log_d_diff), 2) / (n * n)

    if scale_invariant:
        data_loss = s1 - s2
    else:
        data_loss = s1

    data_loss = torch.sqrt(data_loss)

    return data_loss


if __name__ == '__main__':
    # working folder 'KNN-projekt' assumed

    # make a symlink to the dataset or put it into main project folder:
    # ln -s {{path/to/file_or_directory}} {{path/to/symlink}}

    ds_dir = 'datasets/geoPose3K_final_publish/'
    clear_dataset_dir(ds_dir)
    # rotate_images(ds_dir)

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.CenterCrop((384, 512))]
                                        )

    ds = GeoPoseDataset(ds_dir=ds_dir, transforms=data_transform, verbose=False)
    ds_len = len(ds)
    # batch size and num_workers are chosen arbitrarily, try your own ideas
    loader = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=4, collate_fn=ds.collate)
    si_full_loss = 0
    nsi_full_loss = 0
    count = 0
    random_ind = np.random.choice(range(0, ds_len), 10)
    print(random_ind)
    for rndidx in random_ind:
        print(rndidx)
        img, gt, mask, dirpath = ds[int(rndidx)]
        start = time.time()
        # get prediction and gt, resize gt
        pred = np.load(dirpath + '/depth_map_no_sky.npy')
        gt = resize(gt, pred.shape)
        # union the sky masks
        mask_gt = np.where(gt == -1, 0, 1)
        mask_pred = np.where(pred == -1, 0, 1)
        union_mask = torch.Tensor(mask_gt * mask_pred)

        fig, ((ax11, ax12, ax13), (ax21, ax22, ax23)) = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))
        ax11.imshow(img)
        ax12.imshow(gt)
        ax13.imshow(pred)
        # convert to tensor, add +2 to remove values -1 and 0
        pred_t = torch.Tensor(np.log(pred + 2))
        gt_t = torch.Tensor(np.log(gt + 2))

        idx = (union_mask == 0)
        predc = np.copy(pred)
        predc[idx] = -1

        gtc = np.copy(gt)
        gtc[idx] = -1
        ax21.imshow(mask_gt - mask_pred)
        ax22.imshow(gtc)
        ax23.imshow(predc)
        si_loss = rmse_loss(pred_t, gt_t, mask=union_mask, scale_invariant=True)
        not_si_loss = rmse_loss(pred_t, gt_t, mask=union_mask, scale_invariant=False)
        si_loss_num = f'{si_loss.numpy():.3f}'
        not_si_loss_num = f'{not_si_loss.numpy():.3f}'
        plt_title = f'si-RMSE {si_loss_num}\nRMSE {not_si_loss_num}\nindex: {rndidx}'

        fig.suptitle(plt_title, fontsize=16)
        fig.show()
        plt.show()
        end = time.time()
        took = end - start
        print(f'{count}/{ds_len}, LOSS: {si_loss}, took: {took:.3f}s', sep=' ', end='\r', flush=True)
        si_full_loss += si_loss
        nsi_full_loss += not_si_loss
        count += 1
        # plt.show()

    print("\ncount: ", count)
    print("\nsi-RMSE: ", si_full_loss)
    print("si-RMSE/count", si_full_loss / count)
    print("\nRMSE: ", nsi_full_loss)
    print("RMSE/count", nsi_full_loss / count)