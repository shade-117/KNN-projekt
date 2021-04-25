# stdlib
import os
import shutil
import sys
from functools import reduce

# external
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl

# local
from torchvision import transforms

from geopose.losses import rmse_loss, gradient_loss
from geopose.dataset import get_dataset_loaders, GeoPoseDataset


class HourglassModel(pl.LightningModule):
    @staticmethod
    def name():
        return 'HourglassModel'

    def __init__(self, opt, weights_path=None, *args, **kwargs):
        # new, removing the following from kwargs because the base class does not support them
        lr_coef = kwargs.pop('lr_coef', 100)
        self.quiet = kwargs.pop('quiet', True)
        self.scale_invariancy = kwargs.pop('scale_invariancy', False)
        self.batch_size = kwargs.pop('batch_size', 1)
        self.running_in_colab = kwargs.pop('running_in_colab', False)
        self.drive_outputs_path = kwargs.pop('drive_outputs_path', None)

        super(HourglassModel, self).__init__(*args, **kwargs)

        self.lr = opt.lr * lr_coef
        self.opt = opt
        # self.gpu_ids = opt.gpu_ids
        self.gpu_ids = [0]
        self.isTrain = opt.isTrain
        # self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        print("============= LOADING Hourglass NETWORK =============")
        self.model = hg_model_old  # todo try and verify new hg_model
        self.model = torch.nn.parallel.DataParallel(self.model, device_ids=[0])

        if weights_path is None:
            self.load_network('G', 'best_generalization')

            # self.load_network('G', 'saved')
            # self.load_network('G', 'saved_1480.2093')
            # self.load_network('G', 'saved_1457.3193')
            # self.load_network('G', 'saved_1436.4768')
            # self.load_network('G', 'saved_600batches_no-logs_lr2_by_100')
            # self.load_network('G', 'saved_1.1111')
        else:
            self.model.load_state_dict(torch.load(weights_path))

        self.model = self.model.cuda()

    def forward(self, x):
        pred = self.model(x)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(self.opt.beta1, 0.999))
        return optimizer

    def training_step(self, train_batch, batch_idx):
        imgs = train_batch['img']

        depths = train_batch['depth']
        masks = train_batch['mask']
        # paths = train_batch['path']

        preds = self.model(imgs)
        preds = torch.squeeze(preds, dim=1)

        data_loss = rmse_loss(preds, depths, masks, scale_invariant=self.scale_invariancy)
        grad_loss = gradient_loss(preds, depths, masks)
        joint_loss = (data_loss + 0.5 * grad_loss)

        self.log('data_{}loss'.format('si_' * self.scale_invariancy), data_loss, on_epoch=True)
        self.log('grad_loss', grad_loss, on_epoch=True)
        self.log('joint_loss', joint_loss, on_epoch=True)

        return joint_loss

    def validation_step(self, val_batch, val_idx):
        imgs = val_batch['img']

        depths = val_batch['depth']
        masks = val_batch['mask']
        # paths = val_batch['path']

        preds = self.model(imgs)
        preds = torch.squeeze(preds, dim=1)

        # print('\nshape', preds.shape, depths.shape, masks.shape)
        # print('\nisnan', torch.isnan(preds).sum(), torch.isnan(depths).sum(), torch.isnan(masks).sum())

        data_loss = rmse_loss(preds, depths, masks, scale_invariant=False)
        data_si_loss = rmse_loss(preds, depths, masks, scale_invariant=True)
        grad_loss = gradient_loss(preds, depths, masks)
        joint_loss = (data_loss + 0.5 * grad_loss)

        # print('\nloss', data_loss, data_si_loss, grad_loss)

        self.log('val_data_loss', data_loss, on_epoch=True)
        self.log('val_data_si_loss', data_si_loss, on_epoch=True)
        self.log('val_grad_loss', grad_loss, on_epoch=True)
        self.log('val_joint_loss', joint_loss, on_epoch=True)

        return joint_loss

    def training_epoch_end(self, outputs):
        if self.running_in_colab:
            # copy logs to gdrive - HourglassModel needs drive_outputs_path
            # todo finish
            # shutil.copytree('logs', self.drive_outputs_path)
            pass

    def save_network(self, network_label, epoch_label):
        save_filename = '_%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(self.model.cpu().state_dict(), save_path)
        if len(self.gpu_ids) and torch.cuda.is_available():
            self.model.cuda()

    def load_network(self, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = self.save_dir + os.sep + save_filename
        print('Loading model weights from:', save_path)
        if not os.path.isfile(save_path):
            save_path = os.path.join('megadepth', 'checkpoints', save_path)
            if not os.path.isfile(save_path):
                raise UserWarning('base_model: load_network: Weights not loaded :C\ncould not find file:', save_path, file=sys.stderr)

        self.model.load_state_dict(torch.load(save_path))


    """ No class members past here are actually used """

    @staticmethod
    def batch_classify(z_A_arr, z_B_arr, ground_truth):
        threshold = 1.1
        depth_ratio = torch.div(z_A_arr, z_B_arr)

        depth_ratio = depth_ratio.cpu()

        estimated_labels = torch.zeros(depth_ratio.size(0))

        estimated_labels[depth_ratio > threshold] = 1
        estimated_labels[depth_ratio < (1 / threshold)] = -1

        diff = estimated_labels - ground_truth
        diff[diff != 0] = 1

        # error
        inequal_error_count = diff[ground_truth != 0]
        inequal_error_count = torch.sum(inequal_error_count)

        error_count = torch.sum(diff)  # diff[diff !=0]
        # error_count = error_count.size(0)

        equal_error_count = error_count - inequal_error_count

        # total
        total_count = depth_ratio.size(0)
        ground_truth[ground_truth != 0] = 1

        inequal_count_total = torch.sum(ground_truth)
        equal_total_count = total_count - inequal_count_total

        error_list = [equal_error_count, inequal_error_count, error_count]
        count_list = [equal_total_count, inequal_count_total, total_count]

        return error_list, count_list

    def compute_sdr(self, prediction_d, targets):
        #  for each image
        total_error = [0, 0, 0]
        total_samples = [0, 0, 0]

        for i in range(0, prediction_d.size(0)):

            if not targets['has_SfM_feature'][i]:
                continue

            x_A_arr = targets["sdr_xA"][i].squeeze(0)
            x_B_arr = targets["sdr_xB"][i].squeeze(0)
            y_A_arr = targets["sdr_yA"][i].squeeze(0)
            y_B_arr = targets["sdr_yB"][i].squeeze(0)

            predict_depth = torch.exp(prediction_d[i, :, :])
            predict_depth = predict_depth.squeeze(0)
            ground_truth = targets["sdr_gt"][i]

            # print(x_A_arr.size())
            # print(y_A_arr.size())

            z_A_arr = torch.gather(torch.index_select(predict_depth, 1, x_A_arr.cuda()), 0, y_A_arr.view(1,
                                                                                                         -1).cuda())  # predict_depth:index(2, x_A_arr):gather(1, y_A_arr:view(1, -1))
            z_B_arr = torch.gather(torch.index_select(predict_depth, 1, x_B_arr.cuda()), 0, y_B_arr.view(1, -1).cuda())

            z_A_arr = z_A_arr.squeeze(0)
            z_B_arr = z_B_arr.squeeze(0)

            error_list, count_list = self.batch_classify(z_A_arr, z_B_arr, ground_truth)

            for j in range(0, 3):
                total_error[j] += error_list[j]
                total_samples[j] += count_list[j]

        return total_error, total_samples

    def evaluate_sdr(self, input_, targets):
        input_images = Variable(input_.cuda())
        prediction_d = self.model.forward(input_images)

        total_error, total_samples = self.compute_sdr(prediction_d.data, targets)

        return total_error, total_samples

    @staticmethod
    def rmse_loss(log_prediction_d, mask, log_gt):
        n = torch.sum(mask)
        log_d_diff = log_prediction_d - log_gt
        log_d_diff = torch.mul(log_d_diff, mask)
        s1 = torch.sum(torch.pow(log_d_diff, 2)) / n

        s2 = torch.pow(torch.sum(log_d_diff), 2) / (n * n)
        data_loss = s1 - s2

        data_loss = torch.sqrt(data_loss)

        return data_loss

    def evaluate_rmse(self, input_images, prediction_d, targets):
        # input_images unused
        count = 0
        total_loss = Variable(torch.cuda.FloatTensor(1))
        total_loss[0] = 0
        mask_0 = Variable(targets['mask_0'].cuda(), requires_grad=False)
        d_gt_0 = torch.log(Variable(targets['gt_0'].cuda(), requires_grad=False))

        for i in range(0, mask_0.size(0)):
            total_loss += self.rmse_loss(prediction_d[i, :, :], mask_0[i, :, :], d_gt_0[i, :, :])
            count += 1

        return total_loss.data[0], count

    def evaluate_sc_inv(self, input_, targets):
        input_images = Variable(input_.cuda())
        prediction_d = self.model.forward(input_images)
        rmse_loss, count = self.evaluate_rmse(input_images, prediction_d, targets)

        return rmse_loss, count

    def switch_to_train(self):
        self.model.train()

    def switch_to_eval(self):
        self.model.eval()


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input_):
        output = []
        for module in self._modules.values():
            output.append(module(input_))
        return output if output else input_


class Lambda(LambdaBase):
    def forward(self, input_):
        return self.lambda_func(self.forward_prepare(input_))


class Map(LambdaBase):
    def forward(self, input_):
        return list(map(self.lambda_func, self.forward_prepare(input_)))


class Reduce(LambdaBase):
    def forward(self, input_):
        return reduce(self.lambda_func, self.forward_prepare(input_))


def bn(c):
    return nn.BatchNorm2d(c, 1e-05, 0.1, False)


def conv11(c_in, c_out):
    return nn.Conv2d(c_in, c_out, (11, 11), (1, 1), (5, 5)), bn(c_out), nn.ReLU()


def conv7(c_in, c_out):
    return nn.Conv2d(c_in, c_out, (7, 7), (1, 1), (3, 3)), bn(c_out), nn.ReLU()


def conv5(c_in, c_out):
    return nn.Conv2d(c_in, c_out, (5, 5), (1, 1), (2, 2)), bn(c_out), nn.ReLU()


def conv3(c_in, c_out):
    return nn.Conv2d(c_in, c_out, (3, 3), (1, 1), (1, 1)), bn(c_out), nn.ReLU()


def conv1(c_in, c_out):
    return nn.Conv2d(c_in, c_out, (1, 1)), bn(c_out), nn.ReLU()


# @formatter:off
# noinspection DuplicatedCode
hg_model = nn.Sequential(
    *conv7(3, 128), nn.Sequential(
        Map(lambda x: x,  # ConcatTable,
            nn.Sequential(
                nn.MaxPool2d((2, 2), (2, 2)),
                Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),  # Concat,
                       nn.Sequential(*conv1(128, 32)),
                       nn.Sequential(*conv1(128, 32), *conv3(32, 32)),
                       nn.Sequential(*conv1(128, 32), *conv5(32, 32)),
                       nn.Sequential(*conv1(128, 32), *conv7(32, 32))),
                Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),  # Concat,
                       nn.Sequential(*conv1(128, 32)),
                       nn.Sequential(*conv1(128, 32), *conv3(32, 32)),
                       nn.Sequential(*conv1(128, 32), *conv5(32, 32)),
                       nn.Sequential(*conv1(128, 32), *conv7(32, 32))),
                nn.Sequential(Map(lambda x: x,  # ConcatTable,
                                  nn.Sequential(
                                      nn.MaxPool2d((2, 2), (2, 2)),
                                      Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),  # Concat,
                                             nn.Sequential(*conv1(128, 32)),
                                             nn.Sequential(*conv1(128, 32), *conv3(32, 32)),
                                             nn.Sequential(*conv1(128, 32), *conv5(32, 32)),
                                             nn.Sequential(*conv1(128, 32), *conv7(32, 32))),
                                      Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),  # Concat,
                                             nn.Sequential(*conv1(128, 64)),
                                             nn.Sequential(*conv1(128, 32), *conv3(32, 64)),
                                             nn.Sequential(*conv1(128, 32), *conv5(32, 64)),
                                             nn.Sequential(*conv1(128, 32), *conv7(32, 64))),
                                      nn.Sequential(Map(lambda x: x,  # ConcatTable,
                                                        nn.Sequential(
                                                            Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),
                                                                   # Concat,
                                                                   nn.Sequential(*conv1(256, 64)),
                                                                   nn.Sequential(*conv1(256, 32), *conv3(32, 64)),
                                                                   nn.Sequential(*conv1(256, 32), *conv5(32, 64)),
                                                                   nn.Sequential(*conv1(256, 32), *conv7(32, 64))),
                                                            Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),
                                                                   # Concat,
                                                                   nn.Sequential(*conv1(256, 64)),
                                                                   nn.Sequential(*conv1(256, 64), *conv3(64, 64)),
                                                                   nn.Sequential(*conv1(256, 64), *conv7(64, 64)),
                                                                   nn.Sequential(*conv1(256, 64), *conv11(64, 64)))),
                                                        nn.Sequential(
                                                            nn.AvgPool2d((2, 2), (2, 2)),
                                                            Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),
                                                                   # Concat,
                                                                   nn.Sequential(*conv1(256, 64)),
                                                                   nn.Sequential(*conv1(256, 32), *conv3(32, 64)),
                                                                   nn.Sequential(*conv1(256, 32), *conv5(32, 64)),
                                                                   nn.Sequential(*conv1(256, 32), *conv7(32, 64))),
                                                            Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),
                                                                   # Concat,
                                                                   nn.Sequential(*conv1(256, 64)),
                                                                   nn.Sequential(*conv1(256, 32), *conv3(32, 64)),
                                                                   nn.Sequential(*conv1(256, 32), *conv5(32, 64)),
                                                                   nn.Sequential(*conv1(256, 32), *conv7(32, 64))),
                                                            nn.Sequential(
                                                                Map(lambda x: x,  # ConcatTable,
                                                                    nn.Sequential(
                                                                        Reduce(
                                                                            lambda x, y, dim=1: torch.cat((x, y), dim),
                                                                            # Concat,
                                                                            nn.Sequential(
                                                                                *conv1(256, 64)),
                                                                            nn.Sequential(
                                                                                *conv1(256, 32), *conv3(32, 64)),
                                                                            nn.Sequential(
                                                                                *conv1(256, 32), *conv5(32, 64)),
                                                                            nn.Sequential(
                                                                                *conv1(256, 32), *conv7(32, 64))),
                                                                        Reduce(
                                                                            lambda x, y, dim=1: torch.cat((x, y), dim),
                                                                            # Concat,
                                                                            nn.Sequential(
                                                                                *conv1(256, 64)),
                                                                            nn.Sequential(
                                                                                *conv1(256, 32), *conv3(32, 64)),
                                                                            nn.Sequential(
                                                                                *conv1(256, 32), *conv5(32, 64)),
                                                                            nn.Sequential(
                                                                                *conv1(256, 32), *conv7(32, 64)))),
                                                                    nn.Sequential(
                                                                        nn.AvgPool2d((2, 2), (2, 2)),
                                                                        Reduce(
                                                                            lambda x, y, dim=1: torch.cat((x, y), dim),
                                                                            # Concat,
                                                                            nn.Sequential(
                                                                                *conv1(256, 64)),
                                                                            nn.Sequential(
                                                                                *conv1(256, 32), *conv3(32, 64)),
                                                                            nn.Sequential(
                                                                                *conv1(256, 32), *conv5(32, 64)),
                                                                            nn.Sequential(
                                                                                *conv1(256, 32), *conv7(32, 64))),
                                                                        Reduce(
                                                                            lambda x, y, dim=1: torch.cat((x, y), dim),
                                                                            # Concat,
                                                                            nn.Sequential(
                                                                                *conv1(256, 64)),
                                                                            nn.Sequential(
                                                                                *conv1(256, 32), *conv3(32, 64)),
                                                                            nn.Sequential(
                                                                                *conv1(256, 32), *conv5(32, 64)),
                                                                            nn.Sequential(
                                                                                *conv1(256, 32), *conv7(32, 64))),
                                                                        Reduce(
                                                                            lambda x, y, dim=1: torch.cat((x, y), dim),
                                                                            # Concat,
                                                                            nn.Sequential(
                                                                                *conv1(256, 64)),
                                                                            nn.Sequential(
                                                                                *conv1(256, 32), *conv3(32, 64)),
                                                                            nn.Sequential(
                                                                                *conv1(256, 32), *conv5(32, 64)),
                                                                            nn.Sequential(
                                                                                *conv1(256, 32), *conv7(32, 64))),
                                                                        nn.UpsamplingNearest2d(
                                                                            scale_factor=2), ), ),
                                                                Reduce(lambda x, y: x + y),  # CAddTable,
                                                            ), Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),
                                                                      # Concat,
                                                                      nn.Sequential(*conv1(256, 64)),
                                                                      nn.Sequential(*conv1(256, 32), *conv3(32, 64)),
                                                                      nn.Sequential(*conv1(256, 32), *conv5(32, 64)),
                                                                      nn.Sequential(*conv1(256, 32), *conv7(32, 64))),
                                                            Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),
                                                                   # Concat,
                                                                   nn.Sequential(*conv1(256, 64)),
                                                                   nn.Sequential(*conv1(256, 64), *conv3(64, 64)),
                                                                   nn.Sequential(*conv1(256, 64), *conv7(64, 64)),
                                                                   nn.Sequential(*conv1(256, 64), *conv11(64, 64))),
                                                            nn.UpsamplingNearest2d(scale_factor=2))),
                                                    Reduce(lambda x, y: x + y),  # CAddTable,
                                                    ),
                                      Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),  # Concat,
                                             nn.Sequential(*conv1(256, 64)),
                                             nn.Sequential(*conv1(256, 32), *conv3(32, 64)),
                                             nn.Sequential(*conv1(256, 32), *conv5(32, 64)),
                                             nn.Sequential(*conv1(256, 32), *conv7(32, 64))),
                                      Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),  # Concat,
                                             nn.Sequential(*conv1(256, 32)),
                                             nn.Sequential(*conv1(256, 32), *conv3(32, 32)),
                                             nn.Sequential(*conv1(256, 32), *conv5(32, 32)),
                                             nn.Sequential(*conv1(256, 32), *conv7(32, 32))),
                                      nn.UpsamplingNearest2d(scale_factor=2)),
                                  nn.Sequential(
                                      Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),  # Concat,
                                             nn.Sequential(*conv1(128, 32)),
                                             nn.Sequential(*conv1(128, 32), *conv3(32, 32)),
                                             nn.Sequential(*conv1(128, 32), *conv5(32, 32)),
                                             nn.Sequential(*conv1(128, 32), *conv7(32, 32))),
                                      Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),  # Concat,
                                             nn.Sequential(*conv1(128, 32)),
                                             nn.Sequential(*conv1(128, 64), *conv3(64, 32)),
                                             nn.Sequential(*conv1(128, 64), *conv7(64, 32)),
                                             nn.Sequential(*conv1(128, 64), *conv11(64, 32))))),
                              Reduce(lambda x, y: x + y),  # CAddTable,
                              ),
                Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),  # Concat,
                       nn.Sequential(*conv1(128, 32)),
                       nn.Sequential(*conv1(128, 64), *conv3(64, 32)),
                       nn.Sequential(*conv1(128, 64), *conv5(64, 32)),
                       nn.Sequential(*conv1(128, 64), *conv7(64, 32))),
                Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),  # Concat,
                       nn.Sequential(*conv1(128, 16)),
                       nn.Sequential(*conv1(128, 32), *conv3(32, 16)),
                       nn.Sequential(*conv1(128, 32), *conv7(32, 16)),
                       nn.Sequential(*conv1(128, 32), *conv11(32, 16))),
                nn.UpsamplingNearest2d(scale_factor=2)),
            nn.Sequential(
                Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),  # Concat,
                       nn.Sequential(*conv1(128, 16)),
                       nn.Sequential(*conv1(128, 64), *conv3(64, 16)),
                       nn.Sequential(*conv1(128, 64), *conv7(64, 16)),
                       nn.Sequential(*conv1(128, 64), *conv11(64, 16))))),
        Reduce(lambda x, y: x + y),
        # CAddTable,
    ), nn.Conv2d(64, 1, (3, 3), (1, 1), (1, 1)))


hg_model_old = nn.Sequential(  # Sequential,
    nn.Conv2d(3, 128, (7, 7), (1, 1), (3, 3)),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.Sequential(  # Sequential,
        Map(lambda x: x,  # ConcatTable,
                  nn.Sequential(  # Sequential,
                      nn.MaxPool2d((2, 2), (2, 2)),
                      Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),  # Concat,
                                   nn.Sequential(  # Sequential,
                                       nn.Conv2d(128, 32, (1, 1)),
                                       nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                   ),
                                   nn.Sequential(  # Sequential,
                                       nn.Conv2d(128, 32, (1, 1)),
                                       nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                       nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1)),
                                       nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                   ),
                                   nn.Sequential(  # Sequential,
                                       nn.Conv2d(128, 32, (1, 1)),
                                       nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                       nn.Conv2d(32, 32, (5, 5), (1, 1), (2, 2)),
                                       nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                   ),
                                   nn.Sequential(  # Sequential,
                                       nn.Conv2d(128, 32, (1, 1)),
                                       nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                       nn.Conv2d(32, 32, (7, 7), (1, 1), (3, 3)),
                                       nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                   ),
                                   ),
                      Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),  # Concat,
                                   nn.Sequential(  # Sequential,
                                       nn.Conv2d(128, 32, (1, 1)),
                                       nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                   ),
                                   nn.Sequential(  # Sequential,
                                       nn.Conv2d(128, 32, (1, 1)),
                                       nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                       nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1)),
                                       nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                   ),
                                   nn.Sequential(  # Sequential,
                                       nn.Conv2d(128, 32, (1, 1)),
                                       nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                       nn.Conv2d(32, 32, (5, 5), (1, 1), (2, 2)),
                                       nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                   ),
                                   nn.Sequential(  # Sequential,
                                       nn.Conv2d(128, 32, (1, 1)),
                                       nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                       nn.Conv2d(32, 32, (7, 7), (1, 1), (3, 3)),
                                       nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                   ),
                                   ),
                      nn.Sequential(  # Sequential,
                          Map(lambda x: x,  # ConcatTable,
                                    nn.Sequential(  # Sequential,
                                        nn.MaxPool2d((2, 2), (2, 2)),
                                        Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),  # Concat,
                                                     nn.Sequential(  # Sequential,
                                                         nn.Conv2d(128, 32, (1, 1)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                     ),
                                                     nn.Sequential(  # Sequential,
                                                         nn.Conv2d(128, 32, (1, 1)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                         nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                     ),
                                                     nn.Sequential(  # Sequential,
                                                         nn.Conv2d(128, 32, (1, 1)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                         nn.Conv2d(32, 32, (5, 5), (1, 1), (2, 2)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                     ),
                                                     nn.Sequential(  # Sequential,
                                                         nn.Conv2d(128, 32, (1, 1)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                         nn.Conv2d(32, 32, (7, 7), (1, 1), (3, 3)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                     ),
                                                     ),
                                        Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),  # Concat,
                                                     nn.Sequential(  # Sequential,
                                                         nn.Conv2d(128, 64, (1, 1)),
                                                         nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                     ),
                                                     nn.Sequential(  # Sequential,
                                                         nn.Conv2d(128, 32, (1, 1)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                         nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1)),
                                                         nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                     ),
                                                     nn.Sequential(  # Sequential,
                                                         nn.Conv2d(128, 32, (1, 1)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                         nn.Conv2d(32, 64, (5, 5), (1, 1), (2, 2)),
                                                         nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                     ),
                                                     nn.Sequential(  # Sequential,
                                                         nn.Conv2d(128, 32, (1, 1)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                         nn.Conv2d(32, 64, (7, 7), (1, 1), (3, 3)),
                                                         nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                     ),
                                                     ),
                                        nn.Sequential(  # Sequential,
                                            Map(lambda x: x,  # ConcatTable,
                                                      nn.Sequential(  # Sequential,
                                                          Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),
                                                                       # Concat,
                                                                       nn.Sequential(  # Sequential,
                                                                           nn.Conv2d(256, 64, (1, 1)),
                                                                           nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                       ),
                                                                       nn.Sequential(  # Sequential,
                                                                           nn.Conv2d(256, 32, (1, 1)),
                                                                           nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                           nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1)),
                                                                           nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                       ),
                                                                       nn.Sequential(  # Sequential,
                                                                           nn.Conv2d(256, 32, (1, 1)),
                                                                           nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                           nn.Conv2d(32, 64, (5, 5), (1, 1), (2, 2)),
                                                                           nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                       ),
                                                                       nn.Sequential(  # Sequential,
                                                                           nn.Conv2d(256, 32, (1, 1)),
                                                                           nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                           nn.Conv2d(32, 64, (7, 7), (1, 1), (3, 3)),
                                                                           nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                       ),
                                                                       ),
                                                          Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),
                                                                       # Concat,
                                                                       nn.Sequential(  # Sequential,
                                                                           nn.Conv2d(256, 64, (1, 1)),
                                                                           nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                       ),
                                                                       nn.Sequential(  # Sequential,
                                                                           nn.Conv2d(256, 64, (1, 1)),
                                                                           nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                           nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
                                                                           nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                       ),
                                                                       nn.Sequential(  # Sequential,
                                                                           nn.Conv2d(256, 64, (1, 1)),
                                                                           nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                           nn.Conv2d(64, 64, (7, 7), (1, 1), (3, 3)),
                                                                           nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                       ),
                                                                       nn.Sequential(  # Sequential,
                                                                           nn.Conv2d(256, 64, (1, 1)),
                                                                           nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                           nn.Conv2d(64, 64, (11, 11), (1, 1), (5, 5)),
                                                                           nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                       ),
                                                                       ),
                                                      ),
                                                      nn.Sequential(  # Sequential,
                                                          nn.AvgPool2d((2, 2), (2, 2)),
                                                          Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),
                                                                       # Concat,
                                                                       nn.Sequential(  # Sequential,
                                                                           nn.Conv2d(256, 64, (1, 1)),
                                                                           nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                       ),
                                                                       nn.Sequential(  # Sequential,
                                                                           nn.Conv2d(256, 32, (1, 1)),
                                                                           nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                           nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1)),
                                                                           nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                       ),
                                                                       nn.Sequential(  # Sequential,
                                                                           nn.Conv2d(256, 32, (1, 1)),
                                                                           nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                           nn.Conv2d(32, 64, (5, 5), (1, 1), (2, 2)),
                                                                           nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                       ),
                                                                       nn.Sequential(  # Sequential,
                                                                           nn.Conv2d(256, 32, (1, 1)),
                                                                           nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                           nn.Conv2d(32, 64, (7, 7), (1, 1), (3, 3)),
                                                                           nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                       ),
                                                                       ),
                                                          Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),
                                                                       # Concat,
                                                                       nn.Sequential(  # Sequential,
                                                                           nn.Conv2d(256, 64, (1, 1)),
                                                                           nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                       ),
                                                                       nn.Sequential(  # Sequential,
                                                                           nn.Conv2d(256, 32, (1, 1)),
                                                                           nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                           nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1)),
                                                                           nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                       ),
                                                                       nn.Sequential(  # Sequential,
                                                                           nn.Conv2d(256, 32, (1, 1)),
                                                                           nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                           nn.Conv2d(32, 64, (5, 5), (1, 1), (2, 2)),
                                                                           nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                       ),
                                                                       nn.Sequential(  # Sequential,
                                                                           nn.Conv2d(256, 32, (1, 1)),
                                                                           nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                           nn.Conv2d(32, 64, (7, 7), (1, 1), (3, 3)),
                                                                           nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                       ),
                                                                       ),
                                                          nn.Sequential(  # Sequential,
                                                              Map(lambda x: x,  # ConcatTable,
                                                                        nn.Sequential(  # Sequential,
                                                                            Reduce(
                                                                                lambda x, y, dim=1: torch.cat((x, y),
                                                                                                              dim),
                                                                                # Concat,
                                                                                nn.Sequential(  # Sequential,
                                                                                    nn.Conv2d(256, 64, (1, 1)),
                                                                                    nn.BatchNorm2d(64, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                ),
                                                                                nn.Sequential(  # Sequential,
                                                                                    nn.Conv2d(256, 32, (1, 1)),
                                                                                    nn.BatchNorm2d(32, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                    nn.Conv2d(32, 64, (3, 3), (1, 1),
                                                                                              (1, 1)),
                                                                                    nn.BatchNorm2d(64, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                ),
                                                                                nn.Sequential(  # Sequential,
                                                                                    nn.Conv2d(256, 32, (1, 1)),
                                                                                    nn.BatchNorm2d(32, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                    nn.Conv2d(32, 64, (5, 5), (1, 1),
                                                                                              (2, 2)),
                                                                                    nn.BatchNorm2d(64, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                ),
                                                                                nn.Sequential(  # Sequential,
                                                                                    nn.Conv2d(256, 32, (1, 1)),
                                                                                    nn.BatchNorm2d(32, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                    nn.Conv2d(32, 64, (7, 7), (1, 1),
                                                                                              (3, 3)),
                                                                                    nn.BatchNorm2d(64, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                ),
                                                                            ),
                                                                            Reduce(
                                                                                lambda x, y, dim=1: torch.cat((x, y),
                                                                                                              dim),
                                                                                # Concat,
                                                                                nn.Sequential(  # Sequential,
                                                                                    nn.Conv2d(256, 64, (1, 1)),
                                                                                    nn.BatchNorm2d(64, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                ),
                                                                                nn.Sequential(  # Sequential,
                                                                                    nn.Conv2d(256, 32, (1, 1)),
                                                                                    nn.BatchNorm2d(32, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                    nn.Conv2d(32, 64, (3, 3), (1, 1),
                                                                                              (1, 1)),
                                                                                    nn.BatchNorm2d(64, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                ),
                                                                                nn.Sequential(  # Sequential,
                                                                                    nn.Conv2d(256, 32, (1, 1)),
                                                                                    nn.BatchNorm2d(32, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                    nn.Conv2d(32, 64, (5, 5), (1, 1),
                                                                                              (2, 2)),
                                                                                    nn.BatchNorm2d(64, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                ),
                                                                                nn.Sequential(  # Sequential,
                                                                                    nn.Conv2d(256, 32, (1, 1)),
                                                                                    nn.BatchNorm2d(32, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                    nn.Conv2d(32, 64, (7, 7), (1, 1),
                                                                                              (3, 3)),
                                                                                    nn.BatchNorm2d(64, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                ),
                                                                            ),
                                                                        ),
                                                                        nn.Sequential(  # Sequential,
                                                                            nn.AvgPool2d((2, 2), (2, 2)),
                                                                            Reduce(
                                                                                lambda x, y, dim=1: torch.cat((x, y),
                                                                                                              dim),
                                                                                # Concat,
                                                                                nn.Sequential(  # Sequential,
                                                                                    nn.Conv2d(256, 64, (1, 1)),
                                                                                    nn.BatchNorm2d(64, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                ),
                                                                                nn.Sequential(  # Sequential,
                                                                                    nn.Conv2d(256, 32, (1, 1)),
                                                                                    nn.BatchNorm2d(32, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                    nn.Conv2d(32, 64, (3, 3), (1, 1),
                                                                                              (1, 1)),
                                                                                    nn.BatchNorm2d(64, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                ),
                                                                                nn.Sequential(  # Sequential,
                                                                                    nn.Conv2d(256, 32, (1, 1)),
                                                                                    nn.BatchNorm2d(32, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                    nn.Conv2d(32, 64, (5, 5), (1, 1),
                                                                                              (2, 2)),
                                                                                    nn.BatchNorm2d(64, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                ),
                                                                                nn.Sequential(  # Sequential,
                                                                                    nn.Conv2d(256, 32, (1, 1)),
                                                                                    nn.BatchNorm2d(32, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                    nn.Conv2d(32, 64, (7, 7), (1, 1),
                                                                                              (3, 3)),
                                                                                    nn.BatchNorm2d(64, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                ),
                                                                            ),
                                                                            Reduce(
                                                                                lambda x, y, dim=1: torch.cat((x, y),
                                                                                                              dim),
                                                                                # Concat,
                                                                                nn.Sequential(  # Sequential,
                                                                                    nn.Conv2d(256, 64, (1, 1)),
                                                                                    nn.BatchNorm2d(64, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                ),
                                                                                nn.Sequential(  # Sequential,
                                                                                    nn.Conv2d(256, 32, (1, 1)),
                                                                                    nn.BatchNorm2d(32, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                    nn.Conv2d(32, 64, (3, 3), (1, 1),
                                                                                              (1, 1)),
                                                                                    nn.BatchNorm2d(64, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                ),
                                                                                nn.Sequential(  # Sequential,
                                                                                    nn.Conv2d(256, 32, (1, 1)),
                                                                                    nn.BatchNorm2d(32, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                    nn.Conv2d(32, 64, (5, 5), (1, 1),
                                                                                              (2, 2)),
                                                                                    nn.BatchNorm2d(64, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                ),
                                                                                nn.Sequential(  # Sequential,
                                                                                    nn.Conv2d(256, 32, (1, 1)),
                                                                                    nn.BatchNorm2d(32, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                    nn.Conv2d(32, 64, (7, 7), (1, 1),
                                                                                              (3, 3)),
                                                                                    nn.BatchNorm2d(64, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                ),
                                                                            ),
                                                                            Reduce(
                                                                                lambda x, y, dim=1: torch.cat((x, y),
                                                                                                              dim),
                                                                                # Concat,
                                                                                nn.Sequential(  # Sequential,
                                                                                    nn.Conv2d(256, 64, (1, 1)),
                                                                                    nn.BatchNorm2d(64, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                ),
                                                                                nn.Sequential(  # Sequential,
                                                                                    nn.Conv2d(256, 32, (1, 1)),
                                                                                    nn.BatchNorm2d(32, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                    nn.Conv2d(32, 64, (3, 3), (1, 1),
                                                                                              (1, 1)),
                                                                                    nn.BatchNorm2d(64, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                ),
                                                                                nn.Sequential(  # Sequential,
                                                                                    nn.Conv2d(256, 32, (1, 1)),
                                                                                    nn.BatchNorm2d(32, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                    nn.Conv2d(32, 64, (5, 5), (1, 1),
                                                                                              (2, 2)),
                                                                                    nn.BatchNorm2d(64, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                ),
                                                                                nn.Sequential(  # Sequential,
                                                                                    nn.Conv2d(256, 32, (1, 1)),
                                                                                    nn.BatchNorm2d(32, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                    nn.Conv2d(32, 64, (7, 7), (1, 1),
                                                                                              (3, 3)),
                                                                                    nn.BatchNorm2d(64, 1e-05, 0.1,
                                                                                                   False),
                                                                                    nn.ReLU(),
                                                                                ),
                                                                            ),
                                                                            nn.UpsamplingNearest2d(scale_factor=2),
                                                                        ),
                                                                        ),
                                                              Reduce(lambda x, y: x + y),  # CAddTable,
                                                          ),
                                                          Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),
                                                                       # Concat,
                                                                       nn.Sequential(  # Sequential,
                                                                           nn.Conv2d(256, 64, (1, 1)),
                                                                           nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                       ),
                                                                       nn.Sequential(  # Sequential,
                                                                           nn.Conv2d(256, 32, (1, 1)),
                                                                           nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                           nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1)),
                                                                           nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                       ),
                                                                       nn.Sequential(  # Sequential,
                                                                           nn.Conv2d(256, 32, (1, 1)),
                                                                           nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                           nn.Conv2d(32, 64, (5, 5), (1, 1), (2, 2)),
                                                                           nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                       ),
                                                                       nn.Sequential(  # Sequential,
                                                                           nn.Conv2d(256, 32, (1, 1)),
                                                                           nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                           nn.Conv2d(32, 64, (7, 7), (1, 1), (3, 3)),
                                                                           nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                       ),
                                                                       ),
                                                          Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),
                                                                       # Concat,
                                                                       nn.Sequential(  # Sequential,
                                                                           nn.Conv2d(256, 64, (1, 1)),
                                                                           nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                       ),
                                                                       nn.Sequential(  # Sequential,
                                                                           nn.Conv2d(256, 64, (1, 1)),
                                                                           nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                           nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
                                                                           nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                       ),
                                                                       nn.Sequential(  # Sequential,
                                                                           nn.Conv2d(256, 64, (1, 1)),
                                                                           nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                           nn.Conv2d(64, 64, (7, 7), (1, 1), (3, 3)),
                                                                           nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                       ),
                                                                       nn.Sequential(  # Sequential,
                                                                           nn.Conv2d(256, 64, (1, 1)),
                                                                           nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                           nn.Conv2d(64, 64, (11, 11), (1, 1), (5, 5)),
                                                                           nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                                           nn.ReLU(),
                                                                       ),
                                                                       ),
                                                          nn.UpsamplingNearest2d(scale_factor=2),
                                                      ),
                                                      ),
                                            Reduce(lambda x, y: x + y),  # CAddTable,
                                        ),
                                        Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),  # Concat,
                                                     nn.Sequential(  # Sequential,
                                                         nn.Conv2d(256, 64, (1, 1)),
                                                         nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                     ),
                                                     nn.Sequential(  # Sequential,
                                                         nn.Conv2d(256, 32, (1, 1)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                         nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1)),
                                                         nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                     ),
                                                     nn.Sequential(  # Sequential,
                                                         nn.Conv2d(256, 32, (1, 1)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                         nn.Conv2d(32, 64, (5, 5), (1, 1), (2, 2)),
                                                         nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                     ),
                                                     nn.Sequential(  # Sequential,
                                                         nn.Conv2d(256, 32, (1, 1)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                         nn.Conv2d(32, 64, (7, 7), (1, 1), (3, 3)),
                                                         nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                     ),
                                                     ),
                                        Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),  # Concat,
                                                     nn.Sequential(  # Sequential,
                                                         nn.Conv2d(256, 32, (1, 1)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                     ),
                                                     nn.Sequential(  # Sequential,
                                                         nn.Conv2d(256, 32, (1, 1)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                         nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                     ),
                                                     nn.Sequential(  # Sequential,
                                                         nn.Conv2d(256, 32, (1, 1)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                         nn.Conv2d(32, 32, (5, 5), (1, 1), (2, 2)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                     ),
                                                     nn.Sequential(  # Sequential,
                                                         nn.Conv2d(256, 32, (1, 1)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                         nn.Conv2d(32, 32, (7, 7), (1, 1), (3, 3)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                     ),
                                                     ),
                                        nn.UpsamplingNearest2d(scale_factor=2),
                                    ),
                                    nn.Sequential(  # Sequential,
                                        Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),  # Concat,
                                                     nn.Sequential(  # Sequential,
                                                         nn.Conv2d(128, 32, (1, 1)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                     ),
                                                     nn.Sequential(  # Sequential,
                                                         nn.Conv2d(128, 32, (1, 1)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                         nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                     ),
                                                     nn.Sequential(  # Sequential,
                                                         nn.Conv2d(128, 32, (1, 1)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                         nn.Conv2d(32, 32, (5, 5), (1, 1), (2, 2)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                     ),
                                                     nn.Sequential(  # Sequential,
                                                         nn.Conv2d(128, 32, (1, 1)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                         nn.Conv2d(32, 32, (7, 7), (1, 1), (3, 3)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                     ),
                                                     ),
                                        Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),  # Concat,
                                                     nn.Sequential(  # Sequential,
                                                         nn.Conv2d(128, 32, (1, 1)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                     ),
                                                     nn.Sequential(  # Sequential,
                                                         nn.Conv2d(128, 64, (1, 1)),
                                                         nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                         nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                     ),
                                                     nn.Sequential(  # Sequential,
                                                         nn.Conv2d(128, 64, (1, 1)),
                                                         nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                         nn.Conv2d(64, 32, (7, 7), (1, 1), (3, 3)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                     ),
                                                     nn.Sequential(  # Sequential,
                                                         nn.Conv2d(128, 64, (1, 1)),
                                                         nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                         nn.Conv2d(64, 32, (11, 11), (1, 1), (5, 5)),
                                                         nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                         nn.ReLU(),
                                                     ),
                                                     ),
                                    ),
                                    ),
                          Reduce(lambda x, y: x + y),  # CAddTable,
                      ),
                      Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),  # Concat,
                                   nn.Sequential(  # Sequential,
                                       nn.Conv2d(128, 32, (1, 1)),
                                       nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                   ),
                                   nn.Sequential(  # Sequential,
                                       nn.Conv2d(128, 64, (1, 1)),
                                       nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                       nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1)),
                                       nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                   ),
                                   nn.Sequential(  # Sequential,
                                       nn.Conv2d(128, 64, (1, 1)),
                                       nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                       nn.Conv2d(64, 32, (5, 5), (1, 1), (2, 2)),
                                       nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                   ),
                                   nn.Sequential(  # Sequential,
                                       nn.Conv2d(128, 64, (1, 1)),
                                       nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                       nn.Conv2d(64, 32, (7, 7), (1, 1), (3, 3)),
                                       nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                   ),
                                   ),
                      Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),  # Concat,
                                   nn.Sequential(  # Sequential,
                                       nn.Conv2d(128, 16, (1, 1)),
                                       nn.BatchNorm2d(16, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                   ),
                                   nn.Sequential(  # Sequential,
                                       nn.Conv2d(128, 32, (1, 1)),
                                       nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                       nn.Conv2d(32, 16, (3, 3), (1, 1), (1, 1)),
                                       nn.BatchNorm2d(16, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                   ),
                                   nn.Sequential(  # Sequential,
                                       nn.Conv2d(128, 32, (1, 1)),
                                       nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                       nn.Conv2d(32, 16, (7, 7), (1, 1), (3, 3)),
                                       nn.BatchNorm2d(16, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                   ),
                                   nn.Sequential(  # Sequential,
                                       nn.Conv2d(128, 32, (1, 1)),
                                       nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                       nn.Conv2d(32, 16, (11, 11), (1, 1), (5, 5)),
                                       nn.BatchNorm2d(16, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                   ),
                                   ),
                      nn.UpsamplingNearest2d(scale_factor=2),
                  ),
                  nn.Sequential(  # Sequential,
                      Reduce(lambda x, y, dim=1: torch.cat((x, y), dim),  # Concat,
                                   nn.Sequential(  # Sequential,
                                       nn.Conv2d(128, 16, (1, 1)),
                                       nn.BatchNorm2d(16, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                   ),
                                   nn.Sequential(  # Sequential,
                                       nn.Conv2d(128, 64, (1, 1)),
                                       nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                       nn.Conv2d(64, 16, (3, 3), (1, 1), (1, 1)),
                                       nn.BatchNorm2d(16, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                   ),
                                   nn.Sequential(  # Sequential,
                                       nn.Conv2d(128, 64, (1, 1)),
                                       nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                       nn.Conv2d(64, 16, (7, 7), (1, 1), (3, 3)),
                                       nn.BatchNorm2d(16, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                   ),
                                   nn.Sequential(  # Sequential,
                                       nn.Conv2d(128, 64, (1, 1)),
                                       nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                       nn.Conv2d(64, 16, (11, 11), (1, 1), (5, 5)),
                                       nn.BatchNorm2d(16, 1e-05, 0.1, False),
                                       nn.ReLU(),
                                   ),
                                   ),
                  ),
                  ),
        Reduce(lambda x, y: x + y),  # CAddTable,
    ),
    nn.Conv2d(64, 1, (3, 3), (1, 1), (1, 1)),
)
