# stdlib
import os
import sys
from functools import reduce

# external
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel, DataParallel

# local
from geopose.model.nice import build_nice_model
from geopose.model.ugly import build_ugly_model


class Hourglass(nn.Module):
    def __init__(self, arch='ugly', weights=None, gpus=None, parallel='dp', device='cuda:0'):
        super().__init__()
        if arch == 'ugly':
            self.model = build_ugly_model()
        else:  # elif arch == 'nice':  # takhle zaroven i jako else-blok
            self.model = build_nice_model()


        if gpus is None:
            gpus = [0]  # idea: test if gpu available?

        if parallel == 'dp':
            # use to load generalization weights
            self.model = torch.nn.parallel.DataParallel(self.model, device_ids=gpus)
        elif parallel == 'ddp':
            # raise NotImplementedError('DistributedDataParallel not supported yet')
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=gpus, output_device=gpus[0])

        if weights is not None:
            if weights == 'generalization':
                weights = 'geopose/checkpoints/best_generalization_net_G.pth'
            try:
                self.model.load_state_dict(torch.load(weights))
            except RuntimeError as err:
                print('Failed to load weights\n', err, file=sys.stderr)

        self.model = self.model.to(device)

    def switch_to_train(self):
        self.model.train()

    def switch_to_eval(self):
        self.model.eval()


