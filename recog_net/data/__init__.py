import torch

from .dataset import Dataset
from .dataset2 import Dataset2


def generate_loader(phase, opt):
    if "diff" in opt.data_root:
        dataset = Dataset(phase=phase, data_root=opt.data_root, size=opt.patchsize, level=opt.level)
    else:
        dataset = Dataset2(phase=phase, data_root=opt.data_root, size=opt.patchsize, level=opt.level)
    kwargs = {
            "batch_size": opt.batchsize if phase == 'train' else 1,
            "num_workers": opt.num_workers if phase == 'train' else 0,
            "shuffle": phase == 'train',
            "drop_last": phase == 'train'
            }

    return torch.utils.data.DataLoader(dataset, **kwargs)


