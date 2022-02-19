import torch

from .dataset import Dataset


def generate_loader(phase, opt):
    dataset = Dataset(phase=phase, data_root=opt.data_root, size=opt.patchsize, level=opt.level)

    kwargs = {
            "batch_size": opt.batchsize if phase == 'train' else 1,
            "num_workers": opt.num_workers if phase == 'train' else 0,
            "shuffle": phase == 'train',
            "drop_last": phase == 'train'
            }

    return torch.utils.data.DataLoader(dataset, **kwargs)
