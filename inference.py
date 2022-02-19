import os
import glob
import importlib
import tqdm
import numpy as np
import skimage.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import generate_loader
import utils
from option import get_option


def im2tensor(im):
    np_t = np.ascontiguousarray(im.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_t).float()
    return tensor


def load(net, path):
    state_dict = torch.load(
        path, map_location=lambda storage, loc: storage)

    #         if self.opt.strict_load:
    #             self.net.load_state_dict(state_dict)
    #         return

    own_state = net.state_dict()
    for name, param in state_dict.items():
        if name in own_state:
            if isinstance(param, nn.Parameter):
                param = param.data

            try:
                own_state[name].copy_(param)
            except Exception:
                # head and tail modules can be different
                if name.find("head") == -1 and name.find("tail") == -1:
                    raise RuntimeError(
                        "While copying the parameter named {}, "
                        "whose dimensions in the model are {} and "
                        "whose dimensions in the checkpoint are {}."
                            .format(name, own_state[name].size(), param.size())
                    )
        else:
            raise RuntimeError(
                "Missing key {} in model's state_dict".format(name)
            )


@torch.no_grad()
def main(opt):
    os.makedirs(opt.save_root, exist_ok=True)
    dev = torch.device("cuda:{}".format(opt.gpu))
    module = importlib.import_module("model.{}".format(opt.model.lower()))
    net = module.Net(opt).to(dev)
    net = nn.DataParallel(net, device_ids=opt.workers_idx).to(dev)

    load(net, opt.pretrain)
    test_loader = generate_loader("test", opt)
    length = len(test_loader)
    psnr = 0
    for i, inputs in tqdm.tqdm(enumerate(test_loader)):
        clean_im = inputs[1].squeeze(0)
        filename = str(inputs[3])[2:-3]

        # if our memory is enough
        input_im = inputs[0]
        mask = inputs[2]
        restore_im = net(input_im, mask).squeeze(0).clamp(0, 255).round().cpu().byte().permute(1, 2, 0).numpy()

        clean_im = clean_im.cpu().byte().permute(1, 2, 0).numpy().astype(np.uint8)
        restore_im = restore_im.astype(np.uint8)

        save_path = os.path.join(opt.save_root, filename)
        io.imsave(save_path, restore_im)
        P = utils.calculate_psnr(clean_im, restore_im)
        if P > 100:
            length -= 1
        else:
            psnr += utils.calculate_psnr(clean_im, restore_im)
    print("psnr of {} : ".format(opt.level), psnr / length)
    print(length)


if __name__ == "__main__":
    opt = get_option()
    main(opt)