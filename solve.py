import os
import time
import torch
import torch.nn as nn
import numpy as np
import skimage.io as io
from PIL import Image
import pandas as pd

from data import generate_loader
import utils


class Solve():
    def __init__(self, module, opt):
        self.opt = opt
        if not opt.gpu:
            self.dev = torch.device("cpu")
        self.dev = torch.device("cuda:{}".format(opt.gpu))

        self.net = module.Net(opt)
        # for using multi-gpu
        self.net = nn.DataParallel(self.net, device_ids=opt.workers_idx).to(self.dev)
        print("# Params : ", sum(map(lambda x : x.numel(), self.net.parameters())))

        if opt.pretrain:
            self.load(opt.pretrain)

        if opt.loss.lower() == 'mse':
            self.loss_fn = nn.MSELoss().to(self.dev)
        elif opt.loss.lower() == 'l1':
            self.loss_fn = nn.L1Loss().to(self.dev)
        else:
            raise ValueError(
                    "ValueError - wrong type of loss function(need MSE or L1)")

        if not opt.test_only:
            opt.level = 'train'
            self.train_loader = generate_loader("train", opt)
        self.val_loader = generate_loader("val", opt)

        print("Dataset Preparation")
        self.optim = torch.optim.Adam(
                params=self.net.parameters(),
                lr=opt.lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=opt.weight_decay)

        # using scheduler
        if not opt.test_only:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer=self.optim,
                    milestones=[int(len(self.train_loader)*int(d)) for d in opt.decay.split("-")],
                    gamma=0.5) # milestones at 200th and 500th epochs

        self.t1, self.t2 = None, None
        self.best_psnr, self.best_steps = 0, 0

    def fit(self):
        opt = self.opt
        self.t1 = time.time()
        print("let's start training")
        for step in range(opt.max_steps):
            try:
                inputs = next(iters)
            except (UnboundLocalError, StopIteration):
                iters = iter(self.train_loader)
                inputs = next(iters)

            noisy_im = inputs[0].to(self.dev)
            clean_im = inputs[1].to(self.dev)
            restore_im = self.net(noisy_im)

            loss = self.loss_fn(restore_im, clean_im)

            if not torch.isfinite(loss):
                print('Warning: Losses are Nan, negative infinity, or infinity. Stop Training')
                exit(1)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.scheduler.step()

            if (step + 1) % opt.eval_steps == 0:
                self.summary_and_save(step)

    def summary_and_save(self, step):
        step, max_steps = (step+1)//self.opt.eval_steps, self.opt.max_steps//self.opt.eval_steps
        psnr = self.evaluate(step)
        self.t2 = time.time()

        if aver >= self.best_psnr:
            self.best_psnr, self.best_step = psnr, step
            self.save(step)

        curr_lr = self.scheduler.get_last_lr()[0]
        eta = (self.t2-self.t1) * (max_steps-step) / 3600

        print("[{}K/{}K] P {:.2f}(Best: {:.2f}  @ {}K step) LR: {}, ETA: {:.1f} hours"
                .format(step, max_steps, aver, self.best_psnr, self.best_step, curr_lr, eta))

        self.t1 = time.time()

    @torch.no_grad()
    def evaluate(self):
        opt = self.opt
        self.net.eval()
        if opt.save_result:
            save_root = opt.save_root
            os.makedirs(save_root, exist_ok=True)

        psnr = 0
        count = 0
        for i, inputs in enumerate(valid_loader):
            input_im = inputs[0].to(self.dev)
            clean_im = inputs[1].squeeze(0).cpu().byte().permute(1, 2, 0).numpy().astype(np.uint8)
            filename = str(inputs[2])[2:-3]
            restore_im = self.net(input_im).squeeze(0).clamp(0, 255).round().cpu().byte().permute(1, 2, 0).numpy().astype(np.uint8)

            if opt.save_result:
                save_path = os.path.join(save_root, "{}_{}".format(a[b], filename))
                io.imsave(save_path, restore_im)
            if utils.calculate_psnr(clean_im, restore_im) < 40:
                psnr += utils.calculate_psnr(clean_im, restore_im)
                count +=1
            if i == (opt.num_valimages - 1):
                break
        psnr = psnr/count
        self.net.train()

        return psnr

    def load(self, path):
        state_dict = torch.load(
            path, map_location=lambda storage, loc: storage)
        own_state = self.net.state_dict()
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

    def save(self, step):
        os.makedirs(self.opt.ckpt_root, exist_ok=True)
        save_path = os.path.join(self.opt.ckpt_root, str(step)+".pt")
        torch.save(self.net.state_dict(), save_path)