import os
import time
import torch
import torch.nn as nn
import numpy as np
import skimage.io as io
from PIL import Image
import pandas as pd
import model.unet_aspp as unet_aspp
from data import generate_loader
import utils


class Solve():
    def __init__(self, opt):
        self.opt = opt
        if not opt.gpu:
            self.dev = torch.device("cpu")
        self.dev = torch.device("cuda:{}".format(opt.gpu))
        self.net = unet_aspp.Net()
        # for using multi-gpu
        self.net = nn.DataParallel(self.net, device_ids=opt.workers_idx).to(self.dev)
        print("# Params : ", sum(map(lambda x: x.numel(), self.net.parameters())))

        if opt.pretrain:
            self.load(opt.pretrain)

        self.loss_blur = nn.CrossEntropyLoss().to(self.dev)
        self.loss_noise = nn.CrossEntropyLoss().to(self.dev)
        self.loss_jpeg = nn.CrossEntropyLoss().to(self.dev)
        if not opt.test_only:
            opt.level = 'train'
            self.train_loader = generate_loader("train", opt)
        self.val_loader = generate_loader('val', opt)

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
                milestones=[int(len(self.train_loader) * int(d)) for d in opt.decay.split("-")],
                gamma=0.5)  # milestones at 200th and 500th epochs

        self.t1, self.t2 = None, None
        self.best_pacc, self.best_steps = 0, 0
        self.loss_sum = 0

    def fit(self):
        opt = self.opt
        print("let's start training")
        self.net.train()
        self.t1 = time.time()
        for step in range(opt.max_steps):
            try:
                inputs = next(iters)
            except (UnboundLocalError, StopIteration):
                iters = iter(self.train_loader)
                inputs = next(iters)
            noisy_im = inputs[0].to(self.dev)
            blur_label, noise_label, jpeg_label = inputs[1]
            blur, noise, jpeg = self.net(noisy_im)
            loss_blur = self.loss_blur(blur, blur_label.squeeze(1).to(self.dev, dtype=torch.long))
            loss_noise = self.loss_noise(noise, noise_label.squeeze(1).to(self.dev, dtype=torch.long))
            loss_jpeg = self.loss_jpeg(jpeg, jpeg_label.squeeze(1).to(self.dev, dtype=torch.long))
            loss = loss_blur + loss_noise + loss_jpeg

            # loss = self.loss(pred_mask, mask)
            self.loss_sum += loss
            if not torch.isfinite(loss):
                print('Warning: Losses are Nan, negative infinity, or infinity. Stop Training')
                exit(1)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.scheduler.step()
            if (step + 1) % opt.eval_steps == 0:
                print("Loss : {}".format(self.loss_sum / opt.eval_steps))
                self.summary_and_save(step)
                self.loss_sum = 0

    def summary_and_save(self, step):
        step, max_steps = (step + 1) // self.opt.eval_steps, self.opt.max_steps // self.opt.eval_steps
        P = self.evaluate(step)
        self.t2 = time.time()

        if (P[6]) >= self.best_pacc:
            self.best_pacc, self.best_steps = P[6], step
            self.save(step)

        curr_lr = self.scheduler.get_last_lr()[0]
        eta = (self.t2 - self.t1) * (max_steps - step) / 3600
        print("[{}K/{}K] mIoU blur:{:.4f} noise:{:.4f} jpeg:{:.4f} LR: {}, ETA: {:.1f} hours "
              .format(step, max_steps, P[0], P[1], P[2], curr_lr, eta))
        print("Accuracy: {:.2f} blur:{:.2f} noise:{:.2f} jpeg:{:.2f} (Best: {:.4f}  @ {}K step) "
              .format(P[6], P[3], P[4], P[5], self.best_pacc, self.best_steps))

        self.t1 = time.time()

    @torch.no_grad()
    def evaluate(self):
        opt = self.opt
        self.net.eval()

        miou_blur, miou_noise, miou_jpeg = 0, 0, 0
        acc_blur, acc_noise, acc_jpeg, p_acc = 0, 0, 0, 0
        for i, inputs in enumerate(val_loader):
            input_im = inputs[0].to(self.dev)
            blur_label, noise_label, jpeg_label = inputs[1]
            blur_label = blur_label.cpu().numpy()[0, 0, :, :]
            noise_label = noise_label.cpu().numpy()[0, 0, :, :]
            jpeg_label = jpeg_label.cpu().numpy()[0, 0, :, :]

            blur, noise, jpeg = self.net(input_im)
            blur = np.argmax(blur.cpu().numpy()[0, :, :, :], axis=0)
            noise = np.argmax(noise.cpu().numpy()[0, :, :, :], axis=0)
            jpeg = np.argmax(jpeg.cpu().numpy()[0, :, :, :], axis=0)

            _miou_blur, ious_blur = utils.iou(blur, blur_label)
            _miou_noise, ious_noise = utils.iou(noise, noise_label)
            _miou_jpeg, ious_jpeg = utils.iou(jpeg, jpeg_label)

            miou_blur += _miou_blur
            miou_noise += _miou_noise
            miou_jpeg += _miou_jpeg

            size = blur.shape[0] * blur.shape[1]
            acc_blur += (blur == blur_label).sum() / size * 100
            acc_noise += (noise == noise_label).sum() / size * 100
            acc_jpeg += (jpeg == jpeg_label).sum() / size * 100
            p_acc += ((blur == blur_label) & (noise == noise_label) & (jpeg == jpeg_label)).sum() / size * 100

            if i == (opt.num_valimages - 1):
                break
        P = [miou_blur / opt.num_valimages, miou_noise / opt.num_valimages,
             miou_jpeg / opt.num_valimages, acc_blur / opt.num_valimages,
             acc_noise / opt.num_valimages, acc_jpeg / opt.num_valimages,
             p_acc / opt.num_valimages]
        self.net.train()

        return P

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
        save_path = os.path.join(self.opt.ckpt_root, str(step) + ".pt")
        torch.save(self.net.state_dict(), save_path)
