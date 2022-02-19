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
from tensorboardX import SummaryWriter
import pdb


class Solver():
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
        self.val_loader = []
        for level in ['mild', 'severe', 'moderate']:
            opt.level = level
            self.val_loader.append(generate_loader('val', opt))
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

        self.writer = SummaryWriter(log_dir = './log/result/{}'.format(opt.model))

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
        mild, severe, moderate = self.evaluate(step)
        aver = (mild + severe + moderate) / 3
        self.t2 = time.time()

        if aver >= self.best_psnr:
            self.best_psnr, self.best_step = aver, step
            self.save(step)

        curr_lr = self.scheduler.get_last_lr()[0]
        eta = (self.t2-self.t1) * (max_steps-step) / 3600

        print("[{}K/{}K] P {:.2f}(Best: {:.2f}  @ {}K step) LR: {}, ETA: {:.1f} hours"
                .format(step, max_steps, aver, self.best_psnr, self.best_step, curr_lr, eta))

        self.t1 = time.time()
        self.writer.add_scalar('valid/psnr', aver, step)

    @torch.no_grad()
    def evaluate(self, step):
        opt = self.opt
        self.net.eval()
        if opt.save_result:
            save_root = opt.save_root
            os.makedirs(save_root, exist_ok=True)

        P = []
        a = ['mild', 'severe', 'moderate']
        b = 0
        for valid_loader in self.val_loader:
            psnr = 0
            count = 0
            for i, inputs in enumerate(valid_loader):
                input_im = inputs[0].to(self.dev)
                clean_im = inputs[1].squeeze(0)
                filename = str(inputs[2])[2:-3]
                # if our memory is enough
                restore_im = self.net(input_im)

                restore_im = restore_im.squeeze(0).clamp(0, 255).round().cpu().byte().permute(1, 2, 0).numpy().astype(np.uint8)
                clean_im = clean_im.cpu().byte().permute(1, 2, 0).numpy().astype(np.uint8)
                if opt.save_result:
                    save_path = os.path.join(save_root, "{}_{}".format(a[b], filename))
                    io.imsave(save_path, restore_im)
                if utils.calculate_psnr(clean_im, restore_im) < 40:
                    psnr += utils.calculate_psnr(clean_im, restore_im)
                    count +=1
                if i == (opt.num_valimages - 1):
                    break
            P.append(psnr/(count))
#             M.append([mask_acc/(opt.num_valimages), mask_pixel_acc/opt.num_valimages])

            b += 1
        self.net.train()

        return P[0], P[1], P[2]
#         return P[0], P[1], P[2], M[0], M[1], M[2]

    def load(self, path):
        state_dict = torch.load(
            path, map_location=lambda storage, loc: storage)

#         if self.opt.strict_load:
#             self.net.load_state_dict(state_dict)
#         return

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
#         save_path2 = os.path.join(self.opt.ckpt_root, str(step)+"sub.pt")
        torch.save(self.net.state_dict(), save_path)
#         torch.save(self.subnet.state_dict(), save_path2)
########################################################################################################################
########################################################################################################################


# class Solver():
#     def __init__(self, module, opt):
#         self.opt = opt
#         if not opt.gpu:
#             self.dev = torch.device("cpu")
#         self.dev = torch.device("cuda:{}".format(opt.gpu))
#
#         self.net = module.Net(opt)
#         # for using multi-gpu
#         self.net = nn.DataParallel(self.net, device_ids=opt.workers_idx).to(self.dev)
#         print("# Params : ", sum(map(lambda x : x.numel(), self.net.parameters())))
#
#         if opt.pretrain:
#             self.load(opt.pretrain)
#
#         if opt.loss.lower() == 'mse':
#             self.loss_fn = nn.MSELoss().to(self.dev)
#         elif opt.loss.lower() == 'l1':
#             self.loss_fn = nn.L1Loss().to(self.dev)
#         else:
#             raise ValueError(
#                     "ValueError - wrong type of loss function(need MSE or L1)")
#
#         if not opt.test_only:
#             opt.level = 'train'
#             self.train_loader = generate_loader("train", opt)
#         self.val_loader = []
#         for level in ['mild', 'severe', 'moderate']:
#             opt.level = level
#             self.val_loader.append(generate_loader('val', opt))
# #         self.val_loader = generate_loader("val", opt)
#         print("Dataset Preparation")
#         self.optim = torch.optim.Adam(
#                 params=self.net.parameters(),
#                 lr=opt.lr,
#                 betas=(0.9, 0.999),
#                 eps=1e-8,
#                 weight_decay=opt.weight_decay)
#
#         # using scheduler
#         if not opt.test_only:
#             self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
#                     optimizer=self.optim,
#                     milestones=[int(len(self.train_loader)*int(d)) for d in opt.decay.split("-")],
#                     gamma=0.5) # milestones at 200th and 500th epochs
#
#         self.t1, self.t2 = None, None
#         self.best_psnr, self.best_steps = 0, 0
#
#         self.writer = SummaryWriter(log_dir = './log/result/{}_recog'.format(opt.model))
#
#     def fit(self):
#         opt = self.opt
#         self.t1 = time.time()
#         print("let's start training")
#         for step in range(opt.max_steps):
#             try:
#                 inputs = next(iters)
#             except (UnboundLocalError, StopIteration):
#                 iters = iter(self.train_loader)
#                 inputs = next(iters)
#
#             noisy_im = inputs[0].to(self.dev)
#             clean_im = inputs[1].to(self.dev)
#             mask = inputs[2].to(self.dev)
#             restore_im = self.net(noisy_im, mask)
# #             restore_im = self.net(noisy_im)
#
#             loss = self.loss_fn(restore_im, clean_im)
#
#             if not torch.isfinite(loss):
#                 print('Warning: Losses are Nan, negative infinity, or infinity. Stop Training')
#                 exit(1)
#             self.optim.zero_grad()
#             loss.backward()
#             self.optim.step()
#             self.scheduler.step()
#
#             if (step + 1) % opt.eval_steps == 0:
#                 self.summary_and_save(step)
#
#
#     def summary_and_save(self, step):
#         step, max_steps = (step+1)//self.opt.eval_steps, self.opt.max_steps//self.opt.eval_steps
#         mild, severe, moderate = self.evaluate(step)
#         aver = (mild + severe + moderate) / 3
#         self.t2 = time.time()
#
#         if aver >= self.best_psnr:
#             self.best_psnr, self.best_step = aver, step
#             self.save(step)
#
#         curr_lr = self.scheduler.get_last_lr()[0]
#         eta = (self.t2-self.t1) * (max_steps-step) / 3600
#
#         print("[{}K/{}K] P {:.2f}(Best: {:.2f}  @ {}K step) LR: {}, ETA: {:.1f} hours"
#                 .format(step, max_steps, aver, self.best_psnr, self.best_step, curr_lr, eta))
#
#         self.t1 = time.time()
#         self.writer.add_scalar('valid/psnr', aver, step)
#
#
#     @torch.no_grad()
#     def evaluate(self, step):
#         opt = self.opt
#         self.net.eval()
#         if opt.save_result:
#             save_root = opt.save_root
#             os.makedirs(save_root, exist_ok=True)
#
# #         P = 0
# #         psnr = 0
# #         count = 0
# #         for i, inputs in enumerate(self.val_loader):
# #             input_im = inputs[0].to(self.dev)
# #             clean_im = inputs[1].squeeze(0)
# # #             filename = str(inputs[3])[2:-3]
# #             filename = str(inputs[2])[2:-3]
# #             # if our memory is enough
# # #             mask = inputs[2].to(self.dev)
# # #             restore_im = self.net(input_im, mask)
# #             restore_im = self.net(input_im)
# #
# #             restore_im = restore_im.squeeze(0).clamp(0, 255).round().cpu().byte().permute(1, 2, 0).numpy().astype(np.uint8)
# #             clean_im = clean_im.cpu().byte().permute(1, 2, 0).numpy().astype(np.uint8)
# #             if opt.save_result:
# #                 save_path = os.path.join(save_root, "{}_{}".format(a[b], filename))
# # #                 save_path = os.path.join(save_root, "{}".format(filename))
# #                 io.imsave(save_path, restore_im)
# #             if utils.calculate_psnr(clean_im, restore_im) < 50:
# #                 psnr += utils.calculate_psnr(clean_im, restore_im)
# #                 count +=1
# #             if i == (opt.num_valimages - 1):
# #                 break
# #         P = psnr/count
# #         self.net.train()
# #
# #         return P
#         P = []
#         a = ['mild', 'severe', 'moderate']
#         b = 0
#         for valid_loader in self.val_loader:
#             psnr = 0
#             count = 0
#             for i, inputs in enumerate(valid_loader):
#                 input_im = inputs[0].to(self.dev)
#                 clean_im = inputs[1].squeeze(0)
#                 filename = str(inputs[3])[2:-3]
# #                 filename = str(inputs[2])[2:-3]
#                 # if our memory is enough
#                 mask = inputs[2].to(self.dev)
#                 restore_im = self.net(input_im, mask)
# #                 restore_im = self.net(input_im)
#
#                 restore_im = restore_im.squeeze(0).clamp(0, 255).round().cpu().byte().permute(1, 2, 0).numpy().astype(np.uint8)
#                 clean_im = clean_im.cpu().byte().permute(1, 2, 0).numpy().astype(np.uint8)
#                 if opt.save_result:
#                     save_path = os.path.join(save_root, "{}_{}".format(a[b], filename))
#                     io.imsave(save_path, restore_im)
#                 if utils.calculate_psnr(clean_im, restore_im) < 40:
#                     psnr += utils.calculate_psnr(clean_im, restore_im)
#                     count +=1
#                 if i == (opt.num_valimages - 1):
#                     break
#             P.append(psnr/(count))
# #             M.append([mask_acc/(opt.num_valimages), mask_pixel_acc/opt.num_valimages])
#
#             b += 1
#         self.net.train()
#
#         return P[0], P[1], P[2]
# #         return P[0], P[1], P[2], M[0], M[1], M[2]
#
#     def load(self, path):
#         state_dict = torch.load(
#             path, map_location=lambda storage, loc: storage)
#
# #         if self.opt.strict_load:
# #             self.net.load_state_dict(state_dict)
# #         return
#
#         own_state = self.net.state_dict()
#         for name, param in state_dict.items():
#             if name in own_state:
#                 if isinstance(param, nn.Parameter):
#                     param = param.data
#
#                 try:
#                     own_state[name].copy_(param)
#                 except Exception:
#                     # head and tail modules can be different
#                     if name.find("head") == -1 and name.find("tail") == -1:
#                         raise RuntimeError(
#                             "While copying the parameter named {}, "
#                             "whose dimensions in the model are {} and "
#                             "whose dimensions in the checkpoint are {}."
#                             .format(name, own_state[name].size(), param.size())
#                         )
#             else:
#                 raise RuntimeError(
#                     "Missing key {} in model's state_dict".format(name)
#                 )
#
#     def save(self, step):
#         os.makedirs(self.opt.ckpt_root, exist_ok=True)
#         save_path = os.path.join(self.opt.ckpt_root, str(step)+".pt")
# #         save_path2 = os.path.join(self.opt.ckpt_root, str(step)+"sub.pt")
#         torch.save(self.net.state_dict(), save_path)
# #         torch.save(self.subnet.state_dict(), save_path2)