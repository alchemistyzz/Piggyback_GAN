'''
Description: 
Author: Zhang yizhen
Date: 2023-03-07 18:35:29
LastEditors: Zhang yizhen
LastEditTime: 2023-03-22 19:02:38
FilePath: /Piggyback_GAN/model/PiggybackGAN.py

Copyright (c) 2023 by yizhen_coder@outlook.com, All Rights Reserved. 
'''
from difflib import restore
from inspect import getargs
from json import load
import os
from math import ceil
from typing import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import math

import model.networks as networks
from model.networks import MPRNet, MPRNetLoss,UnetGenerator
from model.networks import define_G,define_D
from model.faig import faig
import utils
from utils import  ImageBuffer as ImageBuffer
import itertools

class PiggybackGAN(nn.Module):
    def __init__(self, opt, task_num=1,device="gpu"):
        super(PiggybackGAN, self).__init__()
        self.opt = opt
        self.task_num = task_num
        self.is_train = opt.is_train
        self.model_names = ["netG_A","netG_B","netD_A","netD_B"]

        self.device = device
        self.netG_A_filter_list = []
        self.netG_B_filter_list = []
        self.netG_A_weights = []
        self.netG_B_weights = []
        self.netG_A = define_G(self.opt.input_nc, self.opt.output_nc, self.opt.ngf, self.opt.netG, self.opt.norm,
                                        self.opt.dropout, self.opt.init_type, self.opt.init_gain, self.task_num, self.netG_A_filter_list)
        self.netG_B = define_G(self.opt.input_nc, self.opt.output_nc, self.opt.ngf, self.opt.netG, self.opt.norm,
                                        self.opt.dropout, self.opt.init_type, self.opt.init_gain, self.task_num, self.netG_B_filter_list)

        if opt.is_train:
            self.netD_A = define_D(self.opt.input_nc, self.opt.ndf, self.opt.netD, self.opt.norm, self.opt.init_type, self.opt.init_gain)
            self.netD_B = define_D(self.opt.input_nc, self.opt.ndf, self.opt.netD, self.opt.norm, self.opt.init_type, self.opt.init_gain)

            self.fake_A_pool = ImageBuffer(self.opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImageBuffer(self.opt.pool_size)  # create image buffer to store previously generated images

            self.criterionGAN = networks.GANLoss(self.opt.gan_mode)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=self.opt.train.lr_init[self.task_num-1], betas=(0.9, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=self.opt.train.lr_init[self.task_num-1], betas=(0.9, 0.999))
            
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
            self.scheduler_G = networks.get_scheduler(self.optimizer_G, self.opt, self.task_num-1, self.opt.train.num_epochs[self.task_num-1])
            self.scheduler_D = networks.get_scheduler(self.optimizer_D, self.opt, self.task_num-1, self.opt.train.num_epochs[self.task_num-1])
 


    def set_device(self, device='cpu'):
        if device == 'cpu':
            pass
        elif device == 'gpu':
            
            self.cuda()
            self = nn.DataParallel(self)
            

    def update_learning_rate(self):
        
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        self.scheduler_G.step()
        self.scheduler_D.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def update_task(self):
        # self.protect_filter()
        # self.save_model(model_name='target')
        
        # self._baseline = self._target
        # self._target = None
        self.task_num += 1
        
        self.netG_A = define_G(self.opt.input_nc, self.opt.output_nc, self.opt.ngf, self.opt.netG, self.opt.norm,
                                        self.opt.dropout, self.opt.init_type, self.opt.init_gain, self.task_num, self.netG_A_filter_list)
        self.netG_B = define_G(self.opt.input_nc, self.opt.output_nc, self.opt.ngf, self.opt.netG, self.opt.norm,
                                        self.opt.dropout, self.opt.init_type, self.opt.init_gain, self.task_num, self.netG_B_filter_list)
        self.netD_A = define_D(self.opt.input_nc, self.opt.ndf, self.opt.netD, self.opt.norm, self.opt.init_type, self.opt.init_gain)
        self.netD_B = define_D(self.opt.input_nc, self.opt.ndf, self.opt.netD, self.opt.norm, self.opt.init_type, self.opt.init_gain)
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=self.opt.train.lr_init[self.task_num-1], betas=(0.9, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=self.opt.train.lr_init[self.task_num-1], betas=(0.9, 0.999))
            
        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
        self.schedulers = [networks.get_scheduler(optimizer, self.opt, self.task_num-1, self.opt.train.num_epochs[self.task_num-1]) for optimizer in self.optimizers]
   
        


    def set_input(self, input):
        # AtoB = self.opt.direction == 'AtoB'
        # self.real_A = input['A' if AtoB else 'B'].to(self.device)
        # self.real_B = input['B' if AtoB else 'A'].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']
        def pad_img(img):
            img_multiple_of = 8
            h,w = img.shape[2:]
            H,W = ((h+img_multiple_of)//img_multiple_of)*img_multiple_of, ((w+img_multiple_of)//img_multiple_of)*img_multiple_of
            padh = H-h if h%img_multiple_of!=0 else 0
            padw = W-w if w%img_multiple_of!=0 else 0
            img = F.pad(img, (0,padw,0,padh), 'reflect')
            return img
        #这里的real_A代表的是对应的退化图像,real_B代表的是对应的干净图像
        self.real_A = pad_img(input[0].cuda())
        self.real_B = pad_img(input[1].cuda())
        self.file_name = input[2]
        h,w = min(self.real_A.shape[-2], self.real_B.shape[-2]), min(self.real_A.shape[-1], self.real_B.shape[-1])
        self.real_A = TF.crop(self.real_A, 0, 0, h, w)
        self.real_B = TF.crop(self.real_B, 0, 0, h, w)

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake, retain_graph=False):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Realscheduler
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward(retain_graph=retain_graph)
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A, retain_graph=False)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward(retain_graph=False)


    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad



    # def forward(self):
    #     # self.real_A = self.restore_net(self.degraded_image, optimize=optimize)
    #     self.real_A,_ = self.restore_net(self.degraded_image)
    #     # if optimize:
    #     #     self.real_A = [torch.clamp(img, 0, 1) for img in self.real_A]
    #     # else:
    #     self.real_A = torch.clamp(self.real_A, 0, 1)

    def inference(self):
        with torch.no_grad():
            self.forward()

        return self.fake_B

    def get_current_losses(self):
        loss_dict = OrderedDict()
        ################################
        loss_dict['GAN_Loss_G'] = self.loss_G
        # loss_dict['GAN_Loss_D'] = self.loss_D
        ################################
        return loss_dict
        
    def get_current_psnr(self):
        if isinstance(self.fake_B, list):
            pred_images = self.fake_B[0]
        else:
            pred_images = self.fake_B

        gt_images = self.real_B

        total_psnr = 0
        for pred_image, gt_image in zip(pred_images, gt_images):
            
            pred_img = (pred_image.permute(1,2,0).cpu().detach().numpy()*255).astype(np.uint8)
            gt_img = (gt_image.permute(1,2,0).cpu().detach().numpy()*255).astype(np.uint8)
            total_psnr += psnr(gt_img, pred_img)

        return total_psnr

    def get_current_ssim(self):
        if isinstance(self.fake_B, list):
            pred_images = self.fake_B[0]
        else:
            pred_images = self.fake_B

        gt_images = self.real_B

        total_ssim = 0
        for pred_image, gt_image in zip(pred_images, gt_images):
            pred_img = (pred_image.permute(1,2,0).cpu().detach().numpy()*255).astype(np.uint8)
            gt_img = (gt_image.permute(1,2,0).cpu().detach().numpy()*255).astype(np.uint8)
            total_ssim += ssim(pred_img, gt_img, multichannel=True)

        return total_ssim

    def load_model(self, mode='epoch', task_id=0, epoch=None):
        if mode == 'epoch':
            folder = f'task_{task_id}_epoch_{epoch}_model'
        elif mode == 'latest':
            folder = 'model_latest'
        elif mode == 'best':
            folder = 'model_best'
        else:
            raise NotImplementedError('Model load mode error!')
        load_path = os.path.join(self.opt.checkpoints.save_model_dir, folder)
        assert os.path.exists(load_path), 'Load file not exists!'
        
        load_dict = torch.load(load_path, map_location='cpu')
        
        self.best_epoch = load_dict['best_epoch']
        # self.netG_A = define_G(self.opt.input_nc, self.opt.output_nc, self.opt.ngf, self.opt.netG, self.opt.norm,
        #                                 self.opt.dropout, self.opt.init_type, self.opt.init_gain, self.opt.checkpoints.resume_task_id+1, self.netG_A_filter_list)
        # self.netG_B = define_G(self.opt.input_nc, self.opt.output_nc, self.opt.ngf, self.opt.netG, self.opt.norm,
        #                                 self.opt.dropout, self.opt.init_type, self.opt.init_gain, self.opt.checkpoints.resume_task_id+1, self.netG_B_filter_list)
        # self.netD_A = define_D(self.opt.input_nc, self.opt.ndf, self.opt.netD, self.opt.norm, self.opt.init_type, self.opt.init_gain)
        # self.netD_B = define_D(self.opt.input_nc, self.opt.ndf, self.opt.netD, self.opt.norm, self.opt.init_type, self.opt.init_gain)
        self.netG_A.load_state_dict(load_dict['netG_A'])
        self.netG_B.load_state_dict(load_dict['netG_B'])
        self.netD_A.load_state_dict(load_dict['netD_A'])
        self.netD_B.load_state_dict(load_dict['netD_B'])
        self.fake_A_pool = load_dict['fake_A_pool']
        self.fake_B_pool = load_dict['fake_B_pool']


        for _ in range(load_dict['task_num']-1):
            self.update_task()
       
        # if 'baseline' in load_dict:
        #     self._baseline = load_dict['baseline']
        # if 'target' in load_dict:
        #     self._target = load['target']
        total_epoch_G = load_dict['scheduler_G']['last_epoch'] + load_dict['scheduler_G']['after_scheduler'].last_epoch
        for _ in range(total_epoch_G):
            self.scheduler_G.step()
        total_epoch_D = load_dict['scheduler_D']['last_epoch'] + load_dict['scheduler_D']['after_scheduler'].last_epoch
        for _ in range(total_epoch_D):
            self.scheduler_D.step()
      

    def _save_model(self, mode='epoch', task_id=0, epoch=None):
        if mode == 'epoch':
            folder = f'task_{task_id}_epoch_{epoch}_model'
        elif mode == 'latest':
            folder = f'model_latest'
        elif mode == 'best':
            folder = f'task_{task_id}_model_best'
        else:
            raise NotImplementedError('Model save mode error!')
        
        os.makedirs(self.opt.checkpoints.save_model_dir, exist_ok=True)
        save_path = os.path.join(self.opt.checkpoints.save_model_dir, folder)

        save_dict = {}
        if isinstance(self.netD_A, nn.DataParallel):
            model_dict_netD_A = self.netD_A.module.state_dict()
            model_dict_netD_B = self.netD_B.module.state_dict()
            model_dict_netG_A = self.netG_A.module.state_dict()
            model_dict_netG_B = self.netG_B.module.state_dict()
        else:
            model_dict_netD_A = self.netD_A.state_dict()
            model_dict_netD_B = self.netD_B.state_dict()
            model_dict_netG_A = self.netG_A.state_dict()
            model_dict_netG_B = self.netG_B.state_dict()
            
        
        # save_dict['schedulers'] = self.schedulers.state_dict()
        save_dict['task_num'] = self.task_num
        save_dict['netD_A'] = model_dict_netD_A
        save_dict['netD_B'] = model_dict_netD_B
        save_dict['netG_A'] = model_dict_netG_A
        save_dict['netG_B'] = model_dict_netG_B
        save_dict['fake_A_pool'] = self.fake_A_pool
        save_dict['fake_B_pool'] = self.fake_B_pool
        save_dict['scheduler_D'] = self.scheduler_D.state_dict()
        save_dict['scheduler_G'] = self.scheduler_G.state_dict()
        ################################################################
        save_dict['best_epoch'] = self.best_epoch
        ################################################################
        # if self._baseline is not None:
        #     save_dict['baseline'] = self._baseline
        # if self._target is not None:
        #     save_dict['target'] = self._target
        torch.save(save_dict, save_path)

    def save_model(self, model_name=None, mode='epoch', task_id=0, epoch=None):
        os.makedirs(self.opt.checkpoints.save_model_dir, exist_ok=True)
        if model_name == 'baseline':
            torch.save(self.restore_net.state_dict(), os.path.join(self.opt.checkpoints.save_model_dir, "baseline.pth"))
            if isinstance(self.restore_net, nn.DataParallel):
                baseline_state_dict = self.restore_net.module.state_dict()
            else:
                baseline_state_dict = self.restore_net.state_dict()
            for k,v in baseline_state_dict.items():
                baseline_state_dict[k] = v.cpu()
            self._baseline = baseline_state_dict
        elif model_name == 'target':
            torch.save(self.restore_net.state_dict(), os.path.join(self.opt.checkpoints.save_model_dir, "target.pth"))
            if isinstance(self.restore_net, nn.DataParallel):
                target_state_dict = self.restore_net.module.state_dict()
            else:
                target_state_dict = self.restore_net.state_dict()
            for k,v in target_state_dict.items():
                target_state_dict[k] = v.cpu()
            self._target = target_state_dict
        else:
            self._save_model(mode=mode, task_id=task_id, epoch=epoch)


    # def select_filter(self, imgs):
    #     faig_average_degradation = 0.0
    #     print('Now we sort the filters!')
    #     utils.fix_random_seed()

    #     for idx, data in enumerate(tqdm(imgs,ncols=80)):
    #         degraded_img, clean_img, filname = data
    #         degraded_img, clean_img = degraded_img.cuda(), clean_img.cuda()
    #         faig_degradation = faig(degraded_img, clean_img, self._baseline, self._target, self.total_step)
    #         faig_average_degradation += np.array(faig_degradation)
        
    #     faig_average_degradation /= len(imgs.dataset)
    #     sorted_location = np.argsort(faig_average_degradation)[::-1]
    #     self.sorted_indices.append(sorted_location)
        
    # def protect_filter(self):
    #     idx = self.sorted_indices[self.task_num-1]
    #     bound = ceil(len(idx)*self.rate)
    #     if isinstance(self.restore_net, nn.DataParallel):
    #         model = self.restore_net.module
    #     else:
    #         model = self.restore_net
    #     index = 0
    #     for module in model.modules():
    #         if isinstance(module, nn.Conv2d):
    #             if module.weight.shape[-1]==3 and index in idx[:bound]:
    #                 module.weight.requires_grad = False
    #             index += 1