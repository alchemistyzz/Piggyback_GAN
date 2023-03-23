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
from model.networks import define_G
from model.faig import faig
import utils

class PiggybackGAN(nn.Module):
    def __init__(self, opt, task_num=1):
        super(PiggybackGAN, self).__init__()
        self.opt = opt
        self.task_num = task_num
        self.is_train = opt.is_train

        self.sorted_indices = []

        self.net_filter_list = []
        self.weights = []
        # self.bias = []
        self.restore_net = define_G(self.opt.input_nc, self.opt.output_nc, self.opt.ngf, self.opt.netG, self.opt.norm,
                                        self.opt.dropout, self.opt.init_type, self.opt.init_gain, self.task_num, self.net_filter_list)
        self.model_names = ['restore_net']
        

        if self.is_train:
            self._baseline = None
            self._target = None
            # self.total_step = opt.faig.total_step
            # self.rate = opt.faig.rate
        
            self.criterion = torch.nn.L1Loss()
            self.optimizer = torch.optim.Adam(self.restore_net.parameters(), lr=opt.train.lr_init[self.task_num-1], betas=(0.9, 0.999))
            # self.optimizer_baseline = torch.optim.Adam(self.restore_net.parameters(), lr=opt.train.lr_init[self.task_num-1], betas=(0.9, 0.999))
            self.scheduler = networks.get_scheduler(self.optimizer, self.opt, self.task_num-1, self.opt.train.num_epochs[self.task_num-1])
            # self.scheduler_baseline = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_baseline, T_max=self.opt.train.num_epoch_baseline, eta_min=self.opt.train.lr_min[self.task_num-1])
            ###############################
            self.best_epoch = 1
            # self.ewclamda = opt.train.ewclamda
            # self.importance = {}
            # self.best_params = {}
            # self.previous_best_params = {}
            # self.now_task_id = self.task_num-1
            ###############################


    def set_device(self, device='cpu'):
        if device == 'cpu':
            pass
        elif device == 'gpu':
            # for model_name in self.model_names:
            #     model = getattr(self, model_name)
            #     model.cuda()
            #     model = nn.DataParallel(model)
            #     setattr(self, model_name, model)
            self.cuda()
            self = nn.DataParallel(self)

    def update_learning_rate(self, train_baseline=False):
        if train_baseline:
            old_lr = self.optimizer_baseline.param_groups[0]['lr']
            self.scheduler_baseline.step()
            lr = self.optimizer_baseline.param_groups[0]['lr']
        else:
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
        print(f'Learning rate {old_lr:.8f} -> {lr:.8f}')

    def update_task(self):
        # self.protect_filter()
        # self.save_model(model_name='target')
        
        self._baseline = self._target
        self._target = None
        self.task_num += 1
        self.restore_net = define_G(self.opt.input_nc, self.opt.output_nc, self.opt.ngf, self.opt.netG, self.opt.norm,
                                        self.opt.dropout, self.opt.init_type, self.opt.init_gain, self.task_num, self.net_filter_list)
        self.optimizer = torch.optim.Adam(self.restore_net.parameters(), lr=self.opt.train.lr_init[self.task_num-1], betas=(0.9, 0.999))
        # self.optimizer_baseline = torch.optim.Adam(self.restore_net.parameters(), lr=self.opt.train.lr_init[self.task_num-1], betas=(0.9, 0.999))
        self.scheduler = networks.get_scheduler(self.optimizer, self.opt, self.task_num-1, self.opt.train.num_epochs[self.task_num-1])
        # self.scheduler_baseline = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_baseline, T_max=self.opt.train.num_epoch_baseline, eta_min=self.opt.train.lr_min[self.task_num-1])
        ###############################
        # self.now_task_id +=1
             
        ###############################

    def set_input(self, input):
        def pad_img(img):
            img_multiple_of = 8
            h,w = img.shape[2:]
            H,W = ((h+img_multiple_of)//img_multiple_of)*img_multiple_of, ((w+img_multiple_of)//img_multiple_of)*img_multiple_of
            padh = H-h if h%img_multiple_of!=0 else 0
            padw = W-w if w%img_multiple_of!=0 else 0
            img = F.pad(img, (0,padw,0,padh), 'reflect')
            return img

        self.degraded_image = pad_img(input[0].cuda())
        self.clean_image = pad_img(input[1].cuda())
        self.file_name = input[2]
        h,w = min(self.degraded_image.shape[-2], self.clean_image.shape[-2]), min(self.degraded_image.shape[-1], self.clean_image.shape[-1])
        self.degraded_image = TF.crop(self.degraded_image, 0, 0, h, w)
        self.clean_image = TF.crop(self.clean_image, 0, 0, h, w)

    def backward_G(self):
        lambda_ = 1.0
        # params = {n: p for n, p in self.named_parameters() if p.requires_grad}
        #todo:
        self.loss = lambda_*self.criterion(self.restored_image, self.clean_image)
        self.loss.backward()

    def optimize_parameters(self, train_baseline=False):
        self.forward()
        if train_baseline:
            self.optimizer_baseline.zero_grad()
        else:
            self.optimizer.zero_grad()
        self.backward_G()
        if train_baseline:
            self.optimizer_baseline.step()
        else:
            self.optimizer.step()

    def forward(self):
        # self.restored_image = self.restore_net(self.degraded_image, optimize=optimize)
        self.restored_image = self.restore_net(self.degraded_image)
        # if optimize:
        #     self.restored_image = [torch.clamp(img, 0, 1) for img in self.restored_image]
        # else:
        self.restored_image = torch.clamp(self.restored_image, 0, 1)

    def inference(self):
        with torch.no_grad():
            self.forward()

        return self.restored_image

    def get_current_losses(self):
        loss_dict = OrderedDict()
        ################################
        loss_dict['PiggyGAN_Loss'] = self.loss
        ################################
        return loss_dict
        
    def get_current_psnr(self):
        if isinstance(self.restored_image, list):
            pred_images = self.restored_image[0]
        else:
            pred_images = self.restored_image

        gt_images = self.clean_image

        total_psnr = 0

        for pred_image, gt_image in zip(pred_images, gt_images):
            
            pred_img = (pred_image.permute(1,2,0).cpu().detach().numpy()*255).astype(np.uint8)
            gt_img = (gt_image.permute(1,2,0).cpu().detach().numpy()*255).astype(np.uint8)
            total_psnr += psnr(gt_img, pred_img)

        return total_psnr

    def get_current_ssim(self):
        if isinstance(self.restored_image, list):
            pred_images = self.restored_image[0]
        else:
            pred_images = self.restored_image

        gt_images = self.clean_image

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
        self.restore_net.load_state_dict(load_dict['model'])
        self.best_epoch = load_dict['best_epoch']
        self.net_filter_list = load_dict['net_filter_list']
        self.weights = load_dict['weights']
        # self.bias = load_dict['bias']
    
        

        for _ in range(load_dict['task_num']-1):
            self.update_task()
       
        if 'baseline' in load_dict:
            self._baseline = load_dict['baseline']
        if 'target' in load_dict:
            self._target = load['target']
        total_epoch = load_dict['scheduler']['last_epoch'] + load_dict['scheduler']['after_scheduler'].last_epoch
        for _ in range(total_epoch):
            self.scheduler.step()
      

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
        # if isinstance(self.restore_net, nn.DataParallel):
        # model_dict = self.restore_net.module.state_dict()
        # else:
        model_dict = self.restore_net.state_dict()
            
        
        save_dict['model'] = model_dict
        # print(model_dict)
        # save_dict['indices'] = self.sorted_indices
        # save_dict['rate'] = self.opt.faig.rate
        # save_dict['total_step'] = self.opt.faig.total_step
        save_dict['scheduler'] = self.scheduler.state_dict()
        save_dict['task_num'] = self.task_num
        ################################################################
        save_dict['best_epoch'] = self.best_epoch
        save_dict['net_filter_list'] = self.net_filter_list
        save_dict['weights']  = self.weights
        # save_dict['bias'] =self.bias
        ################################################################
        if self._baseline is not None:
            save_dict['baseline'] = self._baseline
        if self._target is not None:
            save_dict['target'] = self._target
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