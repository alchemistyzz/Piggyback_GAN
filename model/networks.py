'''
Description: 
Author: Zhang yizhen
Date: 2023-03-07 18:35:29
LastEditors: Zhang yizhen
LastEditTime: 2023-03-21 23:02:28
FilePath: /Piggyback_GAN/model/networks.py

Copyright (c) 2023 by yizhen_coder@outlook.com, All Rights Reserved. 
'''

import copy, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.optim import lr_scheduler
from warmup_scheduler import GradualWarmupScheduler
import numpy as np

import functools

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_scheduler(optimizer, opt, task_id, epoch):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions.
                              opt.train.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    # if opt.train.lr_policy == 'linear':
    #     def lambda_rule(epoch):
    #         lr_l = 1.0 - max(0, epoch + opt.TRAINING.NUM_EPOCHS - opt.n_epochs) / float(opt.n_epochs_decay + 1)
    #         return lr_l
    #     scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    warmup_epochs = 3
    if opt.train.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - warmup_epochs) / float(opt.train.num_epochs[task_id] + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.train.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    elif opt.train.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.train.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch-warmup_epochs, eta_min=opt.train.lr_min[task_id])
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.train.lr_policy)
    scheduler_GWS = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler)
    return scheduler_GWS

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__

        if classname.find('Piggyback') != -1:
            if hasattr(m, 'unc_filt'):
                if init_type == 'normal':
                    init.normal_(m.unc_filt.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.unc_filt.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.unc_filt.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.unc_filt.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'weights_mat'):
                if init_type == 'normal':
                    init.normal_(m.weights_mat.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weights_mat.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weights_mat.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weights_mat.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1): #classname.find('Conv') != -1 or 
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def load_pb_conv(network, unc_filt, weight_mat,bias, task_id):
    conv_idx = 0
    cnt=0
    for name, module in network.named_modules():
        if isinstance(module, PiggybackConv):
            module.unc_filt = nn.Parameter(unc_filt[conv_idx][task_id])
            if len(bias) > 0:
                module.bias = nn.Parameter(bias[conv_idx][task_id])
            if task_id > 0:
                module.weights_mat = nn.Parameter(weight_mat[conv_idx][task_id - 1])
                module.concat_unc_filter = torch.cat(unc_filt[conv_idx][0:task_id], dim=0)
            conv_idx += 1
        elif isinstance(module, PiggybackTransposeConv):
            module.unc_filt = nn.Parameter(unc_filt[conv_idx][task_id])
            if len(bias) > 0:
                module.bias = nn.Parameter(bias[conv_idx][task_id])
            if task_id > 0:
                module.weights_mat = nn.Parameter(unc_filt[conv_idx][task_id])
                module.concat_unc_filter = torch.cat(unc_filt[conv_idx][0:task_id], dim=1)
            conv_idx += 1

    # layer_list = list(network)

    # conv_idx = 0

    # for layer_idx in range(len(layer_list)):
    #     if isinstance(layer_list[layer_idx], PiggybackConv):
    #         layer_list[layer_idx].unc_filt = nn.Parameter(unc_filt[conv_idx][task_id])
    #         if task_id > 0:
    #             layer_list[layer_idx].weights_mat = nn.Parameter(weight_mat[conv_idx][task_id - 1])
    #             layer_list[layer_idx].concat_unc_filter = torch.cat(unc_filt[conv_idx][0:task_id], dim=0)
    #         conv_idx += 1
    #     elif isinstance(layer_list[layer_idx], PiggybackTransposeConv):
    #         layer_list[layer_idx].unc_filt = nn.Parameter(unc_filt[conv_idx][task_id])
    #         if task_id > 0:
    #             layer_list[layer_idx].weights_mat = nn.Parameter(weight_mat[conv_idx][task_id - 1])
    #             layer_list[layer_idx].concat_unc_filter = torch.cat(unc_filt[conv_idx][0:task_id], dim=1)
    #         conv_idx += 1

    return network

def define_G(input_nc, output_nc, ngf, netG, norm='instance', use_dropout=False, init_type='normal', init_gain=0.02, task_num=1, filter_list=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm)
    
    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    task_id = task_num - 1 
    net = replace_conv(net, task_id, filter_list)
        # all_layers_netG = nn.ModuleList()
        # all_layers_netG = remove_sequential(net, all_layers_netG)
        # copy_all_layers_netG = copy.deepcopy(all_layers_netG)
        # replaced_netG = replace_conv(copy_all_layers_netG, task_id, filter_list)
        # del net
        # net = nn.Sequential(*replaced_netG)
        # del replaced_netG
        # del copy_all_layers_netG
        # del all_layers_netG

    # if len(opt.gpu) > 0:
    net = nn.DataParallel(net)

    return net



def define_D(input_nc, ndf, netD, norm='instance', init_type='normal', init_gain=0.02, n_layers_D=3):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)

    init_weights(net, init_type, init_gain=init_gain)
    
    return net


def remove_sequential(network, layer_list):
    for layer in network.children():
        if list(layer.children()) == []: # if leaf node, add it to list
            layer_list.append(layer)
        else:
            layer_list = remove_sequential(layer, layer_list)

    return layer_list

def replace_conv(network, task_id=0, filter_list=None, piggylamdas=0.25):#layer_list, task_id=0, filter_list=None):
    conv_idx = 0

    for name, module in network.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, PiggybackConv):
            net = network
            for idx in name.split('.')[:-1]:
                net = getattr(net, idx)
            if task_id == 0:
                unc_filter_list = None
            else:
                unc_filter_list = filter_list[conv_idx]
                conv_idx += 1
            replace_module = PiggybackConv(in_channels=module.in_channels, out_channels=module.out_channels, kernel_size=module.kernel_size, stride=module.stride, padding=module.padding,lambdas=piggylamdas,task=task_id+1, unc_filt_list=unc_filter_list)
            setattr(net, name.split('.')[-1], replace_module)
            continue
    
        if isinstance(module, nn.ConvTranspose2d) or isinstance(module, PiggybackTransposeConv):
            net = network
            for idx in name.split('.')[:-1]:
                net = getattr(net, idx)
            if task_id == 0:
                unc_filter_list = None
            else:
                unc_filter_list = filter_list[conv_idx]
                conv_idx += 1 
            replace_module = PiggybackTransposeConv(in_channels=module.in_channels, out_channels=module.out_channels, kernel_size=module.kernel_size, stride=module.stride, padding=module.padding,lambdas=piggylamdas, task=task_id+1, unc_filt_list=unc_filter_list)
            setattr(net, name.split('.')[-1], replace_module)
            continue
    #################################
    # conv_idx = 0

    # for layer_idx in range(len(filter_list)):

    #     if isinstance(filter_list[layer_idx], nn.Conv2d) or isinstance(filter_list[layer_idx], PiggybackConv):
    #         el = filter_list[layer_idx]
    #         del filter_list[layer_idx]

    #         if task_id == 0:
    #             unc_filter_list = None
    #         else:
    #             unc_filter_list = filter_list[conv_idx]
    #             conv_idx += 1

    #         conv_layer = PiggybackConv(in_channels=el.in_channels, out_channels=el.out_channels, kernel_size=el.kernel_size, stride=el.stride, padding=el.padding, task=task_id+1, unc_filt_list=unc_filter_list)
    #         # conv_layer = nn.Conv2d(in_channels=el.in_channels, out_channels=el.out_channels, kernel_size=el.kernel_size, stride=el.stride, padding=el.padding)
    #         filter_list.insert(layer_idx, conv_layer)

    #         del el
    #         del conv_layer

    #         continue

    #     if isinstance(filter_list[layer_idx], nn.ConvTranspose2d) or isinstance(filter_list[layer_idx], PiggybackTransposeConv):
    #         el = filter_list[layer_idx]
    #         del filter_list[layer_idx]

    #         if task_id == 0:
    #             unc_filter_list = None
    #         else:
    #             unc_filter_list = filter_list[conv_idx]
    #             conv_idx += 1

    #         conv_layer = PiggybackTransposeConv(in_channels=el.in_channels, out_channels=el.out_channels, kernel_size=el.kernel_size, stride=el.stride, padding=el.padding, output_padding=el.output_padding, task=task_id+1, unc_filt_list=unc_filter_list)
    #         # conv_layer = nn.ConvTranspose2d(in_channels=el.in_channels, out_channels=el.out_channels, kernel_size=el.kernel_size, stride=el.stride, padding=el.padding, output_padding=el.output_padding)
    #         filter_list.insert(layer_idx, conv_layer)

    #         del el
    #         del conv_layer

    #         continue
    ##################################

    return network
# %%
class Identity(nn.Module):
    def forward(self, x):
        return x

class PiggybackConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0), bias=True, lambdas=0.25, task=1, unc_filt_list=None):
        super(PiggybackConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.task_num = task 
        self.lambdas = lambdas
        self.lamb_num = math.ceil(lambdas*out_channels)
        self.lamb_rem_num = out_channels - self.lamb_num

        if self.task_num == 1:
            self.unc_filt = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]))
        else: 
            # after training, save unc_filt and weight_mat into files, so that u can use it now. 
            self.unc_filt = nn.Parameter(torch.Tensor(self.lamb_num, self.in_channels, self.kernel_size[0], self.kernel_size[1])) 
            self.weights_mat = nn.Parameter(torch.Tensor((self.out_channels + (self.task_num-2)*self.lamb_num), self.lamb_rem_num))
            self.register_buffer('concat_unc_filter', torch.cat(unc_filt_list, dim=0))

    def forward(self, input_x):
        if self.task_num == 1:
            return F.conv2d(input_x, self.unc_filt, bias=self.bias, stride=self.stride, padding=self.padding)           
        else:
            self.reshape_unc = torch.reshape(self.concat_unc_filter, (self.concat_unc_filter.shape[1]*self.concat_unc_filter.shape[2]*self.concat_unc_filter.shape[3], self.concat_unc_filter.shape[0]))
            self.reshape_unc_mul_w = torch.matmul(self.reshape_unc, self.weights_mat)
            self.pb_filt = torch.reshape(self.reshape_unc_mul_w, (self.reshape_unc_mul_w.shape[1], self.concat_unc_filter.shape[1], self.concat_unc_filter.shape[2], self.concat_unc_filter.shape[3]))
            self.final_weight_mat = torch.cat((self.unc_filt, self.pb_filt),dim=0)
            self.final_weight_mat = self.final_weight_mat.to(input_x.device)

            return F.conv2d(input_x, self.final_weight_mat, bias=self.bias, stride=self.stride, padding=self.padding)

class PiggybackTransposeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0), output_padding=(0,0), bias=True, lambdas=0.25, task=1, unc_filt_list=None):
        super(PiggybackTransposeConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)

        self.task_num = task 
        self.lambdas = lambdas
        self.lamb_num = math.ceil(lambdas*out_channels)
        self.lamb_rem_num = out_channels - self.lamb_num

        if self.task_num == 1:
            self.unc_filt = nn.Parameter(torch.Tensor(self.in_channels, self.out_channels, self.kernel_size[0], self.kernel_size[1]))
        else: 
            self.unc_filt = nn.Parameter(torch.Tensor(self.in_channels, self.lamb_num, self.kernel_size[0], self.kernel_size[1])) 
            self.weights_mat = nn.Parameter(torch.Tensor((self.out_channels + (self.task_num-2)*self.lamb_num), self.lamb_rem_num))
            self.register_buffer('concat_unc_filter', torch.cat(unc_filt_list, dim=1))

    def forward(self, input_x):
        if self.task_num == 1:
            return F.conv_transpose2d(input_x, self.unc_filt, bias=self.bias, stride=self.stride, padding=self.padding, output_padding=self.output_padding)
            
        else:
            self.reshape_unc = torch.reshape(self.concat_unc_filter, (self.concat_unc_filter.shape[0]*self.concat_unc_filter.shape[2]*self.concat_unc_filter.shape[3], self.concat_unc_filter.shape[1]))
            self.reshape_unc_mul_w = torch.matmul(self.reshape_unc, self.weights_mat)
            self.pb_filt = torch.reshape(self.reshape_unc_mul_w, (self.concat_unc_filter.shape[0], self.reshape_unc_mul_w.shape[1], self.concat_unc_filter.shape[2], self.concat_unc_filter.shape[3]))
            self.final_weight_mat = torch.cat((self.unc_filt, self.pb_filt),dim=1)
            self.final_weight_mat = self.final_weight_mat.to(input_x.device)

            return F.conv_transpose2d(input_x, self.final_weight_mat, bias=self.bias, stride=self.stride, padding=self.padding, output_padding=self.output_padding)

class DistAlignLoss(nn.Module):
    def __init__(self, target_src_label=1.0, target_tgt_label=0.0, target_dupe_label=0.5):
        super(DistAlignLoss, self).__init__()
        self.register_buffer('target_src_label', torch.tensor(target_src_label))
        self.register_buffer('target_tgt_label', torch.tensor(target_tgt_label))
        self.register_buffer('target_dupe_label', torch.tensor(target_dupe_label))
        self.loss = nn.MSELoss()
    
    def get_target_tensor(self, prediction, target_is_real, target_is_dupe=False):
        if target_is_dupe:
            target_tensor = self.target_dupe_label
        elif target_is_real:
            target_tensor = self.target_src_label
        else:
            target_tensor = self.target_tgt_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real=None, target_is_dupe=False):
        if isinstance(prediction[0], list):
            loss = torch.tensor(0., requires_grad=True).to(prediction[0][-1].device)
            for pred in prediction:
                pred = pred[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real, target_is_dupe)
                loss = loss.clone() + self.loss(pred, target_tensor)
            return loss.mean()
        else:
            target_tensor = self.get_target_tensor(prediction[-1], target_is_real, target_is_dupe)
            return self.loss(prediction[-1], target_tensor)

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        if self.kernel.device != img.device:
            self.kernel = self.kernel.to(img.device)
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss

class MPRNetLoss(nn.Module):
    def __init__(self):
        super(MPRNetLoss, self).__init__()
        self.criterion_char = CharbonnierLoss()
        self.criterion_edge = EdgeLoss()
    def forward(self, restored, target):
        loss_char = torch.sum(torch.cat(tuple(self.criterion_char(restored[j],target).unsqueeze(0) for j in range(len(restored))), dim=0), dim=0)
        loss_edge = torch.sum(torch.cat(tuple(self.criterion_edge(restored[j],target).unsqueeze(0) for j in range(len(restored))), dim=0), dim=0)
        return loss_char + (0.05 * loss_edge)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out



class UnetGenerator(nn.Module):
    def __init__(self, in_ch=3, l1_ch=64, l2_ch=128, l3_ch=256, out_ch=3):
        super(UnetGenerator, self).__init__()
        # conv1-2
        self.inc = nn.Sequential(
            ResBlk(in_ch, l1_ch),
            ResBlk(l1_ch, l1_ch)
        )
        # conv3-5
        self.conv1 = nn.Sequential(
            ResBlk(l1_ch, l2_ch, 2),
            ResBlk(l2_ch, l2_ch),
            ResBlk(l2_ch, l2_ch)
        )
        # conv6-11
        self.conv2 = nn.Sequential(
            ResBlk(l2_ch, l3_ch, 2),
            ResBlk(l3_ch, l3_ch),
            ResBlk(l3_ch, l3_ch),
            ResBlk(l3_ch, l3_ch),
            ResBlk(l3_ch, l3_ch),
            ResBlk(l3_ch, l3_ch)
        )

        self.mu = nn.Sequential(
            nn.Conv2d(l3_ch, 128, 1),
            nn.Conv2d(128, 64, 1),
            nn.Conv2d(64, 16, 1),
            # nn.Flatten(),
            # nn.Linear(44*44, 8)
            # nn.AdaptiveAvgPool2d(1)
        )

        self.logvar = nn.Sequential(
            nn.Conv2d(l3_ch, 128, 1),
            nn.Conv2d(128, 64, 1),
            nn.Conv2d(64, 16, 1),
            # nn.Flatten(),
            # nn.Linear(44*44, 8)
            # nn.AdaptiveAvgPool2d(1)
        )

        self.decoder_input = nn.Sequential(
            # nn.Linear(8,44*44),
            # nn.Unflatten(1, (1,44,44)),
            nn.Conv2d(16,64,1),
            nn.Conv2d(64,128,1),
            nn.Conv2d(128,l3_ch,1)
        )

        self.up1 = UpSample_U(skip_input=l3_ch * 2 + l2_ch, output_features=l2_ch)
        # conv12-14
        self.conv3 = nn.Sequential(
            ResBlk(l2_ch, l2_ch),
            ResBlk(l2_ch, l2_ch),
            ResBlk(l2_ch, l2_ch),
        )
        self.ca1 = ChannelAttention(l2_ch)

        self.up2 = UpSample_U(skip_input=l2_ch + l1_ch, output_features=l1_ch)

        # conv15-16
        self.conv4 = nn.Sequential(
            ResBlk(l1_ch, l1_ch),
            ResBlk(l1_ch, l1_ch),
        )
        self.ca2 = ChannelAttention(l1_ch)

        self.outc = nn.Sequential(
            outconv(l1_ch, out_ch),
        )
        # self.ca3 = ChannelAttention(out_ch)

    def reparameterize(self, mu, logvar, inference):
        std = torch.exp(0.5 * logvar)
        # eps = torch.randn_like(std)
        eps = torch.randn(mu.shape[0], mu.shape[1], 1, 1).expand(mu.shape[0], mu.shape[1], mu.shape[2], mu.shape[3]).to(mu.device)
        if inference:
            eps = eps * 9.9
        return eps * std + mu

    def forward(self, x, flow='enc_dec', inference=False):
        inx = self.inc(x)  # conv1-3 64

        conv1 = self.conv1(inx)  # conv3-5 128

        conv2 = self.conv2(conv1)  # conv6-11 256
        if flow == 'enc':
            return conv2

        mu = self.mu(conv2)
        logvar = self.logvar(conv2)
        z = self.reparameterize(mu, logvar, inference)
        z = self.decoder_input(z)
        conv2 = torch.cat((conv2, z), dim=1)
        mu = torch.reshape(mu, (mu.shape[0], -1))
        logvar = torch.reshape(logvar, (logvar.shape[0], -1))

        # print(mu.shape)
        # print(logvar.shape)

        kid_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1))
        # print(kid_loss)

        up1 = self.up1(conv2, conv1)  # upscale_1 256,128->128
        conv3 = self.conv3(up1)  # conv12-14
        conv3 = self.ca1(conv3) * conv3

        up2 = self.up2(conv3, inx)  # upscale_2 128, 64->64

        conv4 = self.conv4(up2)  # conv15-16
        conv4 = self.ca2(conv4) * conv4

        out = self.outc(conv4)
        # out = self.ca3(out) * out
        out = torch.sigmoid(out + x)

        return out, kid_loss  # conv4（channel=64)，out（channel=3)

    def get_kid_loss(self):
        return self.kid_loss

class ResBlk(nn.Module):

    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.in1 = nn.InstanceNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.in2 = nn.InstanceNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in or stride != 1:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.InstanceNorm2d(ch_out)
            )

    def forward(self, x):
        """
        :param x:[b, ch, h, w]
        :return:
        """
        out = F.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        # short cut
        # element-wise add: [b, ch_in, h, w] with [b, ch_out, h, w]
        out = self.extra(x) + out
        out = F.relu(out)
        return out

class UpSample_U(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample_U, self).__init__()
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB(self.convB(self.leakyreluA(self.convA(torch.cat([up_x, concat_with], dim=1)))))

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class GaussianNoise(nn.Module):                         # Try noise just for real or just for fake images.
    def __init__(self, std=0.1, decay_rate=0):
        super().__init__()
        self.std = std
        self.decay_rate = decay_rate

    def decay_step(self):
        self.std = max(self.std - self.decay_rate, 0)

    def forward(self, x):
        if self.training:
            return x + torch.empty_like(x).normal_(std=self.std)
        else:
            return x

class Discriminator(nn.Module):
    def __init__(self, opt, input_nc):
        super(Discriminator, self).__init__()
        # input_nc = opt.DISCRIMINATOR.INPUT_NC
        self.num_D = opt.DISCRIMINATOR.NUM_D
        self.n_layers = opt.DISCRIMINATOR.N_LAYERS
        self.getIntermFeat = opt.DISCRIMINATOR.GET_INTERM_FEAT

        ndf = opt.DISCRIMINATOR.NDF
        use_sigmoid = opt.DISCRIMINATOR.USE_SIGMOID
        
        if opt.DISCRIMINATOR.NORM == 'batch':
            norm_layer = nn.BatchNorm2d
        elif opt.DISCRIMINATOR.NORM == 'instance':
            norm_layer = nn.InstanceNorm2d
        else:
            raise ValueError('NORM type is not supported')

        for i in range(self.num_D):
            netD = NLayerDiscriminator(input_nc, opt, ndf, self.n_layers, norm_layer, use_sigmoid, self.getIntermFeat)
            if self.getIntermFeat:
                for j in range(self.n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

def SN(module, mode=True):
    if mode:
        return torch.nn.utils.spectral_norm(module)

    return module





######################   MPRnet   ########################################
##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

##########################################################################
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img

##########################################################################
## U-Net

class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff):
        super(Encoder, self).__init__()

        self.encoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12  = DownSample(n_feat, scale_unetfeats)
        self.down23  = DownSample(n_feat+scale_unetfeats, scale_unetfeats)

        # Cross Stage Feature Fusion (CSFF)
        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat,                     n_feat,                     kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(n_feat+scale_unetfeats,     n_feat+scale_unetfeats,     kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2), kernel_size=1, bias=bias)

            self.csff_dec1 = nn.Conv2d(n_feat,                     n_feat,                     kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat+scale_unetfeats,     n_feat+scale_unetfeats,     kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2), kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])
        
        return [enc1, enc2, enc3]

class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Decoder, self).__init__()

        self.decoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat,                 kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat+scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21  = SkipUpSample(n_feat, scale_unetfeats)
        self.up32  = SkipUpSample(n_feat+scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1,dec2,dec3]

##########################################################################
##---------- Resizing Modules ----------    
class DownSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels+s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

class SkipUpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x

##########################################################################
## Original Resolution Block (ORB)
class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

##########################################################################
class ORSNet(nn.Module):
    def __init__(self, n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):
        super(ORSNet, self).__init__()

        self.orb1 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat, scale_unetfeats)
        self.up_dec1 = UpSample(n_feat, scale_unetfeats)

        self.up_enc2 = nn.Sequential(UpSample(n_feat+scale_unetfeats, scale_unetfeats), UpSample(n_feat, scale_unetfeats))
        self.up_dec2 = nn.Sequential(UpSample(n_feat+scale_unetfeats, scale_unetfeats), UpSample(n_feat, scale_unetfeats))

        self.conv_enc1 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc2 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc3 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)

        self.conv_dec1 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.orb2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))

        x = self.orb3(x)
        x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))

        return x


##########################################################################
class MPRNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=96, scale_unetfeats=48, scale_orsnetfeats=32, num_cab=8, kernel_size=3, reduction=4, bias=False):
        super(MPRNet, self).__init__()

        act=nn.PReLU()
        self.shallow_feat1 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat3 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))

        # Cross Stage Feature Fusion (CSFF)
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage3_orsnet = ORSNet(n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab)

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias)
        
        self.concat12  = conv(n_feat*2, n_feat, kernel_size, bias=bias)
        self.concat23  = conv(n_feat*2, n_feat+scale_orsnetfeats, kernel_size, bias=bias)
        self.tail     = conv(n_feat+scale_orsnetfeats, out_c, kernel_size, bias=bias)

    def forward(self, x3_img, optimize=True):#, flow='enc_dec'):
        if isinstance(x3_img, list):
            x3_img = x3_img[0]
        # Original-resolution Image for Stage 3
        H = x3_img.size(2)
        W = x3_img.size(3)

        # Multi-Patch Hierarchy: Split Image into four non-overlapping patches

        # Two Patches for Stage 2
        x2top_img  = x3_img[:,:,0:int(H/2),:]
        x2bot_img  = x3_img[:,:,int(H/2):H,:]

        # Four Patches for Stage 1
        x1ltop_img = x2top_img[:,:,:,0:int(W/2)]
        x1rtop_img = x2top_img[:,:,:,int(W/2):W]
        x1lbot_img = x2bot_img[:,:,:,0:int(W/2)]
        x1rbot_img = x2bot_img[:,:,:,int(W/2):W]

        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x1ltop = self.shallow_feat1(x1ltop_img)
        x1rtop = self.shallow_feat1(x1rtop_img)
        x1lbot = self.shallow_feat1(x1lbot_img)
        x1rbot = self.shallow_feat1(x1rbot_img)
        
        ## Process features of all 4 patches with Encoder of Stage 1
        feat1_ltop = self.stage1_encoder(x1ltop)
        feat1_rtop = self.stage1_encoder(x1rtop)
        feat1_lbot = self.stage1_encoder(x1lbot)
        feat1_rbot = self.stage1_encoder(x1rbot)
        
        ## Concat deep features
        feat1_top = [torch.cat((k,v), 3) for k,v in zip(feat1_ltop,feat1_rtop)]
        feat1_bot = [torch.cat((k,v), 3) for k,v in zip(feat1_lbot,feat1_rbot)]
        
        ## Pass features through Decoder of Stage 1
        res1_top = self.stage1_decoder(feat1_top)
        res1_bot = self.stage1_decoder(feat1_bot)

        ## Apply Supervised Attention Module (SAM)
        x2top_samfeats, stage1_img_top = self.sam12(res1_top[0], x2top_img)
        x2bot_samfeats, stage1_img_bot = self.sam12(res1_bot[0], x2bot_img)

        ## Output image at Stage 1
        stage1_img = torch.cat([stage1_img_top, stage1_img_bot],2) 
        ##-------------------------------------------
        ##-------------- Stage 2---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x2top  = self.shallow_feat2(x2top_img)
        x2bot  = self.shallow_feat2(x2bot_img)

        ## Concatenate SAM features of Stage 1 with shallow features of Stage 2
        x2top_cat = self.concat12(torch.cat([x2top, x2top_samfeats], 1))
        x2bot_cat = self.concat12(torch.cat([x2bot, x2bot_samfeats], 1))

        ## Process features of both patches with Encoder of Stage 2
        feat2_top = self.stage2_encoder(x2top_cat, feat1_top, res1_top)
        feat2_bot = self.stage2_encoder(x2bot_cat, feat1_bot, res1_bot)

        ## Concat deep features
        feat2 = [torch.cat((k,v), 2) for k,v in zip(feat2_top,feat2_bot)]

        # if flow == 'enc':
        #     return feat2[-1]

        ## Pass features through Decoder of Stage 2
        res2 = self.stage2_decoder(feat2)

        ## Apply SAM
        x3_samfeats, stage2_img = self.sam23(res2[0], x3_img)


        ##-------------------------------------------
        ##-------------- Stage 3---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x3     = self.shallow_feat3(x3_img)

        ## Concatenate SAM features of Stage 2 with shallow features of Stage 3
        x3_cat = self.concat23(torch.cat([x3, x3_samfeats], 1))
        
        x3_cat = self.stage3_orsnet(x3_cat, feat2, res2)

        stage3_img = self.tail(x3_cat)

        if optimize:
            return [stage3_img+x3_img, stage2_img, stage1_img]
        else:
            return stage3_img+x3_img
class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
