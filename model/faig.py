'''
Description: 
Author: Zhang yizhen
Date: 2023-03-07 18:35:29
LastEditors: Zhang yizhen
LastEditTime: 2023-03-21 15:26:54
FilePath: /zyz/Piggyback_GAN/model/faig.py

Copyright (c) 2023 by yizhen_coder@outlook.com, All Rights Reserved. 
'''
import cv2
import glob
import numpy as np
import os
import torch
from torchvision import transforms
from tqdm import tqdm

from model.networks import MPRNet

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.
    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.
    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def faig(imgs, gt_imgs, baseline_state_dict, target_state_dict, total_step):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # calculate the gradient of images with different degradation
    total_gradient_img = 0

    # approximate the integral via 100 discrete points uniformly
    # sampled along the straight-line path
    for step in range(0, total_step):
        alpha = step / total_step
        interpolate_net_state_dict = {}
        for key, _ in baseline_state_dict.items():
            # a straight-line path between baseline model and target model
            interpolate_net_state_dict[key] = alpha * baseline_state_dict[key] + (1 - alpha) * target_state_dict[key]

        interpolate_net = MPRNet()
        interpolate_net.eval()
        interpolate_net = interpolate_net.to(device)
        interpolate_net.load_state_dict(interpolate_net_state_dict)
        interpolate_net = torch.nn.DataParallel(interpolate_net)

        interpolate_net.zero_grad()
        output = interpolate_net(imgs, optimize=False)
        # measure the distance between the network output and the ground-truth
        # refer to the equation 3 in the main paper
        criterion = torch.nn.MSELoss(reduction='sum')
        loss = criterion(gt_imgs, output)
        # calculate the gradient of F to every filter
        loss.backward()
        grad_list_img = []

        for name, module in interpolate_net.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                if module.weight.shape[-1]==3:
                    grad = module.weight.grad
                    grad = grad.reshape(-1,3,3)
                    grad_list_img.append(grad)
        grad_list_img = torch.cat(grad_list_img, dim=0)
        total_gradient_img += grad_list_img

    diff_list = []
    baseline_net = MPRNet()
    baseline_net.eval()
    baseline_net = baseline_net.to(device)
    baseline_net.load_state_dict(baseline_state_dict)

    target_net = MPRNet()
    target_net.eval()
    target_net = target_net.to(device)
    target_net.load_state_dict(target_state_dict)
    for base_mod, tar_mod in zip(baseline_net.modules(), target_net.modules()):
        if isinstance(base_mod, torch.nn.Conv2d) and isinstance(tar_mod, torch.nn.Conv2d):
            if base_mod.weight.shape[-1]==3:
                variation = base_mod.weight.detach() - tar_mod.weight.detach()
                variation = variation.reshape(-1,3,3)
                diff_list.append(variation)
    
    diff_list = torch.cat(diff_list, dim=0)

    single_faig_img1 = total_gradient_img * diff_list / total_step
    single_faig_img1 = torch.sum(torch.sum(abs(single_faig_img1), dim=1), dim=1)

    faig_img1 = single_faig_img1
    return faig_img1.cpu().numpy()