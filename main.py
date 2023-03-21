'''
Description: 
Author: Zhang yizhen
Date: 2023-03-07 18:35:29
LastEditors: Zhang yizhen
LastEditTime: 2023-03-21 15:26:41
FilePath: /zyz/Piggyback_GAN/main.py

Copyright (c) 2023 by yizhen_coder@outlook.com, All Rights Reserved. 
'''

import os
from munch import Munch
import continual_train

def main():
    with open('config.yaml') as f:
        opt = Munch.fromYAML(f)
    
    gpus = ','.join([str(i) for i in opt.gpu])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    continual_train.train_and_evaluate(opt)

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    main()