import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image

class ImageRestorationDataset(Dataset):
    def __init__(self, path, istrain = True, is_baseline = False, patch_size = None, name = None) -> None:
        super(ImageRestorationDataset, self).__init__()
        # self.path = path
        self.ps = patch_size
        self.istrain = istrain
        self.name = name
        self.input_path = os.path.join(path, 'input')
        input_files = sorted(os.listdir(self.input_path))
        self.input_files = [os.path.join(self.input_path, input_file) for input_file in input_files]

        if is_baseline:
            targetname = 'input'
        else:
            targetname = 'target'
        self.target_path = os.path.join(path, targetname)
        target_files = sorted(os.listdir(self.target_path))
        self.target_files = [os.path.join(self.target_path, target_file) for target_file in target_files]

        # if self.mode in ['train', 'val']:
        #     assert(self.ps is not None)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_path = self.input_files[index]
        file_name = input_path.split(os.sep)[-1]
        # file_name.replace('_rain', '_restore')
        input_image = Image.open(input_path)

        inp_img = TF.to_tensor(input_image)
        
        target_path = self.target_files[index]
        target_image = Image.open(target_path)
        tar_img = TF.to_tensor(target_image)

        ps = self.ps
        w, h = input_image.size
        padw = ps-w if w<ps else 0
        padh = ps-h if h<ps else 0

        if padw != 0 or padh != 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')

        hh, ww = tar_img.shape[1], tar_img.shape[2]
        rr     = random.randint(0, hh-ps)
        cc     = random.randint(0, ww-ps)
        inp_img = inp_img[:, rr:rr+ps, cc:cc+ps]
        tar_img = tar_img[:, rr:rr+ps, cc:cc+ps]

        if self.istrain:
            #aug = random.randint(0, 2)
            #if aug == 1:
            #   sat_factor = 1 + (0.2 - 0.4*np.random.rand())
            #   inp_img = TF.adjust_saturation(inp_img, sat_factor)
            #   tar_img = TF.adjust_saturation(tar_img, sat_factor)
    
            aug = random.randint(0, 8)
  
            # Data Augmentations
            if aug==1:
                inp_img = inp_img.flip(1)
                tar_img = tar_img.flip(1)
            elif aug==2:
                inp_img = inp_img.flip(2)
                tar_img = tar_img.flip(2)
            elif aug==3:
                inp_img = torch.rot90(inp_img,dims=(1,2))
                tar_img = torch.rot90(tar_img,dims=(1,2))
            elif aug==4:
                inp_img = torch.rot90(inp_img,dims=(1,2), k=2)
                tar_img = torch.rot90(tar_img,dims=(1,2), k=2)
            elif aug==5:
                inp_img = torch.rot90(inp_img,dims=(1,2), k=3)
                tar_img = torch.rot90(tar_img,dims=(1,2), k=3)
            elif aug==6:
                inp_img = torch.rot90(inp_img.flip(1),dims=(1,2))
                tar_img = torch.rot90(tar_img.flip(1),dims=(1,2))
            elif aug==7:
                inp_img = torch.rot90(inp_img.flip(2),dims=(1,2))
                tar_img = torch.rot90(tar_img.flip(2),dims=(1,2))

        return inp_img, tar_img, file_name
