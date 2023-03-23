import os
import torch
import torchvision.transforms.functional as TF
import numpy as np
import random

def fix_random_seed(seed = 1234):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

class Visualizer():
    def __init__(self, writer, model, open_visualize=True, save_img=True) -> None:
        self.writer = writer
        self.model = model
        self.open_visualize=open_visualize
        self.save_img = save_img

    def visualize_scalars(self, tag, scalar, step):
        if not self.open_visualize:
            return
        if isinstance(scalar, dict):
            self.writer.add_scalars(tag, scalar, step)
        else:
            self.writer.add_scalar(tag, scalar, step)

    def visualize_images(self, epoch, **kwargs):
        if not self.open_visualize:
            return
        for tag, images in kwargs.items():
            self.writer.add_images(tag, images, epoch)

    def save_images(self, folder):
        if not self.save_img:
            return
        image_types = ['degraded', 'clean', 'restored']
        for type_, images in zip(image_types, [self.model.degraded_image, self.model.clean_image, self.model.restored_image]):
            for image, file_name in zip(images, self.model.file_name):
                image = TF.to_pil_image(image.cpu().detach())
                save_path = os.path.join(folder, type_)
                os.makedirs(save_path, exist_ok=True)
                image.save(os.path.join(save_path, file_name))