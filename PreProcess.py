from PIL import Image
import torch
from torchvision.transforms.v2 import Resize, RandomResizedCrop 

class PreProcess(): 
    def __init__(
            self,
            rgb_shape, 
            rgb_mean, 
            rgb_std, 
            crop_area_scale,
            crop_aspect_ratio,
            device,
        ):
        self.train_transforms = torch.nn.Sequential(
            RandomResizedCrop(rgb_shape, crop_area_scale, crop_aspect_ratio, interpolation=Image.BICUBIC, antialias=True),
        ).to(device)
        self.test_transforms = torch.nn.Sequential(
            Resize(rgb_shape, interpolation=Image.BICUBIC, antialias=True),
        ).to(device)
        self.rgb_mean = torch.tensor(rgb_mean, device=device).view(1, 1, -1, 1, 1)
        self.rgb_std = torch.tensor(rgb_std, device=device).view(1, 1, -1, 1, 1)
    
    def rgb_process(self, rgb_static, rgb_gripper, train=False):
        rgb_static = rgb_static.float()*(1/255.)
        rgb_gripper = rgb_gripper.float()*(1/255.)
        if train:
            rgb_static = self.train_transforms(rgb_static)
            rgb_gripper = self.train_transforms(rgb_gripper)
        else:
            rgb_static = self.test_transforms(rgb_static)
            rgb_gripper = self.test_transforms(rgb_gripper)
        # torchvision Normalize forces sync between CPU and GPU, so we use our own
        rgb_static = (rgb_static - self.rgb_mean) / (self.rgb_std + 1e-6)
        rgb_gripper = (rgb_gripper - self.rgb_mean) / (self.rgb_std + 1e-6)
        return rgb_static, rgb_gripper
