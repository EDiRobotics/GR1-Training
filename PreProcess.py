from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import Resize

def RandomShiftsAug(x, pad):
    x = x.float()
    b, t, c, h, w = x.size()
    assert h == w
    x = x.view(b*t, c, h, w)  # reshape x to [B*T, C, H, W]
    padding = tuple([pad] * 4)
    x = F.pad(x, padding, "replicate")
    h_pad, w_pad = h + 2*pad, w + 2*pad  # calculate the height and width after padding
    eps = 1.0 / (h_pad)
    arange = torch.linspace(-1.0 + eps, 1.0 - eps, h_pad, device=x.device, dtype=x.dtype)[:h]
    arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
    base_grid = base_grid.unsqueeze(0).repeat(b*t, 1, 1, 1)

    shift = torch.randint(0, 2 * pad + 1, size=(b, 1, 1, 1, 2), device=x.device, dtype=x.dtype)
    shift = shift.repeat(1, t, 1, 1, 1)  # repeat the shift for each image in the sequence
    shift = shift.view(b*t, 1, 1, 2)  # reshape shift to match the size of base_grid
    shift *= 2.0 / (h_pad)

    grid = base_grid + shift
    output = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
    output = output.view(b, t, c, h, w)  # reshape output back to [B, T, C, H, W]
    return output

class PreProcess(): 
    def __init__(
            self,
            rgb_static_pad,
            rgb_gripper_pad,
            rgb_shape, 
            rgb_mean, 
            rgb_std, 
            device,
        ):
        self.resize = Resize([rgb_shape, rgb_shape], interpolation=Image.BICUBIC, antialias=True).to(device)
        self.rgb_static_pad = rgb_static_pad
        self.rgb_gripper_pad = rgb_gripper_pad
        self.rgb_mean = torch.tensor(rgb_mean, device=device).view(1, 1, -1, 1, 1)
        self.rgb_std = torch.tensor(rgb_std, device=device).view(1, 1, -1, 1, 1)
    
    def rgb_process(self, rgb_static, rgb_gripper, train=False):
        rgb_static = rgb_static.float()*(1/255.)
        rgb_gripper = rgb_gripper.float()*(1/255.)
        if train:
            rgb_static = RandomShiftsAug(rgb_static, self.rgb_static_pad)
            rgb_gripper = RandomShiftsAug(rgb_gripper, self.rgb_gripper_pad)
        rgb_static = self.resize(rgb_static)
        rgb_gripper = self.resize(rgb_gripper)
        # torchvision Normalize forces sync between CPU and GPU, so we use our own
        rgb_static = (rgb_static - self.rgb_mean) / (self.rgb_std + 1e-6)
        rgb_gripper = (rgb_gripper - self.rgb_mean) / (self.rgb_std + 1e-6)
        return rgb_static, rgb_gripper

    def rgb_back_process(self, rgb_static, rgb_gripper):
        rgb_static = rgb_static * (self.rgb_std + 1e-6) + self.rgb_mean
        rgb_gripper = rgb_gripper * (self.rgb_std + 1e-6) + self.rgb_mean
        rgb_static = torch.clamp(rgb_static, 0, 1)
        rgb_gripper = torch.clamp(rgb_gripper, 0, 1)
        rgb_static *= 255.
        rgb_gripper *= 255.
        return rgb_static, rgb_gripper
