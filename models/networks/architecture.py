"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import torchvision
from models.networks.normalization import SPADE
from torch.nn import Parameter as P

# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        # Attributes
        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, opt.semantic_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, opt.semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, opt.semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        return x_s + dx

    def shortcut(self, x, seg):
        return self.conv_s(self.norm_s(x, seg)) if self.learned_shortcut else x

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


# ResNet block used in pix2pixHD
# We keep the same architecture as pix2pixHD.
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)),
        )

    def forward(self, x):
        y = self.conv_block(x)
        return x + y

# In-house feature extractor; by default, always pretrained & after relu
class VGG19(torch.nn.Module):
    def __init__(self, opt=None, pretrained=True, freeze_weights=True, before_relu=False):
        super().__init__()
        self.opt = opt

        import timm
        pretrained_model = timm.create_model('timm/convnext_tiny.in12k_ft_in1k')
        pretrained_model.head.fc = nn.Identity()
        pretrained_model.load_state_dict(torch.load(rf'tcga_tl_checkpoints/convnext_t_dp_spv_supcon_1e-5'))  # modify appropriately
        self.slice1 = torch.nn.Sequential(pretrained_model.stem)
        self.slice2 = torch.nn.Sequential(pretrained_model.stages[0])
        self.slice3 = torch.nn.Sequential(pretrained_model.stages[1])
        self.slice4 = torch.nn.Sequential(pretrained_model.stages[2])
        self.slice5 = torch.nn.Sequential(pretrained_model.stages[3])

        if freeze_weights:
            for param in self.parameters():
                param.requires_grad = False

        self.mean = P(
            torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1), requires_grad=False
        )
        self.std = P(
            torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1), requires_grad=False
        )

    def forward(self, x_org):
        if x_org.shape[-1] >= 5012:
            pad_height = ((x_org.shape[-2] + 223) // 224) * 224 - x_org.shape[-2]
            pad_width = ((x_org.shape[-1] + 223) // 224) * 224 - x_org.shape[-1]
            x_org_padded = torch.nn.functional.pad(x_org, (0, pad_width, 0, pad_height))
            patch_list = []
            i = 0
            while i + 224 <= x_org_padded.shape[-2]:
                j = 0
                while j + 224 <= x_org_padded.shape[-1]:
                    patch_list += [x_org_padded[:, :, i:i+224, j:j+224]]
                    j += 224
                i += 224
            # Add the downsampled x_org to the patch_list
            x_resized = F.interpolate(x_org, size=(224, 224), mode='bilinear', align_corners=True)
            patch_list += [x_resized]
            x = torch.cat(patch_list, dim=0)
        else:
            x = x_org

        
        if self.opt.vgg_normalize_input:
            x = x/2 + .5
            x = (x - self.mean) / self.std
       # if (x.shape[2] != 224 or x.shape[3] != 224) and self.opt.vgg_resize_input:
        #    x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]