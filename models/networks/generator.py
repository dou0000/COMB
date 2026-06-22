import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from typing import Dict
from collections import OrderedDict
from torchvision.models import resnet50
from typing import List, Type
from functools import partial

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
BOTTLENECK_DIM = 2048         

# ------------------------------------------------------------------------------
# Base models
# ------------------------------------------------------------------------------
def make_n_channel_stem(backbone: nn.Module, n: int = 5):
    old_conv = backbone.conv1                              # [64, 3, 7, 7]
    new_conv = nn.Conv2d(n, 64, kernel_size=7,
                         stride=2, padding=3, bias=False)   # [64, n, 7, 7]

    with torch.no_grad():
        # (1) random-initialise everything
        nn.init.kaiming_normal_(new_conv.weight,
                                mode='fan_in', nonlinearity='relu')

        # (2) overwrite the LAST three channels with pretrained RGB kernels
        new_conv.weight[:, -3:] = old_conv.weight

        # (3) optional: scale weights so fan-in statistics match the 3-ch model
        new_conv.weight.mul_(3.0 / n)

    backbone.conv1 = new_conv
    return backbone

# ------------------------------------------------------------------------------
class ChannelSqueeze(nn.Module):
    def __init__(self, in_channels=2048, out_channels=BOTTLENECK_DIM):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        return self.norm(self.proj(x))


# ------------------------------------------------------------------------------
class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model: nn.Module, return_layers: Dict[str, str]):
        names = {name for name, _ in model.named_children()}
        if not set(return_layers).issubset(names):
            raise ValueError("return_layers are not present in model")
        orig = return_layers.copy()
        work = return_layers.copy()

        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in work:
                del work[name]
            if not work:
                break
        super().__init__(layers)
        self.return_layers = orig

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out[self.return_layers[name]] = x
        return x, out

# ------------------------------------------------------------------------------
class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)


class ChannelAttention(nn.Module):
    def __init__(self, in_ch, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(Flatten(), nn.Linear(in_ch, in_ch // r),
                                 nn.ReLU(), nn.Linear(in_ch // r, in_ch))

    def forward(self, x):
        s = torch.sigmoid(self.mlp(self.avg(x)) + self.mlp(self.max(x)))
        return x * s.unsqueeze(2).unsqueeze(3)


class SpatialAttention(nn.Module):
    def __init__(self):#, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.bn   = nn.BatchNorm2d(1)
        # self.bn = norm_layer(1)

    def forward(self, x):
        s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        s = torch.sigmoid(self.bn(self.conv(s)))
        return x * s


class CBAM(nn.Module):
    def __init__(self, in_ch, r=16):
        super().__init__()
        self.ca = ChannelAttention(in_ch, r)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer=nn.BatchNorm2d):
        super().__init__()
        act = nn.ReLU(True)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            norm_layer(out_ch), act
        )

        self.conv = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 3, 1, 1), norm_layer(out_ch), act,
            nn.Conv2d(out_ch, out_ch, 3, 1, 1), norm_layer(out_ch), act)

    def forward(self, x1, x2):
        return self.conv(torch.cat([x2, self.up(x1)], 1))



BOTTLENECK_DIM = 2048  # ResNet50 bottleneck dimension

def _conv3x3(in_c, out_c, stride=1, groups=1, padding=1):
    return nn.Conv2d(in_c, out_c, 3, stride, padding, groups=groups, bias=False)

def _conv1x1(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 1, stride, bias=False)



class _Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d):
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = _conv1x1(inplanes, width)
        self.bn1   = norm_layer(width)
        self.conv2 = _conv3x3(width, width, stride, groups, dilation)
        self.bn2   = norm_layer(width)
        self.conv3 = _conv1x1(width, planes * self.expansion)
        self.bn3   = norm_layer(planes * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):      # <── accept kw
        identity = x
        # out = _call_with_anchor(self.conv1, x, anchor_kw)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


class ResNetBackbone(nn.Module):
    """ResNet-50 feature extractor (conv→layer4)."""
    def __init__(self, layers: List[int] = [3, 4, 6, 3],
                 in_ch: int = 13,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d):
        super().__init__()
        self.inplanes, self.dilation = 64, 1
        self.groups, self.base_width = 1, 64
        
        # stem
        self.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        self.bn1   = norm_layer(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        # stages
        self.layer1 = self._make_layer(_Bottleneck,  64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(_Bottleneck, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(_Bottleneck, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(_Bottleneck, 512, layers[3], stride=2, norm_layer=norm_layer)
        # Kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
                if getattr(m, "weight", None) is not None:
                    nn.init.constant_(m.weight, 1)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)
        # zero-init residual BN3 for better training stability
        for m in self.modules():
            if isinstance(m, _Bottleneck):
                w = getattr(m.bn3, "weight", None)
                if w is not None:
                    nn.init.constant_(w, 0)

    # ------------------------------------------------------------------
    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample,
                        self.groups, self.base_width, self.dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return x

def make_n_channel_stem(backbone: nn.Module, n: int = 5):
    old_conv = backbone.conv1                                     # [64, 3, 7, 7]
    new_conv = nn.Conv2d(n, 64, 7, 2, 3, bias=False)              # [64, n, 7, 7]
    
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        mean_extra = old_conv.weight.mean(1, keepdim=True)
        new_conv.weight[:, 3:] = mean_extra.repeat(1, n - 3, 1, 1)

    backbone.conv1 = new_conv
    return backbone


# ------------------------------------------------------------------------------
class AttUnetGenerator(BaseNetwork):
    def __init__(self, opt, norm_layer=nn.BatchNorm2d, fpn_feature='decoder', num_channels=10):
        super().__init__()
        self.num_channels = num_channels

        if hasattr(opt, 'dataset_mode'):
            if opt.dataset_mode == 'hemit':
                self.num_channels = 3


        if hasattr(opt, 'dataset_mode'):
            if opt.dataset_mode == 'SRS':
                self.num_channels = 5
                print(f"Dataset mode: SRS detected. Setting num_channels to {self.num_channels}.")


        # # if opt contains 'IR_5' then set num_channels to 5 else 10
        # if hasattr(opt, 'IR_5'):
        #     self.num_channels = 5 if opt.IR_5 else 10
        # else:
        #     self.num_channels = num_channels

        if hasattr(opt, 'norm_layer'):
            print(f'Using specified normalization layer: {opt.norm_layer}')
            if opt.norm_layer == 'batch':
                norm_layer = nn.BatchNorm2d
            elif opt.norm_layer == 'instance':
                # norm_layer = partial(nn.InstanceNorm2d
                norm_layer = nn.InstanceNorm2d
            else:
                raise ValueError(f"Unsupported norm_layer: {opt.norm_layer}")
        
        else:
            print(f'Using default normalization layer: {norm_layer}')
            norm_layer = norm_layer
        
        backbone = ResNetBackbone(in_ch=self.num_channels, norm_layer=norm_layer)

        # if norm_layer == nn.InstanceNorm2d:
        #     backbone = ResNetBackbone(in_ch=self.num_channels, norm_layer=norm_layer)
        # elif norm_layer == nn.BatchNorm2d:

        # backbone = make_n_channel_stem(resnet50(pretrained=True), n=self.num_channels)
        # 
        self.encoder = IntermediateLayerGetter(
            backbone,
            {"conv1": 'feat', "layer1": 'feat0', "layer2": 'feat1',
             "layer3": 'feat2', "layer4": 'feat3'}
        )

        self.fpn_feature = fpn_feature
        self.ngf = 64
        self.n_down = 5
        self.out_nc = 3
        for i in range(self.n_down - 1):
            mult = 2 ** (self.n_down - i)
            setattr(self, f'up{i}', Up(self.ngf * mult,
                                        self.ngf * mult // 2 if i != self.n_down - 2 else self.ngf,
                                        norm_layer=norm_layer))
            setattr(self, f'cbam{i}', CBAM(self.ngf * mult))#, norm_layer=norm_layer))
        setattr(self, f'cbam{self.n_down - 1}', CBAM(self.ngf))#, norm_layer=norm_layer))

    
        # replace all BatchNorm2d layers with InstanceNorm2d
        # self.encoder = bn_to_in(self.encoder, affine=True)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(self.ngf, self.ngf // 2, 4, 2, 1),
            norm_layer(self.ngf // 2), nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(self.ngf // 2, self.out_nc, 7, 1, 0),
            nn.Tanh()
        )


    def forward(self, x, *, return_feats=False,
            encode_only=False, layers=(0,1,2,3,4), patch_ids=None):

        feats = []

        # ----- conv1 -----
        f0 = self.encoder.conv1(x)          # stride=2
        if 0 in layers:
            feats.append(f0)                # (B,64,H/2,W/2)

        # ----- BN + ReLU + pool -----
        f = self.encoder.bn1(f0)
        f = self.encoder.relu(f)
        f = self.encoder.maxpool(f)         # now H/4 × W/4

        # ----- residual blocks -----
        for idx, block in enumerate((self.encoder.layer1,
                                    self.encoder.layer2,
                                    self.encoder.layer3,
                                    self.encoder.layer4), start=1):
            f = block(f)
            if idx in layers:
                feats.append(f)

        if encode_only:                     # PatchNCE path
            return [ft for idx, ft in enumerate(feats) if idx in layers]

        # ----- decoder path (unchanged) -----
        deep  = f
        feats_full = [deep] + feats[::-1][1:]
        x = getattr(self, 'cbam0')(feats_full[0])
        for i in range(self.n_down - 1):
            x = getattr(self, f'up{i}')(x,
                getattr(self, f'cbam{i+1}')(feats_full[i+1]))

        out_img = self.final_up(x)
        return (out_img, feats) if return_feats else out_img
    