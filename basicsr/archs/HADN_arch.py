from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import basicsr.archs.Upsamplers as Upsamplers
from basicsr.utils.registry import ARCH_REGISTRY
import sys
sys.path.append("/home/ubuntu/Project/HADN-main")
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

class BSConvU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.pw=torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        )

        # depthwise
        self.dw = torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    eps = 1e-7
    F_variance = (F - F_mean+eps).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
 
class CrossCCA(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(CrossCCA, self).__init__()
    
        self.conv3_1_A = nn.Conv2d(out_channels, out_channels, (3, 1), padding=(1, 0))
        self.conv3_1_B = nn.Conv2d(out_channels, out_channels, (1, 3), padding=(0, 1))
        self.cca =CCALayer(out_channels)
        self.depthwise = nn.Conv2d(out_channels, out_channels, 5, padding=2, groups=out_channels)
        self.depthwise_dilated = nn.Conv2d(out_channels, out_channels, 5,stride=1,padding=6, groups=out_channels,dilation=3)
        self.conv = nn.Conv2d(out_channels , out_channels, 1, padding=0)
        self.active = nn.Sigmoid()
    def forward(self, input):
        x = self.conv3_1_A(input) + self.conv3_1_B(input) 
        x_cca = self.cca(x)
        x_de = self.depthwise(x_cca+input)
        x_de = self.depthwise_dilated(x_de)
        x_fea = x_de + x
        x_fea = self.active(self.conv(x_fea))
        
        return x_fea * input

class ESDB(nn.Module):
    def __init__(self, in_channels, out_channels, conv=nn.Conv2d, p=0.25):
        super(ESDB, self).__init__()
        kwargs = {'padding': 1}

        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels

        self.c1_d = nn.Conv2d(in_channels*2, self.rc, 1,groups = 2)
        self.c1_r = conv(in_channels, self.rc, kernel_size=3,  **kwargs)
        
        self.c2_d = nn.Conv2d(self.remaining_channels*2, self.rc, 1,groups = 2)
        self.c2_r = conv(self.remaining_channels, self.rc, kernel_size=3, **kwargs)
        
        self.c3_d = nn.Conv2d(self.remaining_channels*2, self.rc, 1,groups = 2)
        self.c3_r = conv(self.remaining_channels, self.rc, kernel_size=3, **kwargs)

        self.c4 = conv(self.remaining_channels, self.rc, kernel_size=3, **kwargs)
        self.act = nn.GELU()

        self.CrossCCA = CrossCCA(in_channels,out_channels)

    def forward(self, input):
        r_c1 = (self.c1_r(input))  
        r_c1 = self.act(r_c1 + input)
        r_c1_cat = self.c1_d(torch.cat([input,r_c1],dim=1))

        r_c2 = (self.c2_r(r_c1_cat))
        r_c2 = self.act(r_c2 + r_c1_cat)
        r_c2_cat = self.c2_d(torch.cat([r_c1_cat,r_c2],dim=1 ))

        r_c3 = (self.c3_r(r_c2_cat))
        r_c3 = self.act(r_c3 + r_c2_cat)
        r_c3_cat = self.c3_d(torch.cat([r_c2_cat,r_c3],dim=1))
        r_c4 = self.c4(r_c3_cat)

        out_fused = self.CrossCCA(r_c4)

        return out_fused + input


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


@ARCH_REGISTRY.register()
class HADN(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64, num_block=8, num_out_ch=3, upscale=4,
                 conv='BSConvU', upsampler='pixelshuffledirect', p=0.25):
        super(HADN, self).__init__()
        kwargs = {'padding': 1}
        if conv == 'BSConvU':
            self.conv = BSConvU
        else:
            self.conv = nn.Conv2d
        self.fea_conv = self.conv(num_in_ch * 4, num_feat, kernel_size=3, **kwargs)

        self.B1 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B2 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B3 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B4 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B5 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B6 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B7 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B8 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)

        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1)
        self.GELU = nn.GELU()
        self.up = upscale

        self.c2 = self.conv(num_feat, num_feat, kernel_size=3, **kwargs)
        if upsampler == 'pixelshuffledirect':
            self.upsampler = Upsamplers.PixelShuffleDirect(scale=upscale, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'pixelshuffleblock':
            self.upsampler = Upsamplers.PixelShuffleBlcok(in_feat=num_feat, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'nearestconv':
            self.upsampler = Upsamplers.NearestConv(in_ch=num_feat, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'pa':
            self.upsampler = Upsamplers.PA_UP(nf=num_feat, unf=24, out_nc=num_out_ch)
        else:
            raise NotImplementedError(("Check the Upsampeler. None or not support yet"))

    def forward(self, input):
        input_cat = torch.cat([input, input, input, input], dim=1)
        out_fea = self.fea_conv(input_cat)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B7 = self.B7(out_B6)
        out_B8 = self.B8(out_B7)

        trunk = torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8], dim=1)

        out_B = self.c1(trunk)
        
        out_B = self.GELU(out_B)

        out_lr = self.c2(out_B) + out_fea

        input_up=F.interpolate(input, (input.size(2)*self.up, input.size(3)*self.up), mode='bilinear', align_corners=False)

        output = self.upsampler(out_lr)

        return output+input_up

 
 
