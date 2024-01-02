from thop import profile
import torch
from basicsr.archs import HADN_arch,HADN_S_arch
import sys
import matplotlib.pyplot as plt
from utils.model_summary import get_model_flops, get_model_activation
from utils import utils_image as util
import os
# sys.path.append("/home/ubuntu/Project/HADN-main")
sys.path.append("/Users/yui/Downloads/HADN-main")




upscale = 4
width = (1280 // upscale)
height = (720 // upscale)

# model = HADN_arch.HADN()
model = HADN_S_arch.HADNS()
input = torch.randn(1, 3,height , width)
macs, params = profile(model, inputs=(input, ))
# 测速

print("Multi-adds[G] ")
print(macs/1e9)
print("Parameters [K]")
print(params/1e3)

from utils.model_summary import get_model_activation,get_model_flops
# from thop import profile

input_dim = (3,height , width)  # set the input dimension
activations, num_conv = get_model_activation(model, input_dim)
activations = activations / 10 ** 6
print("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
print("{:>16s} : {:<d}".format("#Conv2d", num_conv))

flops = get_model_flops(model, input_dim, False)
flops = flops / 10 ** 9
print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
num_parameters = num_parameters / 10 ** 6
print("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))


