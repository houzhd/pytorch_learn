# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:07:07 2023

@author: Administrator
"""

import torch
in_channels, out_channels = 5, 10
width, height = 100, 100
kernel_size = 3
bathch_size = 1 

input = torch.randn(bathch_size, in_channels, width, height)

conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size)

output = conv_layer(input)

print(input.shape)
print(output.shape)
print(conv_layer.weight.shape)