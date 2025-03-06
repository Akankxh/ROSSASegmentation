import os 
import sys 
import math 

import numpy as np 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import torchvision.transforms.functional as TF 

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__() 

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False) 
        self.bn1   = nn.BatchNorm2d(num_features=out_channels) 
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(num_features=out_channels)
        self.relu  = nn.ReLU(inplace=True) 

        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                               padding=0, bias=False) 
        else:
            self.proj = None 

    def forward(self, x):
        identity = x 
        
        out = self.conv1(x) 
        out = self.bn1(out) 
        out = self.relu(out) 

        out = self.conv2(out) 
        if self.proj:
            identity = self.proj(identity) 
        
        out += identity 
        out = self.bn2(out) 
        out = self.relu(out) 
        return out 

class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__() 

        self.cSE = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels//reduction, kernel_size=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels //reduction, out_channels=in_channels, kernel_size=1), 
            nn.Sigmoid()
        ) 

        self.SSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid()) 
    
    def forward(self, x):
        out = x* self.cSE(x) + x* self.SSE(x) 
        return out 

class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_layers, skip_channel=None, attn=None):
        super().__init__()

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)  
        self.attn = attn 

        if skip_channel:
            in_channel = in_channel + skip_channel 
        else:
            in_channel = in_channel
        self.block  = nn.Sequential(ConvLayer(in_channels=in_channel, out_channels=out_channel),
                                    *[ConvLayer(in_channels=out_channel, out_channels=out_channel)
                                       for _ in range(num_layers)]) 
        

        if self.attn:
            self.attn_model = SCSEModule(in_channels= in_channel) 
    
    def forward(self, out, skip=None):
        if skip is not None:
            if skip.shape != out.shape: 
                out = TF.resize(out, skip.shape[2:]) 
            
            out = torch.cat([out, skip], dim=1)

        if self.attn:
            out = self.attn_model(out) 
        
        out = self.block(out) 
        return out 