import os 
import sys 
import math 

import numpy as np 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import torchvision.transforms.functional as TF 

class BottleNecklayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__() 

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False) 
        self.bn1   = nn.BatchNorm2d(out_channels) 

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False) 
        self.bn2 = nn.BatchNorm2d(out_channels) 

        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels) 

        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        else:
            self.proj = None 

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x 
        out = self.bn1(self.conv1(x)) 
        out = self.relu(out) 

        out = self.bn2(self.conv2(out)) 
        out = self.relu(out) 

        out = self.conv3(out)

        if self.proj is not None:
            identity = self.proj(identity) 
         
        out += identity 
        out = self.bn3(out) 
        return self.relu(out) 
    
class Decoderblock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, num_layers):
        super().__init__() 

        if skip_channels is not None:
            in_channels = in_channels + skip_channels 
        else:
            in_channels = in_channels 

        self.up = nn.UpsamplingBilinear2d(scale_factor=2) 
        self.block = nn.Sequential(BottleNecklayer(in_channels=in_channels, out_channels=out_channels),
                                   *[BottleNecklayer(in_channels=out_channels, out_channels=out_channels) 
                                     for _ in range(num_layers)]) 
    
    def forward(self, out, skip):
         
        if skip is not None:
            out = self.up(out)
            if skip.shape != out.shape: 
                out = TF.resize(out, skip.shape[2:]) 
            out = torch.cat([out, skip], dim=1) 
        
        out = self.block(out) 
        return out 



# if __name__ == '__main__':

#     os.system('cls') 
#     # model = BottleNecklayer(in_channels=16, out_channels=32) 

#     # x = torch.randn(10, 16, 224, 224) 
#     # out = model(x) 
#     # print(out.shape)

#     dec_model = Decoderblock(in_channels=128, skip_channels=96, out_channels=96, num_layers=2) 
#     inp = torch.randn(10, 128, 14, 14 )
#     skip = torch.randn(10, 96, 28, 28 ) 
#     out = dec_model(inp, skip) 
#     print(out.shape)