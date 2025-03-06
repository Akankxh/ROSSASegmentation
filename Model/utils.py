import os 
import sys 
import math 

import numpy as np 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision.transforms as transforms

'''class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1,):
        super().__init__() 
        
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True) 
    
    def forward(self, x):

        out = self.conv(x)
        out = self.bn(out) 
        return self.relu(out) 



if __name__ == '__main__':
    os.system('clear')

    conv_block = ConvBlock(in_channel=3, out_channel=16)
    print(conv_block) 

    conv_block1 = nn.Sequential(ConvBlock(in_channel=1, out_channel=64), 
                                ConvBlock(in_channel=64, out_channel=64)) 
    
    print(conv_block1)'''


class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)
    
class Encoder(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels,out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self,x):
        down = self.conv(x)
        p = self.pool(down)

        return down,p
    
class Decoder(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size=2,stride=2)
        self.conv = DoubleConv(in_channels,out_channels)

    def forward(self,x1,x2):
        x1 = self.up(x1)
        x = torch.cat([x1,x2],1)
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self,in_channels,num_classes):
        super().__init__()
        self.down_conv1 = Encoder(in_channels, 64)
        self.down_conv2 = Encoder(64,128)
        self.down_conv3 = Encoder(128,256)
        self.down_conv4 = Encoder(256,512)

        self.bottleneck = DoubleConv(512,1024)

        self.up_conv1 = Decoder(1024,512)
        self.up_conv2 = Decoder(512,256)
        self.up_conv3 = Decoder(256,128)
        self.up_conv4 = Decoder(128,64)

        self.out = nn.Conv2d(in_channels=64,out_channels=num_classes,kernel_size=1)

    def forward(self,x):
        down_1, p_1 = self.down_conv1(x)
        down_2, p_2 = self.down_conv2(p_1)
        down_3, p_3 = self.down_conv3(p_2)
        down_4, p_4 = self.down_conv4(p_3)

        b = self.bottleneck(p_4)

        up_1 = self.up_conv1(b,down_4)
        up_2 = self.up_conv2(up_1,down_3)
        up_3 = self.up_conv3(up_2,down_2)
        up_4 = self.up_conv4(up_3,down_1)

        out = self.out(up_4)

        return out


if __name__ == '__main__':
    os.system('clear')

    input_image = torch.rand((1,3,512,512))
    model = UNet(3,2)
    output = model(input_image)
    print(output.size())