import os 
import sys 
import math 

import numpy as np 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from Model.ResUNETutils import BottleNecklayer, Decoderblock 

class Encoder(nn.Module):
    def __init__(self, in_channels, num_layers):
        super().__init__() 

        self.block1 = nn.Sequential(BottleNecklayer(in_channels=in_channels, out_channels=16),
                                    *[BottleNecklayer(in_channels=16, out_channels=16) for _ in range(num_layers)]) 
       
        self.block2 = nn.Sequential(BottleNecklayer(in_channels=16, out_channels=32),
                                    *[BottleNecklayer(in_channels=32, out_channels=32) for _ in range(num_layers)])
      
        self.block3 = nn.Sequential(BottleNecklayer(in_channels=32, out_channels=64),
                                    *[BottleNecklayer(in_channels=64, out_channels=64) for _ in range(num_layers)])

        self.block4 = nn.Sequential(BottleNecklayer(in_channels=64, out_channels=96),
                                    *[BottleNecklayer(in_channels=96, out_channels=96) for _ in range(num_layers)])
        
        self.block5 = nn.Sequential(BottleNecklayer(in_channels=96, out_channels=128),
                                    *[BottleNecklayer(in_channels=128, out_channels=128) for _ in range(num_layers)])
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        features = [ ] 

        out = self.block1(x) 
        features.append(out) 
        out = self.pool(out) 

        out = self.block2(out) 
        features.append(out) 
        out = self.pool(out)

        out = self.block3(out) 
        features.append(out) 
        out = self.pool(out)

        out = self.block4(out) 
        features.append(out) 
        out = self.pool(out)

        out = self.block5(out) 
        return features, out 

class Decoder(nn.Module):
    def __init__(self, num_layers, num_classes):
        super().__init__()

        self.block1 = Decoderblock(in_channels=128, out_channels=96, skip_channels=96, num_layers=num_layers) 
        self.block2 = Decoderblock(in_channels=96, skip_channels=64, out_channels=64, num_layers=num_layers) 
        self.block3 = Decoderblock(in_channels=64, out_channels=32, skip_channels=32, num_layers=num_layers) 
        self.block4 = Decoderblock(in_channels=32, skip_channels=16, out_channels=16, num_layers=num_layers) 
        self.block5 = Decoderblock(in_channels=16, out_channels=16, skip_channels=None, num_layers=num_layers) 

        self.output = nn.Conv2d(in_channels=16, out_channels=num_classes, kernel_size=1)
    
    def forward(self, out, features):
        skip = features.pop() 
        out = self.block1(out, skip) 

        skip = features.pop() 
        out  = self.block2(out, skip) 

        skip = features.pop() 
        out = self.block3(out, skip) 

        skip = features.pop() 
        out  = self.block4(out, skip) 
        
        out = self.block5(out, skip=None) 
        out = self.output(out) 
        return out 

class ResUNeT(nn.Module):
    def __init__(self, in_channels, num_classes, num_layers=2):
        super().__init__() 
        self.enc = Encoder(in_channels=in_channels, num_layers=num_layers) 
        self.dec = Decoder(num_classes=num_classes, num_layers=num_layers) 
    
    def forward(self, x):
        features, out = self.enc(x) 
        out = self.dec(out, features) 
        return out 
    

if __name__ == '__main__':
    os.system('cls') 

    # enc=  Encoder(in_channels=1, num_layers=2) 
    # x = torch.randn(10, 1, 224, 224) 
    # features, out = enc(x) 
    
    # dec = Decoder(num_classes=1, num_layers=2) 
    # out = dec(out, features) 
    # print(out.shape) 

    model = ResUNeT(in_channels=1, num_classes=1, num_layers=2) 
    x  =torch.randn(10, 1, 224, 224) 
    out = model(x) 
    print(out.shape)
        


# import torch
# import torch.nn as nn

# class batchnorm_relu(nn.Module):
#     def __init__(self,in_c):
#         super().__init__()

#         self.bn = nn.BatchNorm2d(in_c)
#         self.relu = nn.ReLU()

#     def forward(self,inputs):
#         x = self.bn(inputs)
#         x = self.relu(x)
#         return x
    
# class residual_block(nn.Module):
#     def __init__(self,in_c,out_c,stride=1):
#         super().__init__()

#         #Convolutional Layer
#         self.b1 = batchnorm_relu(in_c)
#         self.c1 = nn.Conv2d(in_c,out_c,kernel_size=3,padding=1,stride=stride)
#         self.b2 = batchnorm_relu(out_c)
#         self.c2 = nn.Conv2d(out_c,out_c,kernel_size=3,padding=1,stride=1)

#         #Shortcut Connection
#         self.s = nn.Conv2d(in_c,out_c,kernel_size=1,padding=0,stride=stride)

#     def forward(self,inputs):
#         x= self.b1(inputs)
#         x = self.c1(x)
#         x = self.b2(x)
#         x = self.c2(x)
#         s = self.s(inputs)

#         skip = x+s
#         return skip
    
# class decoder_block(nn.Module):
#     def __init__(self,in_c,out_c):
#         super().__init__()

#         self.upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
#         self.r = residual_block(in_c+out_c,out_c)

#     def forward(self,inputs,skip):
#         x = self.upsample(inputs)
#         x = torch.cat([x,skip],axis=1)
#         x = self.r(x)
#         return x
    
# class build_resunet(nn.Module):
#     def __init__(self,in_ch,num_classes):
#         super().__init__()

#         #Encoder 1
#         self.c11 = nn.Conv2d(in_ch,64,kernel_size=3,padding=1)
#         self.br1 = batchnorm_relu(64)
#         self.c12 = nn.Conv2d(64,64,kernel_size=3,padding=1)
#         self.c13 = nn.Conv2d(in_ch,64,kernel_size=1,padding=0)

#         #encoder 2 and 3
#         self.r2 = residual_block(64,128,stride=2)
#         self.r3 = residual_block(128,256,stride=2)

#         #bridge
#         self.r4 = residual_block(256,512,stride=2)

#         #decoders
#         self.d1 = decoder_block(512,256)
#         self.d2 = decoder_block(256,128)
#         self.d3 = decoder_block(128,64)

#         #output
#         self.output = nn.Conv2d(in_channels=64,out_channels=num_classes,kernel_size=1,padding=0)
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self,inputs):
#         #encoder 1
#         x = self.c11(inputs)
#         x = self.br1(x)
#         x = self.c12(x)
#         s = self.c13(inputs)
#         skip1 = x+s

#         #encoders 2 and 3
#         skip2 = self.r2(skip1)
#         skip3=self.r3(skip2)

#         #bridge or bottleneck

#         b= self.r4(skip3)

#         #decoders
#         d1 = self.d1(b,skip3)
#         d2 = self.d2(d1,skip2)
#         d3 = self.d3(d2,skip1)

#         #output
#         output = self.output(d3)
#         output = self.sigmoid(output)

#         return output
    
# if __name__ == "__main__":
#     inputs = torch.randn((4,1,256,256))
#     model = build_resunet(1,1)
#     y = model(inputs)
#     print(y.shape)