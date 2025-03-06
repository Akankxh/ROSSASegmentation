import os 
import sys 
import math 

import cv2 
import glob 


import numpy as np 

from imageio.v3 import imread 
from PIL import Image 

import torch
import torch.utils.data as data 

class OCTADataset(data.Dataset):
    def __init__(self, train=True,  lblenc = None,  augmen = None):
        super().__init__()
        self.train = train 
        self.lblenc = lblenc 
        self.augmen = augmen 
        # self.subfolder = subfolder
        # 

        if self.train:
            self.data = glob.glob(os.path.join('ROSSA','data','train','image','*.png'))
            self.subfolder = os.path.join('train','image')
            self.maskfolder = os.path.join('train','label')
        else:
            self.data = glob.glob(os.path.join('ROSSA', 'data','val','image','*.png')) 
            self.subfolder = os.path.join('val','image')
            self.maskfolder = os.path.join('val','label')


    def __len__(self):
        return len(self.data) 
    
    def Readimage(self, imageName):
        image = np.array(imread(imageName)).astype(np.float32)
        return image
    
    def PreprocessImage(self, image):
        image = np.float32(image) / 255.0
        return np.expand_dims(image,-1)

    def ReadMask(self, maskName):
        image = imread(maskName) 
        h, w = image.shape 
        image = image.ravel() 
        image = self.lblenc.transform(image)
        image = image.reshape((h, w)) 
        return image

    def __getitem__(self,id):
        imageName  = self.data[id]
        maskName   = imageName.replace(self.subfolder, self.maskfolder) 
        
        image = self.Readimage(imageName)
        mask = self.ReadMask(maskName)
        
        if self.augmen: 
            aug = self.augmen(image=image, mask=mask)
            Img = aug['image']
            mask = aug['mask']
        else:
            Img  = image 
            mask = mask 
        
        Img = self.PreprocessImage(Img)
        Img = Img.transpose(2, 0, 1)
        mask =np.expand_dims( mask, axis=0)
        Img = torch.from_numpy(Img).float() 
        mask = torch.from_numpy(mask).long() 
        return Img, mask 