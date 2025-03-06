import os 
import sys 
import math 
import glob 
import cv2 
import albumentations  as A 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder as LabelEncoder 

from imageio.v3 import imread 
from PIL import Image 

import torch
import torch.utils.data as data 

from utils import Data_Augmentation, LabelGeneration

class OCTADataset(data.Dataset):
    def __init__(self, train=True, subfolder = 'train/image', lblenc = None, maskfolder = 'train/label', augmen = None):
        super().__init__()
        self.train = train 
        self.lblenc = lblenc 
        self.augmen = augmen 
        self.subfolder = subfolder
        self.maskfolder = maskfolder

        if self.train:
            self.data = glob.glob(os.path.join('ROSSA','data','train','image','*.png'))
        else:
            self.data = glob.glob(os.path.join('ROSSA', 'data','val','image','*.png')) 


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


if __name__ == '__main__':

    os.system('clear')
    
    subfolder= 'train/image'
    maskfolder = 'train/label'
    dataType = 'train'
    #from utils import LabelGeneration, Data_Augmentation 

    encoder=LabelGeneration(subfolder=maskfolder, path= os.path.join('ROSSA', 'dataset')) 
    Traindata_aug = Data_Augmentation(dataType=dataType)
    Testdata_aug = Data_Augmentation(dataType='test')
    train_dataset = OCTADataset( lblenc=encoder, subfolder=subfolder , train=True, augmen =Traindata_aug)  
    test_dataset  = OCTADataset(lblenc=encoder, subfolder=subfolder , train=False, augmen =Testdata_aug)
    print(len(train_dataset), len(test_dataset),Traindata_aug, Testdata_aug, sep='\n')


   
    for i in range(len(train_dataset)):
        Img, mask = train_dataset[i]

        print(Img.max(), Img.min(), Img.shape, mask.shape, mask.min(), mask.max()) 
        break 

    Img = np.uint8(Img.numpy().squeeze() * 255.0)
    Mask = np.uint8(mask.numpy().squeeze() *255.0)
    print(Img.shape, Mask.shape, Img.max())
    Img = Image.fromarray(Img) 
    print(Img.size, Img.getextrema(), type(Img))
    Mask = Image.fromarray(Mask) 

    Img.show() 
    Mask.show()
