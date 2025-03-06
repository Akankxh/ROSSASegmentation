import os 
import sys 
import math 

import glob 

import numpy as np
import albumentations as A 
import matplotlib.pyplot as plt 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 


from imageio.v3 import imread
from sklearn.preprocessing import LabelEncoder



def Data_Augmentation(dataType='train'):
    if dataType.lower() == 'train':
        transform = A.Compose([ 
                              A.HorizontalFlip(p=0.5), 
                              A.VerticalFlip(p=0.5), 
                              A.RandomBrightnessContrast(p=0.1, brightness_limit=0.1, contrast_limit=0.1), 
                              A.RandomGamma(p=0.5),
                              A.Rotate(p=0.5,limit=45), 
                            #   A.MotionBlur(p=0.5), 
                            #   A.GaussianBlur(p=0.5),
                            #   A.GlassBlur(),
                              #A.GaussNoise(p=0.5), 
                              A.ElasticTransform(p=0.5),
                              A.CoarseDropout(p=0.5, max_holes=20, min_holes=5) 
                                       ])
    else:
        transform = None 
    
    return transform 

def LabelGeneration(path = './ROSSA/data', subfolder ='train/label'):
    path = os.path.join(path, subfolder) 
    imageNames = glob.glob(os.path.join(path, '*.png'))
    print(path)
    Images = [imread(imageName) for imageName in imageNames] 
    Images = np.array(Images) 
   
    #print(type(Images), Images.shape)
    
    n, h, w = Images.shape
    encoder = LabelEncoder() 
    encoder.fit(Images.ravel())
    return encoder 

def Save_Model(model, optimizer, epoch,dice_Score=None, maskfolder='train/label', path='model_wts'):

    state = { 'epoch': epoch,'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),'dice_score':dice_Score}
    if not os.path.exists(path=path):
        os.mkdir(path=path) 

    save_name = 'best_f_wts'+maskfolder+'.pth' 
    savepath=os.path.join(path, save_name)
    torch.save(state,savepath)

def Load_model(model, optimizer, maskfolder='train/label', path='model_wts'):
    save_name = 'best_f_wts'+maskfolder+'.pth'
    savepath=os.path.join(path, save_name)
    checkpoint = torch.load(savepath)
    model.load_state_dict(checkpoint['state_dict']) 
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    score = checkpoint['dice_score']
    print(f'Model has sucessfully loaded')

    return model, optimizer, epoch, score
