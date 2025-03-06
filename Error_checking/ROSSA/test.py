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

import torch
import torch.utils.data as data 

def LabelGeneration(path = 'dataset/train', subfolder ='label'):
    path = os.path.join(path, subfolder) 
    print(path)
    imageNames = glob.glob(os.path.join(path, '*.png'))
    Images = [imread(imageName) for imageName in imageNames] 
    Images = np.array(Images) 
    print(type(Images), Images.shape)
    n, h, w = Images.shape
    print(n,h,w)
    '''
    n, h, w = Images.shape
    encoder = LabelEncoder() 
    encoder.fit(Images.ravel())
    return encoder 
    '''
LabelGeneration()