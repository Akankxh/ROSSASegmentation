import os 
import sys 
import math 

import numpy as np 

from tqdm import tqdm  

import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.utils.data as data 

from Loss.dice import DiceLoss 
from Loss.focal import FocalLoss 
# #from Loss.cosine import CosineLoss 

from Model.utils import UNet
from dataset import OCTADataset
from TrainTest import train_test
from utils import Data_Augmentation, LabelGeneration, Load_model 


def main(lr=None, num_classes=None, in_ch=None, batch_size=None, activation=None, preprocessing =None, device = None,
         subfolder=None, maskfolder=None, num_epochs=None): 
    train_transforms = Data_Augmentation(dataType='train') 
    valid_transform  = Data_Augmentation(dataType='valid')
    lblenc = LabelGeneration(subfolder=maskfolder)
    print(lblenc.classes_) 


    train_dataset = OCTADataset(train=True, lblenc=lblenc, augmen=train_transforms, subfolder=subfolder, 
                                 maskfolder=maskfolder)
    valid_dataset = OCTADataset(train=False, lblenc=lblenc, augmen=valid_transform, subfolder=subfolder,
                                 maskfolder=maskfolder)
    print(f'length of train dataset {len(train_dataset)} and test dataset {len(valid_dataset)}')

    # for x, y in valid_dataset:
    #     print(x.shape, y.shape) 
    #     break 

    train_loder = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2) 
    valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = UNet(in_ch=in_ch, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr) 
    dice_loss = DiceLoss()#, classes=np.array([0.25, 0.75]))
    focal_loss = FocalLoss()
    #cosine_loss = CosineLoss(exp=2)
    losses = [dice_loss, focal_loss] 
    #losses = cosine_loss
    train_test(model=model, train_loader = train_loder, test_loader = valid_loader, optimizer=optimizer, 
                                    losses = losses, device = device, num_epochs = num_epochs, 
                                    maskfolder=maskfolder)

    model, optimizer, epoch, score = Load_model(model=model, optimizer=optimizer, maskfolder=maskfolder,) 
    print(epoch, score) 


if __name__ == '__main__':

    os.system('clear') 
    lr = 1e-3
    
    in_ch = 1
    num_classes = 1
    batch_size = 32
    activation = 'sigmoid'
    subfolder= '../dataset/train/image'
    maskfolder = '..dataset/train/label'
    num_epochs = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    #print(device, torch.cuda.is_available(), torch.cuda.current_device(), sep='\n')  
    
    main(lr=lr, num_classes = num_classes, batch_size=batch_size, activation=activation, in_ch= in_ch,
                     device=device, subfolder= subfolder, maskfolder=maskfolder, num_epochs=num_epochs)
    