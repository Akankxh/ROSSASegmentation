import os 
import sys 
import math 

import numpy as np 


import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from torch.nn.modules.loss import _Loss 

from .functional import soft_dice_score 

class DiceLoss(_Loss):
    def __init__( self, log_loss = False,  
          from_logits = True,smooth= 0.0, ignore_index = None,eps= 1e-7,):
        
        super(DiceLoss, self).__init__()
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index
    
    def forward(self, y_pred, y_true):
        assert y_true.size(0) == y_pred.size(0) 

        if self.from_logits:
            y_pred = F.logsigmoid(y_pred).exp()  
        

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2) 

        y_true = y_true.view(bs, 1, -1)
        y_pred = y_pred.view(bs, 1, -1) 

        scores = self.compute_score( y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims) 
        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores
        
        return self.aggregate_loss(loss)
    
    def aggregate_loss(self, loss):
        return loss.mean()
    
    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None):
        return soft_dice_score(output, target, smooth, eps, dims)
