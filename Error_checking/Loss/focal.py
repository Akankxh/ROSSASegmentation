import os 
import sys 
import math 

import numpy as np 

from functools import partial 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from torch.nn.modules.loss import _Loss 

from .functional import focal_loss_with_logits 


class FocalLoss(_Loss):
    def __init__( self, alpha = None, gamma = 2.0, ignore_index = None,
        reduction = "mean", normalized = False, reduced_threshold = None,):

        super().__init__()
        self.focal_loss_fn = partial( focal_loss_with_logits, alpha=alpha, gamma=gamma,
            reduced_threshold=reduced_threshold, reduction=reduction, normalized=normalized,) 
    
    def forward(self, y_pred, y_true):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)

        loss = self.focal_loss_fn(y_pred, y_true)
        return loss