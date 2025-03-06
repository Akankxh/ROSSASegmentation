import os 
import sys 
import math
from typing import Any 

import numpy as np 

import torch 
import torch.nn as nn 
import torch.nn.functional as F  

class BinaryMetric:
    def __init__(self, eps=1e-5, activation="0-1"):
        self.eps = eps 
        self.activation = activation 
    
    def _calculate_overlap_metrics(self, gt, pred): 
        output = pred.view(-1, )
        target = gt.view(-1, ).float() 

        tp = torch.sum(output * target)  # TP
        fp = torch.sum(output * (1 - target))  # FP 
        fn = torch.sum((1 - output) * target)  # FN 
        tn = torch.sum((1 - output) * (1 - target))  # TN 

        pixel_acc = (tp + tn + self.eps) / (tp + tn + fp + fn + self.eps)
        dice = (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps) 
        iou  = (tp + self.eps) / (tp + fp + fn + self.eps) 
        precision = (tp + self.eps) / (tp + fp + self.eps) 
        recall = (tp + self.eps) / (tp + fn + self.eps) 
        specificity = (tn + self.eps) / (tn + fp + self.eps) 
        return pixel_acc, dice, iou, precision, specificity, recall 
    
    def __call__(self, y_true, y_pred):
        # y_true: (N, H, W)
        # y_pred: (N, 1, H, W) 

        if len(y_true.shape) !=3: 
            y_true = y_true.squeeze(1)  
        
        if self.activation == "sigmoid":
            activation_fn = nn.Sigmoid() 
            activated_pred = activation_fn(y_pred)
        elif self.activation == "0-1": 
            sigmoid_pred = nn.Sigmoid()(y_pred)
            activated_pred = (sigmoid_pred > 0.5).float().to(y_pred.device)
        
        acc, dice, iou, precision, specificity, recall = self._calculate_overlap_metrics ( activated_pred, y_true) 
        return acc, dice, iou, precision, specificity, recall 
        
        