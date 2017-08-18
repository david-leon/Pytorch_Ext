# coding:utf-8
# Utility functions for Pytorch_Ext
# Created   :   8, 11, 2017
# Revised   :   8, 18, 2017
# All rights reserved
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'
import torch

def get_device(x):
    """
    Convenient & unified function for getting device a Tensor or Variable resides on
    :param x: torch Tensor or Variable
    :return: device ID, -1 for CPU; 0, 1, ... for GPU
    """
    if x.is_cuda is False:
        return -1
    else:
        return x.get_device()
		
def set_device(x, device):
    """
    Convenient & unified function for moving a Tensor or Variable to a specified device
    :param x: torch Tensor or Variable
    :param device: device ID, -1 for CPU; 0, 1, ... for GPU
    :return: Tensor or Variable on the specified device
    """
    if device < 0:
        if x.is_cuda:
            return x.cpu()
        else:
            return x
    else:
        return x.cuda(device)

def grad_clip(parameters, min, max):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    for p in parameters:
        p.grad.data = torch.clamp(p.grad.data, min=min, max=max)