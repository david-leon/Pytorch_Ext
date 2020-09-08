# coding:utf-8
"""
Unit test for BatchNorm()
Created   :   9,  8, 2020
Revised   :   9,  8, 2020
All rights reserved
"""
__author__ = 'dawei.leng'

import torch
from pytorch_ext.module import BatchNorm
import numpy as np

def test_case_0():
    B, C, H, W = 4, 5, 256, 256
    bn = BatchNorm(input_shape=(None, C, H, W), beta=None, gamma=None)
    x = torch.from_numpy(np.random.rand(B, C, H, W))
    y = bn.forward(x)
    assert(y.shape == (B, C, H, W))

    B, C, D = 4, 256, 128
    bn = BatchNorm(input_shape=(None, C, D), axes=(0,2), update_buffer_size=2)
    x = torch.from_numpy(np.random.rand(B, C, D))
    # bn.use_input_stat = False
    y = bn.forward(x)
    print('actual batch size = ', bn.n)
    y = bn.forward(torch.from_numpy(np.random.rand(5, C, D)))
    print('actual batch size = ', bn.n)
    y = bn.forward(x+3.0)
    assert(y.shape == (B, C, D))

if __name__ == '__main__':
    test_case_0()
    print('test_BatchNorm passed~')
