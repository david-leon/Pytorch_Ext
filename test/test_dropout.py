# coding:utf-8
"""
Unit test for dropout()
Created   :   11, 24, 2020
Revised   :   11, 24, 2020
All rights reserved
"""
__author__ = 'dawei.leng'

import torch
from pytorch_ext.functional import dropout
import numpy as np

def test_case_0():
    xt = torch.from_numpy(np.random.rand(5, 10).astype('float32')) + 1.0
    # xt = xt.to(torch.device("cuda:0"))
    yt = dropout(xt, p=0.3, shared_axes=(1,), rescale=True)
    assert xt.shape == yt.shape
    mask = yt[:, 0] == 0.0
    for j in range(1, 10):
        mask_j = yt[:, j] == 0.0
        if any(mask != mask_j):
            raise ValueError('mask not consistent')
    print('xt=', xt)
    print('yt=', yt)

if __name__ == '__main__':
    test_case_0()
    print('test_dropout passed~')
