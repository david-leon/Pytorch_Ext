# coding:utf-8
'''
Unitest for CTC_Log
Created   :  11, 13, 2018
Revised   :  11, 13, 2018
All rights reserved
'''
# ------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'

import os

os.environ['THEANO_FLAGS'] = "floatX=float32, mode=FAST_RUN, warn_float64='raise'"
import numpy as np, time
import theano
from theano import tensor
from dandelion.objective import CTC_Logscale
import torch
from pytorch_ext.objective import CTC_Log

def test_case_0(B=10, C=50, L=1, T=500, device='cpu'):
    #--- theano version ---#
    x1, x2, x3, x4, x5 = tensor.fmatrix(name='queryseq'), \
                         tensor.tensor3(dtype='float32', name='scorematrix'), \
                         tensor.fmatrix(name='queryseq_mask'),\
                         tensor.fmatrix(name='scorematrix_mask'), \
                         tensor.fscalar(name='blank_symbol')

    scorematrix = np.random.rand(T, C + 1, B).astype(np.float32)
    query       = np.random.randint(0, C, (L, B)).astype(np.float32)
    query_mask  = np.random.rand(L, B) > 0.1
    sm_mask     = np.random.rand(T, B) > 0.1

    result = CTC_Logscale.cost(x1, x2, x3, x4, x5, align='pre')
    f2 = theano.function([x1, x2, x3, x4, x5], result, on_unused_input='warn')

    time2 = time.time()
    result_theano = f2(query, scorematrix, query_mask.astype(np.float32), sm_mask.astype(np.float32), C+0.0)
    print('theano: %0.10f' % result_theano)
    time3 = time.time()

    #--- pytorch version ---#
    time0 = time.time()
    ctc       = CTC_Log(align='pre')
    sm_v      = torch.tensor(scorematrix, dtype=torch.float32).transpose(1,2).transpose(0,1).to(device)
    q_v       = torch.tensor(query.astype(np.int64), dtype=torch.long).transpose(0,1).to(device)
    q_mask_v  = torch.tensor(query_mask.astype(np.uint8), dtype=torch.uint8).transpose(0,1).to(device)
    sm_mask_v = torch.tensor(sm_mask.astype(np.uint8), dtype=torch.uint8).transpose(0,1).to(device)


    result_torch = ctc.cost(q_v, sm_v, q_mask_v, sm_mask_v, C)
    result_torch = result_torch.cpu().numpy()
    print('torch : %0.10f' % result_torch)
    time1 = time.time()

    print('Time = torch: %0.6fs | theano: %0.6fs' % (time1-time0, time3-time2))

    dif = abs(result_theano - result_torch)
    if dif > 0.001:
        raise ValueError('Result from torch not consistent with theano')

if __name__ == '__main__':

    test_case_0()

    print('Test passed')