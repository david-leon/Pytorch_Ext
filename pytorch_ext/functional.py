"""Functional interface"""
import numpy as np
import torch

def dropout(input, p=0.5, shared_axes=(), rescale=True, fill_value=0.0):
    """
    Dropout function for any dimension input.
    :param input: tensor to be dropout
    :param p: float scalar or list of floats, probability to drop a value (replaced with `fill_value`). If `p` is a list, the actual
              probability will be calculated in an interval of p[i] and p[i+1] randomly. By passing a list to `p`, dropout will be executed
              with varying probability.
    :param shared_axes: tuple of int, axes to share the dropout mask over. By default, each value is dropped individually.
                       shared_axes=(0,) uses the same mask across the batch. shared_axes=(2, 3) uses the same mask across
                       the spatial dimensions of 2D feature maps, i.e., drop channels.
    :param rescale: whether rescale the input by retaining probability
    :param fill_value: in our implementation, the "dropped" values will be replaced with `fill_value`, default = 0
    """
    if isinstance(p, list) or isinstance(p, tuple):
        for pi in p:
            assert 1.0 >= pi >= 0.0, 'p must be within [0, 1.0]'
        n_p = len(p)
        i = np.random.randint(0, n_p-1)
        p = p[i] + np.random.rand() * (p[i+1] - p[i])
    else:
        if p < 0 or p > 1.0:
            raise ValueError('p must be within [0, 1.0]')
    # print('p=', p)
    retain_p = 1.0 - p
    if rescale:
        input = input / retain_p
    mask_shape = input.shape

    # apply dropout, respecting shared axes
    if len(shared_axes) > 0:
        shared_axes = tuple(a if a >= 0 else a + input.ndim for a in shared_axes)  # support for negative indexing
        mask_shape = tuple(1 if a in shared_axes else s for a, s in enumerate(mask_shape))
    p_matrix = torch.zeros(size=mask_shape, device=input.device) + retain_p
    mask = torch.bernoulli(p_matrix)

    r = input * mask
    if fill_value != 0.0:
        r = r + (1.0 - mask) * fill_value
    return r

