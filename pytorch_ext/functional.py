"""Functional interface"""
import numpy as np
import torch

def dropout(input, p=0.5, shared_axes=(), rescale=True):
    """
    Dropout function for any dimension input.
    :param input: tensor to be dropout
    :param p: float, probability to drop a value (set to 0)
    :param shared_axes: tuple of int, axes to share the dropout mask over. By default, each value is dropped individually.
                       shared_axes=(0,) uses the same mask across the batch. shared_axes=(2, 3) uses the same mask across
                       the spatial dimensions of 2D feature maps, i.e., drop channels.
    :param rescale: whether rescale the input by retaining probability
    """
    if p < 0 or p > 1.0:
        raise ValueError('p must be within [0, 1.0]')
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
    return input * mask

