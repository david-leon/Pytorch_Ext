import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter

class Center(nn.Module):
    r"""
    Used for center loss, maybe clustering in future
    """

    def __init__(self, feature_dim, center_num, alpha=0.9, centers=None):
        super(Center, self).__init__()
        if centers is not None:
            self.register_buffer('centers', torch.from_numpy(centers))
        else:
            self.register_buffer('centers', torch.empty(center_num, feature_dim))
            self.reset_parameters()
        self.alpha = alpha
        self.center_num = center_num

    def reset_parameters(self):
        with torch.no_grad():
            stdv = 1. / math.sqrt(self.centers.size(1))
            self.centers.uniform_(-stdv, stdv)

    def forward(self, features=None, labels=None):
        """
        Call .forward(...) explicitly
        :param inputs:
        :return:
        """
        if not self.training:
            pass
        else:
            diff = (self.alpha - 1.0) * (self.centers.index_select(0, labels.data) - features.data)
            self.centers.index_add_(0, labels.data, diff)
        return self.centers

    def __repr__(self):
        return self.__class__.__name__ + ' (%d centers)' % self.center_num

class BatchNorm1d(nn.BatchNorm1d):
    """
    Modified from torch.nn.BatchNorm1d
    torch._BatchNorm base class lacks the exception handling when input's batch size=1, the modified logic
    is when input.shape[0]=1, all the parameters will be used as in evaluation mode whether the current self.training mode
    """

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        """ Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)
        bn_training = input.shape[0] > 1 and bn_training  # [DV] this is where input shape exception should be handled
        """Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight, self.bias, bn_training, exponential_average_factor, self.eps)

class BatchNorm(nn.Module):
    """
    Batch normalization for any dimention input, adapted from Dandelion's BatchNorm class

    According to experiment results, keep input statistics in backward propagation graph is crucial for the success of BN
    """
    def __init__(self,
                 input_shape         = None,
                 axes                = 'auto',
                 eps                 = 1e-5,
                 alpha               = 0.01,
                 beta                = 0.0,
                 gamma               = 1.0,
                 mean                = 0.0,
                 inv_std             = 1.0,
                 ):
        """
         :param input_shape: tuple or list of int or tensor. Including batch dimension. Any shape along axis defined in `axes` can be set to None
         :param axes: 'auto' or tuple of int. The axis or axes to normalize over. If ’auto’ (the default), normalize over
                       all axes except for the second: this will normalize over the minibatch dimension for dense layers,
                       and additionally over all spatial dimensions for convolutional layers.
         :param eps: Small constant 𝜖 added to the variance before taking the square root and dividing by it, to avoid numerical problems
         :param alpha: mean = (1 - alpha) * mean + alpha * batch_mean
         :param beta:  set to None to disable this parameter
         :param gamma: set to None to disable this parameter
         :param mean:  set to None to disable this buffer
         :param inv_std: set to None to disable this buffer. If both `mean` and `inv_std` are disabled, then input mean and variance will be used instead
         """

        super().__init__()
        if input_shape is None:
            raise ValueError('`input_shape` must be specified for BatchNorm class')
        self.input_shape = input_shape
        if axes == 'auto':      # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(input_shape)))
        if isinstance(axes, int):
            axes = (axes,)
        self.axes  = axes
        self.eps   = eps
        self.alpha = alpha

        shape = [size for axis, size in enumerate(input_shape) if axis not in self.axes]  # remove all dimensions in axes
        if any(size is None for size in shape):
            raise ValueError("BatchNorm needs specified input sizes for all axes not normalized over.")

        self.broadcast_shape = [1] * len(self.input_shape)
        for i in range(len(self.input_shape)):
            if i not in self.axes:
                self.broadcast_shape[i] = self.input_shape[i]   # broadcast_shape = [1, C, 1, 1]

        # --- beta & gamma are trained by BP ---#
        if beta is None:
            self.beta = None
        else:
            self.beta = Parameter(torch.zeros(size=shape) + beta)

        if gamma is None:
            self.gamma = None
        else:
            self.gamma = Parameter(torch.zeros(size=shape) + gamma)

        # --- mean & inv_std are trained by self-updating ---#
        if mean is not None:
            self.register_buffer(name='mean', tensor=torch.zeros(size=shape) + mean)
        else:
            self.mean = None
        if inv_std is not None:
            self.register_buffer(name='inv_std', tensor=torch.zeros(size=shape) + inv_std)
        else:
            self.inv_std = None
        self.n = 0

    def reset_parameters(self):
        with torch.no_grad():
            if self.mean is not None:
                self.mean.fill_(0.0)
            if self.inv_std is not None:
                self.inv_std.fill_(1.0)
            if self.beta is not None:
                self.beta.fill_(0.0)
            if self.gamma is not None:
                self.gamma.fill_(1.0)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        if self.mean is None and self.inv_std is None or self.training:
            n = 1
            for dim in self.axes:
                n *= x.shape[dim]
            self.n = n                        # this is the actual *batch size*
            input_mean = x.mean(self.axes)
            input_inv_std = 1.0 / (torch.sqrt(x.var(self.axes) + self.eps))
            if self.n <= 1:
                input_mean.fill_(0.0)
                input_inv_std.fill_(1.0)

        if self.training and self.n > 1:
            if self.mean is not None:
                self.mean = (1 - self.alpha) * self.mean + self.alpha * input_mean.detach()
            if self.inv_std is not None:
                self.inv_std = (1 - self.alpha) * self.inv_std + self.alpha * input_inv_std.detach()

        if self.mean is not None:
            mean = self.mean
        else:
            mean = 0.0
        if self.inv_std is not None:
            inv_std = self.inv_std
        else:
            inv_std = 1.0
        if self.mean is None and self.inv_std is None or self.training and self.n > 1:
            mean    = input_mean
            inv_std = input_inv_std

        mean    = mean.reshape(self.broadcast_shape)
        inv_std = inv_std.reshape(self.broadcast_shape)
        beta    = 0.0 if self.beta  is None else torch.reshape(self.beta, self.broadcast_shape)
        gamma   = 1.0 if self.gamma is None else torch.reshape(self.gamma, self.broadcast_shape)

        normalized = (x - mean) * (gamma * inv_std) + beta
        return normalized

