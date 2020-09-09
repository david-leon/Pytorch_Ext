import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter

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
    """
    def __init__(self,
                 input_shape         = None,
                 axes                = 'auto',
                 eps                 = 1e-5,
                 alpha               = 0.1,
                 beta                = 0.0,
                 gamma               = 1.0,
                 mean                = 0.0,
                 inv_std             = 1.0,
                 update_buffer_size  = 1,
                 update_batch_limit  = None
                 ):
        """
         :param input_shape: tuple or list of int or tensor variable. Including batch dimension. Any shape along axis defined in `axes` can be set to None
         :param axes: 'auto' or tuple of int. The axis or axes to normalize over. If ’auto’ (the default), normalize over
                       all axes except for the second: this will normalize over the minibatch dimension for dense layers,
                       and additionally over all spatial dimensions for convolutional layers.
         :param eps: Small constant 𝜖 added to the variance before taking the square root and dividing by it, to avoid numerical problems
         :param alpha: mean = (1 - alpha) * mean + alpha * batch_mean
         :param beta:  set to None to disable this parameter
         :param gamma: set to None to disable this parameter
         :param mean:  set to None to disable this buffer
         :param inv_std: set to None to disable this buffer
         :param update_buffer_size: int, default = 1. If > 1, the running mean & std statistics will be calculated across multiple recent
                                    batches, this will help mitigate inconsistent statistics problem caused by small batch size. Note here
                                    batch size is calculated by dimention product specified by `axes`. This number can be varying for different
                                    input, you can retrieve it by attribute `self.n`
         :param update_batch_limit: int, valid only when `update_buffer_size` > 1. This is an additional condition means only when the effective
                                    batch size < `update_batch_limit`, the running mean & std statistics will be calculated across multiple recent
                                    batches. Default = None, disabled.
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
        self.update_buffer_size = update_buffer_size
        self.update_batch_limit = update_batch_limit


        shape = [size for axis, size in enumerate(input_shape) if axis not in self.axes]  # remove all dimensions in axes
        if any(size is None for size in shape):
            raise ValueError("BatchNorm needs specified input sizes for all axes not normalized over.")

        self.broadcast_shape = [1] * len(self.input_shape)
        for i in range(len(self.input_shape)):
            if i not in self.axes:
                self.broadcast_shape[i] = self.input_shape[i]   # broadcast_shape = [1, C, 1, 1]

        # print('axes=', self.axes)
        # print('shape=', shape)
        # print('broadcast_shape=', self.broadcast_shape)

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

        self.stat_buffer = []
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
        # print('input.shape=', x.shape)
        if self.mean is None and self.inv_std is None or self.training:
            input_mean = x.mean(self.axes)
            input_inv_std = 1.0 / (torch.sqrt(x.var(self.axes) + self.eps))
            n = 1
            for dim in self.axes:
                n *= x.shape[dim]
            self.n = n  # this is the actual *batch size*

            if self.update_buffer_size > 1:
                stat = [input_mean, input_inv_std, n]
                self.stat_buffer.append(stat)
                if len(self.stat_buffer) > self.update_buffer_size:
                    self.stat_buffer.pop(0)
                if self.update_batch_limit is None or self.update_batch_limit is not None and self.n < self.update_batch_limit:
                    input_mean *= n
                    input_inv_std *= n
                    n_total = n
                    for mean_batch, inv_std_batch, n_batch in self.stat_buffer[:-1]:
                        input_mean += mean_batch * n_batch
                        input_inv_std += input_inv_std * n_batch
                        n_total += n_batch
                    # print('n_total=', n_total)
                    input_mean /= n_total
                    input_inv_std /= n_total
            # print('input_mean.shape', input_mean.shape)

        if self.training:
            if self.mean is not None:
                self.mean = (1 - self.alpha) * self.mean + self.alpha * input_mean
            if self.inv_std is not None:
                self.inv_std = (1 - self.alpha) * self.inv_std + self.alpha * input_inv_std

        if self.training:
            mean = input_mean.reshape(self.broadcast_shape)
            inv_std = input_inv_std.reshape(self.broadcast_shape)
        else:
            if self.mean is not None:
                mean = self.mean.reshape(self.broadcast_shape)
            else:
                mean = 0.0
            if self.inv_std is not None:
                inv_std = self.inv_std.reshape(self.broadcast_shape)
            else:
                inv_std = 1.0
        beta  = 0.0 if self.beta  is None else torch.reshape(self.beta, self.broadcast_shape)
        gamma = 1.0 if self.gamma is None else torch.reshape(self.gamma, self.broadcast_shape)

        normalized = (x - mean) * (gamma * inv_std) + beta
        return normalized


class Center(nn.Module):
    r"""
    Used for center loss, maybe clustering in future
    """

    def __init__(self, feature_dim, N_center, alpha=0.9, centers=None):
        super(Center, self).__init__()
        if centers is not None:
            self.register_buffer('centers', torch.from_numpy(centers))
        else:
            self.register_buffer('centers', torch.empty(N_center, feature_dim))
            self.reset_parameters()
        self.alpha = alpha
        self.N_center = N_center

    def reset_parameters(self):
        with torch.no_grad():
            stdv = 1. / math.sqrt(self.centers.size(1))
            self.centers.uniform_(-stdv, stdv)

    def forward(self, inputs):  # inputs[0] = features (B, D), inputs[1] = target labels (B,)
        """
        Call .forward(...) explicitly
        :param inputs:
        :return:
        """
        if not self.training:
            pass
        else:
            features, labels = inputs
            diff = (self.alpha - 1.0) * (self.centers.index_select(0, labels.data) - features.data)
            self.centers.index_add_(0, labels.data, diff)
        return self.centers

    def __repr__(self):
        return self.__class__.__name__ + ' (%d centers)' % self.N_center
