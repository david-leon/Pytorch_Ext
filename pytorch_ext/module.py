import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BatchNorm1d(nn.BatchNorm1d):
    """
    Modified from torch.nn.BatchNorm1d
    torch._BatchNorm base class lacks the exception handling when input's batch size=1, the modified logic
    is when input.shape[0]=1, all the parameters will be used as in evaluation mode whether the current self.training mode
    """
    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            input.shape[0] > 1 and (self.training or not self.track_running_stats),  # this is where input shape exception should be handled
            exponential_average_factor, self.eps)



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
        return self.centers  # todo: maybe the returned type should be converted to Vairable instead of current Tensor

    def __repr__(self):
        return self.__class__.__name__ + ' (%d centers)' % self.N_center