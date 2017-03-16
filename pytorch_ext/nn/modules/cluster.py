import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math

class Center(nn.Module):
    r"""
    Used for center loss, maybe clustering in future
    """

    def __init__(self, feature_dim, N_center, alpha=0.9, centers=None):
        super(Center, self).__init__()
        if centers is not None:
            self.register_buffer('centers', torch.from_numpy(centers))
        else:
            self.register_buffer('centers', torch.Tensor(N_center, feature_dim))
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
        return self.centers  # ToDo: maybe the returned type should be converted to Vairable instead of current Tensor

    def __repr__(self):
        return self.__class__.__name__ + ' (%d centers)' % self.N_center