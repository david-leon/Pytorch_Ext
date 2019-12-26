# coding:utf-8
# Unit test for util.free_module()
# Created   :  12, 26, 2019
# Revised   :  12, 26, 2019
# All rights reserved
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from pytorch_ext.module import BatchNorm1d
import numpy as np, warnings

from pytorch_ext.util import get_trainable_parameters, freeze_module, unfreeze_module

class GraphConv(nn.Module):
    """
    A generic graph *convolution* module.

    Note:
    * when `degree_wise` mode is enabled, inputs of `forward()` are required to be specially formatted.
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 aggregation_methods=('sum', 'max'),       # {'sum', 'mean', 'max', 'min', 'att'}
                 multiple_aggregation_merge_method='cat',  # {'cat', 'sum'}
                 affine_before_merge=False,
                 update_method='cat',                      # {'cat', 'sum', 'rnn}
                 backbone='default',
                 degree_wise=True,
                 max_degree=1,
                 **kwargs
                 ):
        """
        :param input_dim:
        :param output_dim:
        :param aggregation_methods: tuple of strings in  {'sum', 'mean', 'max', 'min', 'att'}
        :param multiple_aggregation_merge_method: {'cat', 'sum'}, how their results should be merged
                                                  if there are multiple aggregation methods simultaneously
        :param affine_before_merge: if True, output of each neighborhood aggregation method will be further
                                    affine-transformed before they are merged
        :param update_method: {'cat', 'sum', 'rnn'}, how the center node feature should be merged with aggregated neighbor feature
        :param backbone: nn.Module for feature transformation, a two-layer dense module will be used by default, you can
                         set it to `None` to disable this transformation.
        :param degree_wise: set it to True to enable degree-wise neighborhood aggregation
        :param max_degree: maximum degree allowed. If you have a few nodes with degree > `max_degree` and you don't want to treat
                           them separately degree-wise, you just need put them into a certain degree group and feed the corresponding
                           `x, edges, degree_slices` accordingly. You can even further "quantitize" the degree groups by treating
                           certain degree values as a single value, for example, say you have `max_degree` = 10, you can make a "coarser"
                           degree-wise operation by treating nodes with 0 degree as a group, nodes with 1, 2, 3, 4, 5 degrees as a group,
                           and nodes with 6, 7, 8, 9, 10 & > 10 degrees as a group.
        :param kwargs:  1) head_num: attention head number, for `att` aggregation method, default = 1
                        2) att_mode: {'combo', 'single'}, specify attention mode for `att` aggregation method. The `att`
                           method is basically correlating node features with the attention vector, this correlation can
                           be done at single node level or at neighbor-center combination level. For the latter mode, attention
                           is done on concatenation of each tuple of (neighbor, center) node features.
                        3) eps: initial value for the self-connection weight, learnable. Only effective when `update_method` = 'sum'.
                           Default = 1.0
        """
        super().__init__()
        self.input_dim           = input_dim
        self.output_dim          = output_dim
        self.affine_before_merge = affine_before_merge
        self.aggregation_methods = []
        for item in aggregation_methods:
            item = item.lower()
            if item not in {'sum', 'mean', 'max', 'min', 'att'}:
                raise ValueError("aggregation_method should be in {'sum', 'mean', 'max', 'min', 'att'}")
            self.aggregation_methods.append(item)
            if item == 'att':
                if 'head_num' in kwargs:
                    self.head_num = kwargs['head_num']
                else:
                    self.head_num = 1
                assert self.input_dim % self.head_num == 0, 'input_dim must be multiple of head_num'
                if 'att_mode' in kwargs:
                    self.att_mode = kwargs['att_mode']
                else:
                    self.att_mode = 'combo'
                assert self.att_mode in {'single', 'combo'}
                if self.att_mode == 'single':
                    self.att_weight = Parameter(torch.empty(size=(1, self.head_num, self.input_dim//self.head_num)))
                else:
                    self.att_weight = Parameter(torch.empty(size=(1, self.head_num, 2 * self.input_dim // self.head_num)))
        self.multiple_aggregation_merge_method = multiple_aggregation_merge_method.lower()
        assert self.multiple_aggregation_merge_method in {'cat', 'sum'}
        aggregation_num = len(self.aggregation_methods)
        if self.affine_before_merge:
            self.affine_tranforms = nn.ModuleList()
            for i in range(aggregation_num):
                self.affine_tranforms.append(nn.Linear(in_features=input_dim, out_features=input_dim))
        self.alpha = [1.0]        # default dummy value
        if aggregation_num > 1:
            if self.multiple_aggregation_merge_method == 'sum':
                self.alpha = Parameter(torch.tensor(np.ones(aggregation_num), dtype=torch.float32))  # to be learned
                torch.nn.utils.clip_grad_value_(self.alpha, 0.1)
            else:
                self.merge_layer = nn.Linear(in_features=input_dim * aggregation_num, out_features=input_dim)
        self.update_method = update_method.lower()
        assert self.update_method in {'cat', 'sum', 'rnn'}
        if self.update_method == 'sum':
            if 'eps' in kwargs:
                eps = kwargs['eps']
            else:
                eps = 1.0
            self.eps = Parameter(torch.scalar_tensor(eps, dtype=torch.float32))  # to be learned
            torch.nn.utils.clip_grad_value_(self.eps, 0.1)
        elif self.update_method == 'rnn':
            self.rnn = nn.GRUCell(input_size=input_dim, hidden_size=input_dim)

        if backbone.lower() == 'default':
            if self.update_method == 'cat':
                backbone_input_dim = 2 * input_dim
            else:
                backbone_input_dim = input_dim
            hidden_dim = min(2 * input_dim, 256)
            self.backbone = nn.Sequential(nn.Linear(in_features=backbone_input_dim, out_features=hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(in_features=hidden_dim, out_features=output_dim),
                                          nn.ReLU(),
                                          BatchNorm1d(num_features=output_dim))
        else:
            self.backbone = backbone

        self.degree_wise = degree_wise
        if self.degree_wise:
            self.max_degree      = max_degree
            self.linear_neighbor = nn.Linear(in_features=input_dim, out_features=input_dim, bias=False)
            self.linear_center   = nn.Linear(in_features=input_dim, out_features=input_dim, bias=False)
            self.linears_degree  = nn.ModuleList()
            for i in range(self.max_degree):
                self.linears_degree.append(nn.Linear(in_features=input_dim, out_features=input_dim, bias=False))

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'att_weight'):
            nn.init.xavier_normal_(self.att_weight)

    def forward(self, x, edges, edge_weights=None, include_self_in_neighbor=False, degree_slices=None):
        """
        Forward with degree-wise aggregation support
        :param x: (node_num, D), requiring that nodes of the same degree are grouped together when self.degree_wise = True
        :param edges: (2, edge_num), requiring that edges with center node of the same degree are grouped together when
                      self.degree_wise = True. Note due to this requirement `include_self_in_neighbor` argument is not
                      supported when self.degree_wise = True, you need to add self connections in `edges` before feeding
                      the data into the module.
        :param edge_weights: (edge_num,), edge weights, order the same with `edges`
        :param include_self_in_neighbor: when performing neighborhood operations, whether include self (center) nodes, note
                                         this argument must be set to False when self.degree_wise = True, in that case, you
                                         need to add self connections in `edges` before feeding the data into the module
        :param degree_slices: (max_degree_in_batch+1, 2), each row in format of (start_idx, end_idx), in which '*_idx' corresponds
                              to edges indices; i.e., each row is the span of edges whose center node is of the same degree,
                              required when self.degree_wise = True, otherwise leave it to None
        :return:
        """
        node_num, feature_dim = x.shape
        x_org = x
        if include_self_in_neighbor:
            if self.degree_wise:
                raise ValueError('`include_self_in_neighbor` param must be set to False when `degree_wise`=True')
            else:
                edges, edge_weights = add_remaining_self_loops(edges, edge_weights, num_nodes=node_num)
        x_neighbor  = x_org[edges[0, :], :]
        if self.degree_wise:
            x_neighbor  = self.linear_neighbor(x_neighbor)
            x           = self.linear_center(x)

        if degree_slices is None:
            edge_num = edges.shape[1]
            degree_slices = np.array([[0, edge_num]], dtype=np.int64)
            degree_slices = torch.from_numpy(degree_slices).to(x.device)

        x_aggregated_degree_list = []
        for degree, span in enumerate(degree_slices):
            if self.degree_wise:
                node_num_degree = (span[1] - span[0]) // max(degree, 1)
                # node_num_degree = np.int64(node_num_degree)
            else:
                node_num_degree = node_num
            if node_num_degree <= 0:
                continue
            if self.degree_wise and degree == 0:  # no neighbors
                x_aggregated_degree = torch.zeros(node_num_degree, feature_dim, dtype=x.dtype, device=x.device)
                x_aggregated_degree_list.append(x_aggregated_degree)
                continue
            edges_degree        = edges[:, span[0]:span[1]]   # no copy
            edge_weights_degree = edge_weights[span[0]:span[1]] if edge_weights is not None else None
            x_neighbor_degree   = x_org[edges_degree[0, :], :]
            if edge_weights_degree is not None:
                x_neighbor_degree = x_neighbor_degree * edge_weights_degree
            if self.degree_wise:
                x_neighbor_degree = self.linears_degree[degree](x_neighbor_degree)
                x_neighbor_degree += x_neighbor[span[0]:span[1], :]

            #---- neighborhood aggregation ----#
            aggr_outputs = []
            if self.degree_wise:
                scatter_index = edges_degree[1, :] - edges_degree[1, :].min()
            else:
                scatter_index = edges_degree[1, :]
            for i, aggr_method in enumerate(self.aggregation_methods):
                if aggr_method == 'max':
                    x_aggregated_degree, _ = torch_scatter.scatter_max(x_neighbor_degree, scatter_index, dim=0, dim_size=node_num_degree, fill_value=0)
                elif aggr_method == 'min':
                    x_aggregated_degree, _ = torch_scatter.scatter_min(x_neighbor_degree, scatter_index, dim=0, dim_size=node_num_degree, fill_value=0)
                elif aggr_method == 'mean':
                    x_aggregated_degree = torch_scatter.scatter_mean(x_neighbor_degree, scatter_index, dim=0, dim_size=node_num_degree, fill_value=0)
                elif aggr_method == 'sum':   # aggr_method == 'sum'
                    x_aggregated_degree = torch_scatter.scatter_add(x_neighbor_degree, scatter_index, dim=0, dim_size=node_num_degree, fill_value=0)
                elif aggr_method == 'att':
                    edge_num_degree = span[1] - span[0]
                    x_neighbor = x_neighbor_degree.view(edge_num_degree, self.head_num, -1)  # (N, D) -> (N, heads, out_channels)
                    query = x_neighbor
                    if self.att_mode == 'combo':
                        x_center = x[edges_degree[1, :], :].view(edge_num_degree, self.head_num, -1)
                        query = torch.cat([query, x_center], dim=-1)  # (N, heads, 2*out_channels)
                    alpha = query * self.att_weight
                    alpha = alpha.sum(dim=-1)  # (N, heads)
                    alpha = F.leaky_relu(alpha, 0.2)  # (N, heads), use leaky relu as in GAT paper
                    alpha = torch_scatter.composite.scatter_softmax(alpha, edges[1, :].view(edge_num_degree, 1), dim=0)
                    x_neighbor = x_neighbor * alpha.view(-1, self.head_num, 1)  # (N, heads, out_channels)
                    x_neighbor = x_neighbor.view(edge_num_degree, -1)
                    x_aggregated_degree = torch_scatter.scatter_add(x_neighbor, edges_degree[1, :], dim=0, dim_size=node_num_degree)
                else:
                    raise ValueError('aggregation method = %s not supported' % aggr_method)
                if self.affine_before_merge:
                    x_aggregated_degree = self.affine_tranforms[i](x_aggregated_degree)
                aggr_outputs.append(x_aggregated_degree)
            if self.multiple_aggregation_merge_method == 'sum':
                x_aggregated_degree = 0
                for i, aggr_out in enumerate(aggr_outputs):
                    x_aggregated_degree += self.alpha[i] * aggr_out
            else:  # concatenation
                if len(self.aggregation_methods) > 1:
                    x_aggregated_degree = torch.cat(aggr_outputs, dim=1)
                    x_aggregated_degree = self.merge_layer(x_aggregated_degree)  # for dimension normalization
                else:
                    x_aggregated_degree = aggr_outputs[0]
            x_aggregated_degree_list.append(x_aggregated_degree)

        x_aggregated = torch.cat(x_aggregated_degree_list, dim=0)

        #---- center update ---#
        if self.update_method == 'sum':
            x = self.eps * x + x_aggregated
        elif self.update_method == 'cat':
            x = torch.cat([x, x_aggregated], dim=1)
        elif self.update_method == 'rnn':
            x = self.rnn(x, x_aggregated)
        else:
            raise ValueError('update method = %s not supported' % self.update_method)

        if self.backbone is not None:
            x = self.backbone(x)
        return x

class model_0(nn.Module):
    """
    Baseline auto-encoder w.r.t node features
    """
    def __init__(self,
                 num_embedding=0,
                 embedding_dim=75,
                 hidden_dims=(256, 64, 32),
                 aggregation_methods=('max', 'sum'),
                 multiple_aggregation_merge_method='cat',
                 affine_before_merge=False,
                 node_feature_update_method='cat',
                 tie_embedding_weights=True,
                 **kwargs
                 ):
        super().__init__()
        self.num_embedding              = num_embedding
        self.embedding_dim              = embedding_dim
        self.embedding                  = nn.Embedding(num_embeddings=num_embedding, embedding_dim=embedding_dim)
        hidden_dims                     = list(hidden_dims)
        self.hidden_dims                = hidden_dims + hidden_dims[-2::-1] + [embedding_dim]
        self.aggregation_methods        = aggregation_methods
        self.multiple_aggregation_merge = multiple_aggregation_merge_method
        self.blocks                     = nn.ModuleList()
        block_input_dim = embedding_dim
        for i, block_output_dim in enumerate(self.hidden_dims):
            self.blocks.append(GraphConv(input_dim=block_input_dim, output_dim=block_output_dim,
                                         aggregation_methods=aggregation_methods,
                                         multiple_aggregation_merge_method=multiple_aggregation_merge_method,
                                         affine_before_merge=affine_before_merge,
                                         update_method=node_feature_update_method,
                                         degree_wise=False,
                                         backbone='default',
                                         **kwargs,
                                         ))
            block_input_dim = block_output_dim
        self.dense = nn.Linear(embedding_dim, num_embedding, bias=False)
        if tie_embedding_weights:
            self.dense.weight = self.embedding.weight

    def forward(self, x, edges=None, dropout=0.0):
        """
        :param x: (node_num,) int64 if embedding is enabled;
        :param edges: (2, edge_num), int64, each column in format of (neighbor_node, center_node)
        :param dropout: dropout value
        :return: x, (node_num, num_embedding)
        """
        x = self.embedding(x)
        #--- aggregation ---#
        block_input = x
        for i in range(len(self.blocks)):
            block_input = F.dropout(block_input, p=dropout, training=self.training)
            x = self.blocks[i](x=block_input, edges=edges,
                               include_self_in_neighbor=False)
            block_input = x

        #--- decode ---#
        x = self.dense(x)
        return x

def test_case_0():
    model = model_0(num_embedding=10)
    param_list = list(model.named_parameters())
    freeze_module(model.blocks[0])
    freeze_module(model.embedding)
    param_list2 = list(get_trainable_parameters(model, with_name=True))
    unfreeze_module(model.blocks[0])
    unfreeze_module(model.embedding)
    param_list3 = list(get_trainable_parameters(model, with_name=True))

    name_list = [name for name, tensor in param_list]
    name_list2 = [name for name, tensor in param_list2]
    name_list3 = [name for name, tensor in param_list3]
    print(name_list)
    print(name_list2)
    print(name_list3)
    assert name_list == name_list3
    assert name_list != name_list2
    for name in name_list:
        if name not in name_list2:
            print(name)


if __name__ == '__main__':
    test_case_0()
