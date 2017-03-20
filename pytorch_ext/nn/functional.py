"""Functional interface"""

import torch
from torch.autograd import Variable

class CTC_Log(object):
    """
    This implementation uses log scale computation.
    Batch supported.
    B: BATCH_SIZE
    L: query sequence length (maximum length of a batch)
    C: class number
    T: time length (maximum time length of a batch)
    """
    def __init__(self, eps=1E-12, inf=1E12, align='pre'):
        """
        :param eps:
        :param inf:
        :param align: {'pre' or 'post'}
        """
        super(CTC_Log, self).__init__()
        self.eps = eps
        self.inf = inf
        self.align = align

    def cost(self, queryseq, scorematrix, queryseq_mask=None, scorematrix_mask=None, blank_symbol=None):
        """
        Compute CTC cost, using only the forward pass
        :param queryseq:          -> (B, L)        Variable of LongTensor
        :param scorematrix:       -> (B, T, C+1)   Variable of FloatTensor
        :param queryseq_mask:     -> (B, L)        Variable of ByteTensor
        :param scorematrix_mask:  -> (B, T)        Variable of ByteTensor
        :param blank_symbol: scalar, = C by default      Integer
        :return: negative log likelihood averaged over a batch
        """
        if blank_symbol is None:
            blank_symbol = scorematrix.size(2) - 1
        queryseq_padded, queryseq_mask_padded = self._pad_blanks(queryseq, blank_symbol, queryseq_mask)

        NLL, alphas = self.path_probability(queryseq_padded, scorematrix, queryseq_mask_padded, scorematrix_mask, blank_symbol)

        NLL_avg = torch.mean(NLL)
        return NLL_avg

    def path_probability(self, queryseq_padded, scorematrix, queryseq_mask_padded=None, scorematrix_mask=None,
                         blank_symbol=None):
        """
        Compute p(l|x) using only the forward variable and log scale
        :param queryseq_padded: (2L+1, B)   -> (B, 2L+1)     LongTensor
        :param scorematrix: (T, C+1, B)     -> (B, T, C+1)   FloatTensor
        :param queryseq_mask_padded: (2L+1, B) -> (B, 2L+1)  ByteTensor
        :param scorematrix_mask: (T, B)        -> (B, T)     ByteTensor
        :param blank_symbol: = C by default                  Integer
        :return:
        """
        if blank_symbol is None:
            blank_symbol = scorematrix.size(2) - 1
        if queryseq_mask_padded is None:
            queryseq_mask_padded = Variable(torch.ones(queryseq_padded.size()).byte())
        if scorematrix_mask is None:
            scorematrix_mask = Variable(torch.ones(scorematrix.size(0), scorematrix.size(1)).byte())

        pred_y   = self._class_batch_to_labeling_batch(queryseq_padded, scorematrix, scorematrix_mask)  # (B, T, 2L+1), reshaped scorematrix
        r2, r3   = self._recurrence_relation(queryseq_padded, queryseq_mask_padded, blank_symbol)  # r2 (2L+1, 2L+1), r3 (2L+1, 2L+1, B)
        pred_y   = self._epslog(pred_y)
        B, T, L2 = pred_y.size()
        p_prev   = Variable(self._epslog(torch.eye(1, L2).expand(B,L2) * torch.ones(B, L2)))    # (B, 2L+1)
        queryseq_mask_padded = queryseq_mask_padded.float()
        if self.align == 'pre':
            alphas = Variable(torch.Tensor(T, B, L2))    # (T, B, 2L+1)
        for i in range(T):
            p_curr = pred_y[:, i, :]                                 # (B, 2L+1)
            p2     = self._log_dot_matrix(p_prev, r2)      # (B, 2L+1)
            p3     = self._log_dot_tensor(p_prev, r3)      # (B, 2L+1)
            p123   = self._log_add(p3, self._log_add(p_prev, p2))        # (B, 2L+1)
            p_prev = p_curr + p123 + self._epslog(queryseq_mask_padded)
            if self.align == 'pre':
                alphas[i, :, :] = p_prev

        LL  = torch.sum(queryseq_mask_padded, 1).long().squeeze()     # (B,)
        # Idx = torch.range(0, B-1)                           # (B,)

        if self.align == 'pre':
            TL = torch.sum(scorematrix_mask, 1).long().squeeze()  # (B,)
            a1 = torch.stack([alphas[TL.data[i]-1,i,LL.data[i]-1] for i in range(B)])
            a2 = torch.stack([alphas[TL.data[i]-1,i,LL.data[i]-2] for i in range(B)])
        else:
            alphas = p_prev
            a1 = torch.stack([alphas[i,LL.data[i]-1] for i in range(B)])
            a2 = torch.stack([alphas[i,LL.data[i]-2] for i in range(B)])

        NLL = -self._log_add(a1, a2)
        return NLL, alphas            # NLL.shape = (B,)

    def _epslog(self, x):
        return torch.log(torch.clamp(x, self.eps, self.inf))

    def _log_add(self, a, b):
        max_ = torch.max(a, b)
        return max_ + torch.log1p(torch.exp(a + b - 2 * max_))

    def _log_dot_matrix(self, x, z):
        log_dot = torch.mm(x, z)                            #(m,n)
        zeros_to_minus_inf = (z.max(0)[0] - 1) * self.inf        #(n,)
        return log_dot + zeros_to_minus_inf.expand(log_dot.size())

    def _log_dot_tensor(self, x, z):
        log_dot = (x.transpose(0,1).unsqueeze(1).expand_as(z) * z).sum(0).squeeze().t()   # (B, 2L+1)
        zeros_to_minus_inf = (z.max(0)[0] - 1).squeeze().t() * self.inf                        # (B, 2L+1)
        return log_dot + zeros_to_minus_inf


    @staticmethod
    def _pad_blanks(queryseq, blank_symbol, queryseq_mask=None):
        """
        Pad queryseq and corresponding queryseq_mask with blank symbol
        :param queryseq  (L, B)      -> (B, L)    LongTensor
        :param queryseq_mask (L, B)  -> (B, L)    ByteTensor
        :param blank_symbol  integer scalar
        :return queryseq_padded, queryseq_mask_padded, both with shape (2L+1, B) -> (B, 2L+1)   LongTensor, ByteTensor
        """
        # for queryseq
        queryseq_extended = queryseq.unsqueeze(2)                                        # (B, L) -> (B, L, 1)
        blanks            = Variable(torch.zeros(queryseq_extended.size()).long() + blank_symbol)         # (B, L, 1)
        concat            = torch.cat((queryseq_extended, blanks), 2)                    # concat.shape = (B, L, 2)
        res               = concat.view(concat.size(0), concat.size(1) * concat.size(2)) # (B, 2L), the reshape will cause the last 2 dimensions interlace
        begining_blanks   = Variable(torch.zeros((res.size(0), 1)).long() + blank_symbol)                 # (B, 1)
        queryseq_padded   = torch.cat((begining_blanks, res), 1)                         # (B, 2L+1)

        # for queryseq_mask
        if queryseq_mask is not None:
            queryseq_mask_extended = queryseq_mask.unsqueeze(2)                              # (B, L) -> (B, L, 1)
            concat                 = torch.cat((queryseq_mask_extended, queryseq_mask_extended), 2)  # concat.shape = (B, L, 2)
            res                    = concat.view(concat.size(0), concat.size(1) * concat.size(2))
            begining_blanks        = Variable(torch.ones((res.size(0), 1)).byte())
            queryseq_mask_padded   = torch.cat((begining_blanks, res), 1)        # (B, 2L+1)
        else:
            queryseq_mask_padded   = None
        return queryseq_padded, queryseq_mask_padded

    @staticmethod
    def _class_batch_to_labeling_batch(queryseq_padded, scorematrix, scorematrix_mask=None):
        """
        Convert dimension 'class' of scorematrix to 'label'
        :param queryseq_padded: (2L+1, B)   -> (B, 2L+1)     LongTensor
        :param scorematrix: (T, C+1, B)     -> (B, T, C+1)   FloatTensor
        :param scorematrix_mask: (T, B)     -> (B, T)        ByteTensor
        :return: (T, 2L+1, B)               -> (B, T, 2L+1)  FloatTensor
        """
        if scorematrix_mask is not None:
            scorematrix = scorematrix * scorematrix_mask.unsqueeze(2).expand_as(scorematrix).float()
        B, T = scorematrix.size(0), scorematrix.size(1)
        L2   = queryseq_padded.size(1)
        res  = scorematrix.gather(2, queryseq_padded.unsqueeze(1).expand(B, T, L2)) # (B, T, 2L+1), indexing each row of scorematrix with queryseq_padded
        return res

    @staticmethod
    def _recurrence_relation(queryseq_padded, queryseq_mask_padded=None, blank_symbol=None):
        """
        Generate structured matrix r2 & r3 for dynamic programming recurrence
        :param queryseq_padded: (2L+1, B)         (B, 2L+1)          LongTensor
        :param queryseq_mask_padded: (2L+1, B)    (B, 2L+1)          ByteTensor
        :param blank_symbol: = C                                     Integer
        :return: r2 (2L+1, 2L+1), r3 (2L+1, 2L+1, B)   (B, 2L+1, 2L+1)  both FloatTensor
        """
        B, L2 = queryseq_padded.size()                                                      # = 2L+1
        blanks = Variable(torch.zeros(queryseq_padded.size(0), 2).long() + blank_symbol)                     # (B, 2)
        ybb = torch.cat((queryseq_padded, blanks), 1)                                       # (B, 2L+3)
        sec_diag = (1 - torch.eq(ybb[:, :-2], ybb[:, 2:])) * torch.eq(ybb[:, 1:-1], blank_symbol)  # (B, 2L+1)
        if queryseq_mask_padded is not None:
            sec_diag = sec_diag * queryseq_mask_padded
        r2 = Variable(torch.Tensor(np.eye(L2, L2, 1)))                                                # upper diagonal matrix (2L+1, 2L+1)
        r3 = Variable(torch.Tensor(np.eye(L2, L2, 2)).unsqueeze(2).expand(L2, L2, B)) * sec_diag.float().transpose(0,1).unsqueeze(1).expand(L2,L2,B)   # until [3-17-2017], pytorch does not support dimension broadcast, use expand here
        return r2, r3  # (2L+1, 2L+1, B)

if __name__ == '__main__':
    import numpy as np, time
    import theano
    from lasagne_ext.objectives import CTC_Logscale
    from theano import tensor
    from torch.autograd import Variable
    # from ctc import best_path_decode
    # np.random.seed(33)
    B = 2
    C = 5
    L = 3
    T = 4
    x1, x2, x3, x4, x5 = tensor.imatrix(name='queryseq'), \
                         tensor.tensor3(dtype='float32', name='scorematrix'), \
                         tensor.fmatrix(name='queryseq_mask'),\
                         tensor.fmatrix(name='scorematrix_mask'), \
                         tensor.iscalar(name='blank_symbol')



    scorematrix = np.random.rand(T, C + 1, B).astype(np.float32)
    query       = np.random.randint(0, C, (L, B)).astype(np.int32)
    query_mask  = np.random.rand(L, B) > 0.5
    sm_mask     = np.random.rand(T, B) > 0.5




    result = CTC_Logscale.cost(x1, x2, x3, x4, x5, align='post')
    f2 = theano.function([x1, x2, x3, x4, x5], result)

    time2 = time.time()
    result = f2(query, scorematrix, query_mask.astype(np.float32), sm_mask.astype(np.float32), C)
    print('theano:', result)
    time3 = time.time()

    time0 = time.time()
    ctc = CTC_Log(align='post')
    sm_v = Variable(torch.Tensor(scorematrix).transpose(1,2).transpose(0,1), requires_grad=True)
    q_v  = Variable(torch.LongTensor(query.astype(np.int64)).transpose(0,1))
    q_mask_v = Variable(torch.ByteTensor(query_mask.astype(np.uint8)).transpose(0,1))
    sm_mask_v = Variable(torch.ByteTensor(sm_mask.astype(np.uint8)).transpose(0,1))


    result_torch = ctc.cost(q_v, sm_v, q_mask_v, sm_mask_v, C)
    print('torch:', result_torch.data.numpy())
    time1 = time.time()



    print('Time = %0.6fs | %0.6fs' % (time1-time0, time3-time2))

    result_torch.backward()
    print(sm_v.grad)