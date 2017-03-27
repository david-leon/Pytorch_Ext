# coding:utf-8
__author__ = 'dawei.leng (David Leon)'
__version__ = '1.48'

"""
 A pure pytorch implementation of CTC (Connectionist Temoral Classification) objective.
 Equivalent to its Theano counterpart lasagne_ext.objectives.CTC_Logscale() [https://github.com/david-leon/Lasagne_Ext/blob/master/lasagne_ext/ctc_theano.py]
 The `cost()` function of `CTC_Log` class returns the average NLL over a batch samples given query sequences and score matrices.

 This implementation features:
    1) batch / mask supported.
    2) decoding & CER calculation supported
    3) pure pytorch

 Created   :   3, 10, 2017
 Revised   :   3, 24, 2017   ver 1.48  add document

 Reference :  [1] Alex Graves, etc., Connectionist temporal classification: labelling unsegmented sequence data with
                  recurrent neural networks, ICML, 2006
              [2] Alex Graves, Supervised sequence labelling with recurrent neural networks, 2014
              [3] Lawrence R. Rabiner, A tutorial on hidden Markov models and selected applications in speech recognition,
                  Proceedings of the IEEE, 1989
              [4] Maas Andrew, etc., https://github.com/amaas/stanford-ctc/blob/master/ctc_fast/ctc-loss/ctc_fast.pyx
              [5] Mohammad Pezeshki, https://github.com/mohammadpz/CTC-Connectionist-Temporal-Classification/blob/master/ctc_cost.py
              [6] Shawn Tan, https://github.com/shawntan/rnn-experiment/blob/master/CTC.ipynb
"""
import numpy as np
import torch
from torch.autograd import Variable

class CTC_Log(object):
    """
    This implementation uses log scale computation.
    Batch supported.
    B: batch size
    L: query sequence length (maximum length of a batch)
    C: class number
    T: time length (maximum time length of a batch)
    """
    def __init__(self, eps=1E-12, inf=1E12, align='pre', device=None):
        """
        :param eps:
        :param inf:
        :param align: {'pre' or 'post'}, indicating how samples in a batch are aligned.
        :param device: run on which device (due to pytorch limitation, device must be specified explicitly; it would be better if transparent on any device)
        """
        super(CTC_Log, self).__init__()
        self.eps = eps
        self.inf = inf
        self.align = align
        self.device = device     # < 0 means cpu, >=0 means gpu

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
        if self.device is None:
            if not scorematrix.is_cuda():
                self.device = -1
            else:
                self.device = scorematrix.get_device()
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
        :param queryseq_padded:        -> (B, 2L+1)     LongTensor
        :param scorematrix:            -> (B, T, C+1)   FloatTensor
        :param queryseq_mask_padded:   -> (B, 2L+1)     ByteTensor
        :param scorematrix_mask:       -> (B, T)        ByteTensor
        :param blank_symbol: = C by default             Integer
        :return:
        """
        if self.device is None:
            if not scorematrix.is_cuda():
                self.device = -1
            else:
                self.device = scorematrix.get_device()
        if blank_symbol is None:
            blank_symbol = scorematrix.size(2) - 1
        if queryseq_mask_padded is None:
            queryseq_mask_padded = Variable(torch.ones(queryseq_padded.size()).byte())    # pytorch's tensor constructor sucks!
            if self.device >= 0:
                queryseq_mask_padded = queryseq_mask_padded.cuda(self.device)
        if scorematrix_mask is None:
            scorematrix_mask = Variable(torch.ones(scorematrix.size(0), scorematrix.size(1)).byte())
            if self.device >= 0:
                scorematrix_mask = scorematrix_mask.cuda(self.device)

        pred_y   = self._class_batch_to_labeling_batch(queryseq_padded, scorematrix, scorematrix_mask)  # (B, T, 2L+1), reshaped scorematrix
        r2, r3   = self._recurrence_relation(queryseq_padded, queryseq_mask_padded, blank_symbol)       # r2 (2L+1, 2L+1), r3 (2L+1, 2L+1, B)
        pred_y   = self._epslog(pred_y)
        B, T, L2 = pred_y.size()
        p_prev   = Variable(self._epslog(torch.eye(1, L2).expand(B,L2) * torch.ones(B, L2)))            # (B, 2L+1)
        if self.device >= 0:
            p_prev = p_prev.cuda(self.device)
        queryseq_mask_padded = queryseq_mask_padded.float()
        if self.align == 'pre':
            alphas = Variable(torch.Tensor(T, B, L2))                # (T, B, 2L+1)
            if self.device >= 0:
                alphas = alphas.cuda(self.device)
        for i in range(T):
            p_curr = pred_y[:, i, :]                                 # (B, 2L+1)
            p2     = self._log_dot_matrix(p_prev, r2)                # (B, 2L+1)
            p3     = self._log_dot_tensor(p_prev, r3)                # (B, 2L+1)
            p123   = self._log_add(p3, self._log_add(p_prev, p2))    # (B, 2L+1)
            p_prev = p_curr + p123 + self._epslog(queryseq_mask_padded)
            if self.align == 'pre':
                alphas[i, :, :] = p_prev

        LL  = torch.sum(queryseq_mask_padded, 1).long().squeeze()     # (B,)
        # Idx = torch.range(0, B-1)                           # (B,)

        if self.align == 'pre':
            TL = torch.sum(scorematrix_mask, 1).long().squeeze()      # (B,)
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

    def _pad_blanks(self, queryseq, blank_symbol, queryseq_mask=None):
        """
        Pad queryseq and corresponding queryseq_mask with blank symbol
        :param queryseq         -> (B, L)    LongTensor
        :param queryseq_mask    -> (B, L)    ByteTensor
        :param blank_symbol  integer scalar
        :return queryseq_padded, queryseq_mask_padded, both with shape (B, 2L+1)   LongTensor, ByteTensor
        """
        # for queryseq
        queryseq_extended = queryseq.unsqueeze(2)                                         # (B, L) -> (B, L, 1)
        blanks            = Variable(torch.zeros(queryseq_extended.size()).long() + blank_symbol)         # (B, L, 1)
        if self.device >= 0:
            blanks = blanks.cuda(self.device)
        concat            = torch.cat((queryseq_extended, blanks), 2)                     # concat.shape = (B, L, 2)
        res               = concat.view(concat.size(0), concat.size(1) * concat.size(2))  # (B, 2L), the reshape will cause the last 2 dimensions interlace
        begining_blanks   = Variable(torch.zeros((res.size(0), 1)).long() + blank_symbol) # (B, 1)
        if self.device >= 0:
            begining_blanks = begining_blanks.cuda(self.device)
        queryseq_padded   = torch.cat((begining_blanks, res), 1)                          # (B, 2L+1)

        # for queryseq_mask
        if queryseq_mask is not None:
            queryseq_mask_extended = queryseq_mask.unsqueeze(2)                           # (B, L) -> (B, L, 1)
            concat                 = torch.cat((queryseq_mask_extended, queryseq_mask_extended), 2)  # concat.shape = (B, L, 2)
            res                    = concat.view(concat.size(0), concat.size(1) * concat.size(2))
            begining_blanks        = Variable(torch.ones((res.size(0), 1)).byte())
            if self.device >= 0:
                begining_blanks = begining_blanks.cuda(self.device)
            queryseq_mask_padded   = torch.cat((begining_blanks, res), 1)                 # (B, 2L+1)
        else:
            queryseq_mask_padded   = None
        return queryseq_padded, queryseq_mask_padded

    def _class_batch_to_labeling_batch(self, queryseq_padded, scorematrix, scorematrix_mask=None):
        """
        Convert dimension 'class' of scorematrix to 'label'
        :param queryseq_padded:      -> (B, 2L+1)     LongTensor
        :param scorematrix:          -> (B, T, C+1)   FloatTensor
        :param scorematrix_mask:     -> (B, T)        ByteTensor
        :return:                     -> (B, T, 2L+1)  FloatTensor
        """
        if scorematrix_mask is not None:
            scorematrix = scorematrix * scorematrix_mask.unsqueeze(2).expand_as(scorematrix).float()
        B, T = scorematrix.size(0), scorematrix.size(1)
        L2   = queryseq_padded.size(1)
        res  = scorematrix.gather(2, queryseq_padded.unsqueeze(1).expand(B, T, L2)) # (B, T, 2L+1), indexing each row of scorematrix with queryseq_padded
        return res

    def _recurrence_relation(self, queryseq_padded, queryseq_mask_padded=None, blank_symbol=None):
        """
        Generate structured matrix r2 & r3 for dynamic programming recurrence
        :param queryseq_padded:          (B, 2L+1)          LongTensor
        :param queryseq_mask_padded:     (B, 2L+1)          ByteTensor
        :param blank_symbol: = C                            Integer
        :return: r2 (2L+1, 2L+1), r3 (2L+1, 2L+1, B)   (B, 2L+1, 2L+1)  both FloatTensor
        """
        B, L2 = queryseq_padded.size()                                                      # = 2L+1
        blanks = Variable(torch.zeros(queryseq_padded.size(0), 2).long() + blank_symbol)    # (B, 2)
        if self.device >= 0:
            blanks = blanks.cuda(self.device)
        ybb = torch.cat((queryseq_padded, blanks), 1)                                       # (B, 2L+3)
        sec_diag = (1 - torch.eq(ybb[:, :-2], ybb[:, 2:])) * torch.eq(ybb[:, 1:-1], blank_symbol)  # (B, 2L+1)
        if queryseq_mask_padded is not None:
            sec_diag = sec_diag * queryseq_mask_padded
        r2 = Variable(torch.Tensor(np.eye(L2, L2, 1)))                                             # upper diagonal matrix (2L+1, 2L+1)
        r3 = Variable(torch.Tensor(np.eye(L2, L2, 2)).unsqueeze(2).expand(L2, L2, B))
        if self.device >= 0:
            r2 = r2.cuda(self.device)
            r3 = r3.cuda(self.device)
        r3 = r3 * sec_diag.float().transpose(0,1).unsqueeze(1).expand(L2,L2,B)   # until [3-17-2017], pytorch does not support dimension broadcast, use expand here
        return r2, r3  # (2L+1, 2L+1, B)

    @staticmethod
    def best_path_decode(scorematrix, scorematrix_mask=None, blank_symbol=None):
        """
        Computes the best path by simply choosing most likely label at each timestep
        :param scorematrix:      (B, T, C+1)    FloatTensor
        :param scorematrix_mask: (B, T)         ByteTensor
        :param blank_symbol: = C by default     Integer
        :return: resultseq (B, T)               LongTensor
                 resultseq_mask(B, T)           ByteTensor
        """
        B, T, C1          = scorematrix.size()
        if blank_symbol is None:
            blank_symbol = C1 - 1
        bestp, bestlabels = scorematrix.max(2)      # (B, T, 1)
        resultseq         = bestlabels.squeeze()
        if scorematrix_mask is None:
            resultseq_mask = scorematrix.new(B, T).type(torch.ByteTensor).zero_() + 1
        else:
            resultseq_mask    = scorematrix_mask.clone()

        for i in range(B):
            for j in range(T):
                if resultseq[i, j] == blank_symbol:
                    resultseq_mask[i, j] = 0
                    continue
                if j!=0 and resultseq[i, j] == resultseq[i, j-1]:
                    resultseq_mask[i, j] = 0
                    continue
        return resultseq, resultseq_mask

    @staticmethod
    def _editdist(s, t):
        """
        :param s, t : both LongTensor with size = (L,)
        :return: integer
        """
        ns, nt = s.numel(), t.numel()
        if s.numel() == 0:
            return nt
        elif t.numel() == 0:
            return ns
        v0 = s.new(range(nt+1))
        v1 = s.new(nt+1).zero_()
        for i in range(ns):
            v1[0] = i + 1
            for j in range(nt):
                cost = 0 if s[i] == t[j] else 1
                v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
            for j in range(nt+1):
                v0[j] = v1[j]
        return v1[t.numel()]

    def calc_CER(self, resultseq, targetseq, resultseq_mask=None, targetseq_mask=None):
        """
         Calculate the character error rate (CER) given ground truth 'targetseq' and CTC decoding output 'resultseq'
         :param resultseq      (B, T1)          LongTensor
         :param resultseq_mask (B, T1)          ByteTensor
         :param targetseq      (B, T2)          LongTensor
         :param targetseq_mask (B, T2)          ByteTensor
         :return: (CER, Total_Error_Len, Total_GT_Len)
         """
        assert(resultseq.size(0) == targetseq.size(0))
        B = resultseq.size(0)
        total_gt_len, total_ed = 0, 0
        for i in range(B):
            s = resultseq[i,:]
            if resultseq_mask is not None:
                s = s.masked_select(resultseq_mask[i,:])
            t = targetseq[i,:]
            if targetseq_mask is not None:
                t = t.masked_select(targetseq_mask[i,:])
            ed = self._editdist(s, t)
            total_ed += ed
            total_gt_len += t.numel()
        CER = total_ed / total_gt_len * 100.0
        return CER, total_ed, total_gt_len


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


