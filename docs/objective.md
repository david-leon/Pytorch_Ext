## CTC
 A pure pytorch implementation of CTC (Connectionist Temoral Classification) objective.
 Equivalent to its Theano counterpart [`dandelion.objective.CTC_Logscale()`](https://github.com/david-leon/Dandelion/blob/master/dandelion/ctc_theano.py)
 The `cost()` function of `CTC_Log` class returns the average NLL over a batch samples given query sequences and score matrices.

```python
class CTC_Log(eps=1E-12, inf=1E12, align='pre')
```
* **align**: {'pre' or 'post'}, indicating how samples in a batch are aligned.

```python
.cost(queryseq, scorematrix, queryseq_mask=None, scorematrix_mask=None, blank_symbol=None)
```
Compute CTC cost, using only the forward pass

* **queryseq**:           (B, L)          LongTensor
* **scorematrix**:        (B, T, C+1)     FloatTensor
* **queryseq_mask**:      (B, L)          ByteTensor
* **scorematrix_mask**:   (B, T)          ByteTensor
* **blank_symbol**: scalar, = C by default  Integer
* **return**: negative log likelihood averaged over a batch
_______________________________________________________________________

