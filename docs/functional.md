## dropout
Pooling 1 dimension along the given axis, support for any dimensional input.
```python
pool_1d(x, ws=2, ignore_border=True, stride=None, pad=0, mode='max', axis=-1)
```
* **ws**: scalar int. Factor by which to downsample the input
* **ignore_border**: bool. When `True`, dimension size=5 with `ws`=2 will generate a dimension size=2 output. 3 otherwise.
* **stride**: scalar int. The number of shifts over rows/cols to get the next pool region. If stride is None, it is considered equal to ws (no overlap on pooling regions), eg: `stride`=1 will shifts over one row for every iteration.
* **pad**: pad zeros to extend beyond border of the input
* **mode**: {`max`, `sum`, `average_inc_pad`, `average_exc_pad`}. Operation executed on each window. `max` and `sum` always exclude the padding in the computation. `average` gives you the choice to include or exclude it.
* **axis**: scalar int. Specify along which axis the pooling will be done

_______________________________________________________________________

