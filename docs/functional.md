## dropout
Dropout function for any dimension input.
```python
dropout(input, p=0.5, shared_axes=(), rescale=True, fill_value=0.0)
```
* **input**: tensor to be applied.
* **p**: float scalar or list of floats, probability to drop a value (replaced with `fill_value`). If `p` is a list, the actual probability will be calculated in an interval of `p[i]` and `p[i+1]` randomly. By passing a list to `p`, dropout will be executed with varying probability. For example by passing `p = [0.1, 0.5]` to the function, the dropout probability will be varying at least 0.1 and at most 0.5.
* **shared_axes**: tuple of int, axes to share the dropout mask over. By default, each value is dropped individually. For example, shared_axes=(0,) means using the same mask across the batch. shared_axes=(2, 3) means using the same mask across the spatial dimensions of 2D feature maps, i.e., drop channels.
* **rescale**: if `True` (default), the input tensor will be rescaled by `1-p` to compensate mean fluctuation due to dropout
* **fill_value**: in our implementation, the *dropped* values will be replaced with `fill_value` (default = 0, this is equivalent to the behavior of pytorch's builtin `dropout()`)
_______________________________________________________________________

