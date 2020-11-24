## dropout
Dropout function for any dimension input.
```python
dropout(input, p=0.5, shared_axes=(), rescale=True)
```
* **input**: tensor to be applied.
* **p**: float, probability to drop a value (set to zero)
* **shared_axes**: tuple of int, axes to share the dropout mask over. By default, each value is dropped individually. For example, shared_axes=(0,) means using the same mask across the batch. shared_axes=(2, 3) means using the same mask across the spatial dimensions of 2D feature maps, i.e., drop channels.
* **rescale**: if `True` (default), the input tensor will be rescaled by `1-p` to compensate mean fluctuation due to dropout
_______________________________________________________________________

