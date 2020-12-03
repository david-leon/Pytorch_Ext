# History

## version 0.9.2 [12-3-2020]
* **MODIFIED**: replace `torch.clamp()` with in-place `Tensor.clamp_()` for `util.grad_clip()`

## version 0.9.1 [11-25-2020]
* **NEW**: `functional.dropout()` now supports 1) varying dropout probability 2) replace dropped values with given `fill_value`

## version 0.9.0 [11-24-2020]
* **NEW**: add online documentation
* **NEW**: add `functional.dropout()`

## version 0.8.0 [9-8-2020]
* **NEW**: add `BatchNorm` module for replacing pytorch's implementation

## version 0.7.6 [5-7-2020]
* **MODIFIED**: add support for gzip compression for `gpickle.loads()` & `gpickle.dumps()` functions

## version 0.7.5 [1-8-2020]
* **NEW**: add `util.get_file_md5()` for file md5 check

## version 0.7.4 [1-3-2020]
* **NEW**: add `util.set_value()` for tensor initialization with numpy array

## version 0.7.3 [12-26-2019]
* **NEW**: Add `util.freeze_module()/unfreeze_module()/get_trainable_parameters()` for training parameters handling

## version 0.7.2 [11-5-2019]
* **NEW**: Add `util.verbose_print()` for `print` with verbose level filtering 
