## freeze_module
Freeze a module during training
```python
freeze_module(module)
```
* **module**: instance of torch.nn.Module
* **return**: no return
_______________________________________________________________________
## unfreeze_module
Un-freeze a module for training
```python
unfreeze_module(module)
```
* **module**: instance of torch.nn.Module
* **return**: no return

_______________________________________________________________________
## get_trainable_parameters
Retrieve only trainable parameters, for feeding optimizer
```python
get_trainable_parameters(module, with_name=False)
```
* **module**: instance of torch.nn.Module
* **with_name**: if `True`, output in format of (name, tensor), else only tensor returned
* **return**: generator of trainable parameters
  
_______________________________________________________________________
## set_value
Set tensor value with numpy array
```python
set_value(t, v)
```
* **t**: tensor
* **v**: numpy array
* **return**: no return

_______________________________________________________________________
## get_device
Retrieve device from tensor or module
```python
get_device(x)
```
* **x**: tensor or instance of nn.Module
* **return**: torch.device

_______________________________________________________________________
## torch_safe_run
Retrieve device from tensor or module
```python
torch_safe_run(fn, inputs)
```
* **fn**: function to run
* **inputs**: dict passed to function `fn`
* **return**: (`status`, `result`) in which `status` = 0 if no exception, = 1 if CUDA OOM exception occurred; `result` is as returned by calling `fn(**inputs)`
_______________________________________________________________________
## gpickle
Pickle with gzip compression enabled.
```python
.dump(data, filename, compresslevel=9, protocol=4)
```
Dump data and save to file.

* **data**: data to be dumped to file
* **filename**: file path
* **compresslevel**: gzip compression level, default = 9.
* **protocol**: protocol version of pickle, defalut = 4.

```python
.load(filename)
```
Load dumped data from file

* **filename**: file to be loaded
* **return**: data unpickled

```python
.dumps(data, compresslevel=9, protocol=4)
```  
Dump data into bytes

* **return**: data pickled & compressed into bytes

```python
.loads(zipped_bytes)
```
Load dumped data from bytes

* **return**: data unpickled

_______________________________________________________________________
## verbose_print
`print` with verbose level filtering
```python
class verbose_print(level=0, prefix=None)
```
* **level**: predefined verbose level. Instance of `verbose_print` functions the same with python's builtin `print()` with an additional `l` arg (default = 0); when `l` < this predefined verbose level, the print content will be suppressed, thus only content with verbose level >= `level` can be actually printed on screen.
* **prefix**: if given, each print will be preceded by this fixed prefix.

Example
```python
vprint = verbose_print(level=2, prefix='LMExp')
vprint('this line will be actually printed', l=3)
vprint('this line will NOT be printed by verbose level filtering', l=0)
```


