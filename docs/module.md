## BatchNorm
Batch normalization for any dimention input, adapted from [Dandelion](https://github.com/david-leon/Dandelion)'s BatchNorm class. The normalization is done as 
$$
\begin{align}  
x' = \gamma * \frac{(x-\mu)}{\sigma} + \beta
\end{align}
$$
You can fabricate nonstandard BN variant by disabling any parameter among {$\mu$, $\sigma$, $\gamma$, $\beta$}

```python
class BatchNorm(input_shape=None, axes='auto', eps=1e-5, alpha=0.01, 
                beta=0.0, gamma=1.0, mean=0.0, inv_std=1.0)
```
* **input_shape**: tuple or list of ints or tensor. Input shape of `BatchNorm` module, including batch dimension. 
* **axes**: `auto` or tuple of int. The axis or axes to normalize over. If `auto` (the default), normalize over all axes except for the second: this will normalize over the minibatch dimension for dense layers, and additionally over all spatial dimensions for convolutional layers.
* **eps**: small constant 𝜖 added to the variance before taking the square root and dividing by it, to avoid numerical problems
* **alpha**: coefficient for the exponential moving average of batch-wise means and standard deviations computed during training; the closer to one, the more it will depend on the last batches seen
* **gamma, beta**: these two parameters can be set to `None` to disable the controversial scale and shift as well as save computing power. According to [Deep Learning Book, Section 8.7.1](http://www.deeplearningbook.org/contents/optimization.html), disabling $\gamma$ and $\beta$ *might* reduce the expressive power of the neural network.
* **mean, inv_std**: initial values for $\mu$ and $\frac{1}{\sigma}$. These two parameters can also be set to `None` to diable the mean substraction and variance scaling.

```python
.forward(x)
```
Use `self.training` attribute to switch between training mode and inference mode.

_______________________________________________________________________
## Center
Estimate class centers by moving averaging, adapted from [Dandelion](https://github.com/david-leon/Dandelion)'s Center class

```python
class Center(feature_dim, center_num, alpha=0.9, centers=None)
```
* **feature_dim**: feature dimension 
* **center_num**: class center number
* **center**: initialization of class centers, should be in shape of `(center_num, feature_dim)`
* * **alpha**: moving averaging coefficient, the closer to one, the more it will depend on the last batches seen: $C_{new} = \alpha*C_{batch} + (1-\alpha)*C_{old}$

```python
.forward(features=None, labels=None)
```
* **features**: batch features, from which the class centers will be estimated
* **labels**: `features`'s corresponding class labels
* **return**: centers estimated. Use `self.training` attribute to switch between training mode and inference mode. In training mode, `features` and `labels` are required for input; in inference mode these inputs will be ignored.

