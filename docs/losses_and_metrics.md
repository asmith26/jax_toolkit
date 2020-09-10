# Losses and Metrics

The `lossses` and `metrics` modules contain some commonly used such functions implemented with jax to benefit from [jax primitives](https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html) like `grad` and `jit`. The documentation below also lists some commonly used such metrics that we have not implemented in jax, and we instead point readers to well-used implementations (often scikit-learn, marked with an asterisk *). Such functions have not been implemented because we believe they will not be required to work with the [jax primitives](https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html) (e.g. metrics do not often need to be differentiated) - of course if you feel this library would benefit from such implementations, contributions are very welcome. 

Loss functions are normally minimised (e.g. for learning/optimising a model), and metrics are normally maximised (e.g for further evaluating the performance of a model). Most loss and metric functions have been designed to work in a similar way to [scikit-learn metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics) if available, otherwise [tensorflow](https://www.tensorflow.org/api_docs/python/tf)/[tensorflow addons](https://www.tensorflow.org/addons/api_docs/python/tfa/) (e.g. same names, similar implementations), and have the form:

```python
function_name(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray
``` 


## Bounding Box
### Losses
```python
from jax_toolkit.losses.bounding_box import giou_loss

giou_loss(boxes1: jnp.ndarray, boxes2: jnp.ndarray) -> jnp.ndarray
# boxes are encoded as [y_min, x_min, y_max, x_max], e.g. jnp.array([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
```

| Name | Notes |
|---|---|
| [giou_loss](https://github.com/asmith26/jax_toolkit/blob/master/jax_toolkit/losses/bounding_box.py#L6) | Generalized Intersection over Union (GIoU) is designed to improve on intersection_over_union metric (benefits include being differentiable, so can use to train neural networks). More benefits and details can be found [here](https://giou.stanford.edu/). |


## Classification
#### Losses
```python
from jax_toolkit.losses.classification import LOSS_FUNCTION
```

| Name | Notes |
|---|---|
| [log_loss](https://github.com/asmith26/jax_toolkit/blob/master/jax_toolkit/losses/classification.py#L30) (aka. binary/multi-class log loss or binary/categorical crossentropy) | This applies a large penalty for confident (i.e. with probability 1) wrong predictions. |
| [squared_hinge](https://github.com/asmith26/jax_toolkit/blob/master/jax_toolkit/losses/classification.py#L37) | This has been shown to converge faster, provide better performance and be more robust to noise (see [this paper](https://arxiv.org/abs/1702.05659)). Expects `y_true` to be binary or multiclass classifications in the set {-1, +1}. |
| [sigmoid_focal_crossentropy](https://github.com/asmith26/jax_toolkit/blob/master/jax_toolkit/losses/classification.py#L75) | Shown to be useful for classification when you have highly imbalanced classes (e.g. for ["object detection where the imbalance between the background class and other classes is extremely high"](https://www.tensorflow.org/addons/api_docs/python/tfa/losses/SigmoidFocalCrossEntropy)). |


y_true = 0 | y_true = 1
:---:|:---:
![](img/comparing_loss_functions_when_y_true=0.png)  |  ![](img/comparing_loss_functions_when_y_true=1.png)

![](img/colour_schemes.png)

Loss | Plot
:---:|:---:
log_loss | ![](img/log_loss.png)
squared_hinge | ![](img/squared_hinge.png)
sigmoid_focal_crossentropy | ![](img/sigmoid_focal_crossentropy.png)

#### Metrics
```python
from jax_toolkit.metrics.classification import LOSS_FUNCTION
```

| Name | Notes |
|---|---|
| [balanced_accuracy_score*](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn-metrics-balanced-accuracy-score) | Good interpretability, thus useful for displaying/explaining results. |
| [intersection_over_union](https://github.com/asmith26/jax_toolkit/blob/master/jax_toolkit/metrics/classification.py#L14) (aka. Jaccard Index) | Useful for image segmentation problems, including for handling imbalanced classes (it gives all classes equal weight). |
| [matthews_correlation_coefficient*](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html#sklearn-metrics-matthews-corrcoef) | - Lots of symmetry (none of True/False Positives/Negatives are more important over another).<br/>- Good interpretability<br/>&nbsp;&nbsp;1 := perfect prediction<br/>&nbsp;&nbsp;0 := random prediction<br/>&nbsp;&nbsp;−1 := total disagreement between prediction & observation) |


## Probabilistic
```python
from jax_toolkit.losses.probabilistic import kullback_leibler_divergence
```

#### Losses
| Name | Notes |
|---|---|
| [kullback_leibler_divergence](https://github.com/asmith26/jax_toolkit/blob/master/jax_toolkit/losses/probabilistic.py#L6) | Measure how the probability distributions of y_true and y_pred differ (0 := identical). Often used in generative modelling. |


## Regression
#### Losses
```python
from jax_toolkit.losses.regression import LOSS_FUNCTION
```

| Name | Notes |
|---|---|
| [mean_absolute_error](https://github.com/asmith26/jax_toolkit/blob/master/jax_toolkit/losses/regression.py#L6) | Good interpretability, thus useful for displaying/explaining results. |
| [median_absolute_error](https://github.com/asmith26/jax_toolkit/blob/master/jax_toolkit/losses/regression.py#L13) | - Good interpretability.<br/>- Median can be more robust that the mean (e.g the mean number of legs a dog has is less than 4, whilst the median is 4). |
| [max_absolute_error](https://github.com/asmith26/jax_toolkit/blob/master/jax_toolkit/losses/regression.py#L20) | Good interpretability. |
| [mean_squared_error](https://github.com/asmith26/jax_toolkit/blob/master/jax_toolkit/losses/regression.py#L27) | Relatively simple and (mathematically) convenient. |
| [mean_squared_log_error](https://github.com/asmith26/jax_toolkit/blob/master/jax_toolkit/losses/regression.py#L32) | For problems where y_true has a wide spread or large values, this does not punish a model as heavily as mean squared error. |

#### Metrics
```python
from jax_toolkit.metrics.regression import LOSS_FUNCTION
```

| Name | Notes |
|---|---|
| [r2_score](https://github.com/asmith26/jax_toolkit/blob/master/jax_toolkit/metrics/regression.py#L6) | Indication of goodness of fit.<br/>- `0` := constant model that always predicts the mean of y<br/>- `1` := perfect fit |


## Utils
If you are familiar with [haiku](https://github.com/deepmind/dm-haiku), a JAX-based neural network library, you can use the [`get_haiku_loss_function()`](https://github.com/asmith26/jax_toolkit/blob/master/jax_toolkit/losses/utils.py#L44) function to get a loss from jax_toolkit that can be used with haiku:

```python
import haiku as hk
import jax
import jax.numpy as jnp

def net_function(x: jnp.ndarray) -> jnp.ndarray:
    net = hk.Sequential([
        ...
    ])
    predictions: jnp.ndarray = net(x)
    return predictions
net_transform = hk.transform(net_function)

loss_function = get_haiku_loss_function(net_transform, loss="mean_squared_error")

# Train model,
...
grads = jax.grad(loss_function)(params, x, y)
...
``` 


## Useful resources
[1] [24 Evaluation Metrics for Binary Classification (And When to Use Them)](https://neptune.ai/blog/evaluation-metrics-binary-classification)
