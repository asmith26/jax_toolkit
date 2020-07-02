# Losses and Metrics

Loss functions are normally minimised (e.g. for learning/optimising a model), and metrics are normally maximised (e.g for further evaluating the performance of a model). All loss and metric functions have the form:

```python
function_name(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray
```

We have attempted to currently include only functions here that are quite robust for a wide range of problems (e.g. not too affected by unbalanced classes). Of course the choice of loss/metric function(s) are often quite dependent to a specific problem.

## Classification
#### Losses
| Name | Notes |
|---|---|
| [log_loss](https://github.com/asmith26/jax_toolkit/blob/master/jax_toolkit/loss.py#L9) (aka. binary/multiclass log loss or binary/categorical crossentropy) | |
| [hinge] | |
| [squared_hinge] | |
| [kullback_leibler_divergence] | |

#### Metrics
| Name | Notes |
|---|---|
| [balanced_accuracy] | |
| [intersection_over_union] (aka. Jaccard Index) | |
| [matthews_correlation_coefficient] | |


## Regression
#### Losses
| Name | Notes |
|---|---|
| [mean_absolute_error] | - Good interpretability, thus useful for displaying/explaining results. |
| [median_absolute_error] | - Good interpretability, thus useful for displaying/explaining results.<br/>- Median can be more robust that the mean (e.g the mean number of legs a dog has is less than 4, whilst the median is 4). |
| [max_absolute_error] | - Good interpretability, thus useful for displaying/explaining results. |
| [mean_squared_error] | |
| [mean_squared_log_error] | |

#### Metrics
| Name | Notes |
|---|---|
| [r_squared] | |
