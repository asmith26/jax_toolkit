# Accessors API

## df.ml. Methods
### `standard_scaler` *<small>[[source](https://github.com/asmith26/jax_toolkit/blob/master/jax_toolkit/ml/__init__.py#L13)]</small>*
`standard_scaler`*(<span style='color:green'>column</span>: <span style='color:blue'>str</span>) -> pd.Series*

**Parameters**
> **column:** Column denoting feature to standardize.

**Returns**
> Standardized featured by removing the mean and scaling to unit variance: `z = (x - u) / s`.

Examples
```python
>>> df = pd.DataFrame({"x": [0, 1],
                       "y": [0, 1]},
                       index=[0, 1])
>>> df["standard_scaler_x"] = df.ml.standard_scaler(column="x")
>>> df["standard_scaler_x"]
pd.Series([-1, 1])
```

### `train_validation_split` *<small>[[source](https://github.com/asmith26/jax_toolkit/blob/master/jax_toolkit/ml/__init__.py#L37)]</small>*
`train_validation_split`*(<span style='color:green'>train_frac</span>: <span style='color:blue'>float</span>, <span style='color:green'>random_seed</span>: <span style='color:blue'>int = None</span>) -> Tuple[pd.DataFrame, pd.DataFrame]*

**Parameters**
> **train_frac:** Fraction of rows to be added to df_train.

> **random_seed:** Seed for the random number generator (e.g. for reproducible splits).

**Returns**
> df_train and df_validation, split from the original dataframe.

Examples
```python
>>> df = pd.DataFrame({"x": [0, 1, 2],
                       "y": [0, 1, 2]},
                       index=[0, 1, 2])
>>> df_train, df_validation = df.ml.train_validation_split(train_frac=2/3)
>>> df_train
pd.DataFrame({"x": [0, 1], "y": [0, 1]}, index=[0, 1]),
>>> df_validation
pd.DataFrame({"x": [2], "y": [2]}, index=[2])
```

## df.nn. Methods
### `init` *<small>[[source](https://github.com/asmith26/jax_toolkit/blob/master/jax_toolkit/nn/__init__.py#L38)]</small>*
`init`*(<span style='color:green'>x_columns</span>: <span style='color:blue'>List[str]</span>, <span style='color:green'>y_columns</span>: <span style='color:blue'>List[str]</span>, <span style='color:green'>net_function</span>: <span style='color:blue'>Callable[[jnp.ndarray] jnp.ndarray]</span>, <span style='color:green'>loss</span>: <span style='color:blue'>str</span>, <span style='color:green'>optimizer</span>: <span style='color:blue'>InitUpdate = optix.adam(learning_rate=1e-3)</span>, <span style='color:green'>batch_size</span>: <span style='color:blue'>int = None</span>) -> pd.DataFrame*

**Parameters**
> **x_columns:** Columns to be used as input for the model.

> **y_columns:** Columns to be used as output for the model.

> **net_function:** A function that defines a haiku.Sequential neural network and how to predict uses it (this
function is passed to hk.transform). This should have the signature *net_function(x: jnp.ndarray) ->
jnp.ndarray*.

> **loss:** Loss function to use. See available loss functions in [jax_toolkit]().

> **optimizer:** Optimizer to use. See [jax](https://jax.readthedocs.io/en/latest/jax.experimental.optix.html).

> **batch_size:** Batch size to use. If not specified, the number of rows in the entire dataframe is used.

**Returns**
> A pd.DataFrame containing a neural network model ready for training with jax_toolkit.

Examples
```python
>>> def net_function(x: jnp.ndarray) -> jnp.ndarray:
...     net = hk.Sequential([relu])
...     predictions: jnp.ndarray = net(x)
...     return predictions
>>> df_train = df_train.nn.init(x_columns=["x"],
...                             y_columns=["y"],
...                             net_function=net_function,
...                             loss="mean_squared_error")
>>> for _ in range(10):  # num_epochs
...     df_train = df_train.nn.update(df_validation=df_validation)
```

### `get_model` *<small>[[source](https://github.com/asmith26/jax_toolkit/blob/master/jax_toolkit/nn/__init__.py#L93)]</small>*
`get_model`*() -> Model*

 **Returns**
 > A jax_toolkit.nn.Model object. As this is not linked to a pd.DataFrame, it is much more lightweight
 and could be used in e.g. a production setting.

Examples
```python
>>> model = df_train.get_model()
>>> model.predict(x=jnp.ndarray([42]))
```

### `hvplot_losses` *<small>[[source](https://github.com/asmith26/jax_toolkit/blob/master/jax_toolkit/nn/__init__.py#L107)]</small>*
`hvplot_losses`*() -> None*

**Returns**
> A Holoviews object for interactive (Bokeh backend), real-time ploting of training and validation loss
curves. For an example usage, see [this notebook](
https://github.com/asmith26/jax_toolkit/blob/master/notebooks/sine.ipynb)

Examples
```python
>>> df_train.nn.hvplot_losses()
```

### `update` *<small>[[source](https://github.com/asmith26/jax_toolkit/blob/master/jax_toolkit/nn/__init__.py#L136)]</small>*
`update`*(<span style='color:green'>df_validation_to_plot</span>: <span style='color:blue'>pd.DataFrame = None</span>) -> pd.DataFrame*

**Parameters**
> **df_validation_to_plot:** Validation data to evaluate and update loss curve with.

**Returns**
> A pd.DataFrame containing an updated neural network model (trained on one extra epoch).

Examples
```python
>>> for _ in range(10):  # num_epochs
...     df_train = df_train.nn.update(df_validation_to_plot=df_validation)
```

### `predict` *<small>[[source](https://github.com/asmith26/jax_toolkit/blob/master/jax_toolkit/nn/__init__.py#L169)]</small>*
`predict`*(<span style='color:green'>x_columns</span>: <span style='color:blue'>List[str] = None</span>) -> pd.Series*

**Parameters**
> **x_columns:** Columns to predict on. If None, the same x_columns names used to train the model are used.

**Returns**
> A pd.Series of predictions.

Examples
```python
>>> df_new = pd.DataFrame({"x": [-10, -5, 22]})
>>> df_new.model = df_train.nn.get_model()
>>> df_new["predictions"] = df_new.nn.predict()
```

### `evaluate` *<small>[[source](https://github.com/asmith26/jax_toolkit/blob/master/jax_toolkit/nn/__init__.py#L193)]</small>*
`evaluate`*(<span style='color:green'>x_columns</span>: <span style='color:blue'>List[str] = None</span>, <span style='color:green'>y_columns</span>: <span style='color:blue'>List[str] = None</span>) -> pd.Series*

**Parameters**
> **x_columns:** Columns to predict on. If None, the same x_columns names used to train the model are used.
> **y_columns:** Columns to compare predictions with. If None, the same y_columns names used to train the model
are used.

**Returns**
> A pd.Series of evalations.

Examples
```python
>>> df_test = pd.DataFrame({"x": [-1, 0, 1], "y": [0, 0, 1]})
>>> df_test.model = df_train.nn.get_model()
>>> df_test["evaluation_loss"] = df_test.nn.evaluate()
```

