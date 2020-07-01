from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@pd.api.extensions.register_dataframe_accessor("ml")
class MachineLearningAccessor:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def standard_scaler(self, column: str) -> pd.Series:
        """
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
        """
        s = self._df[column]
        scaler = StandardScaler()
        arr_scaled_col: np.ndarray = scaler.fit_transform(s.values.reshape(-1, 1))
        s_scaled_col = pd.Series(data=arr_scaled_col.flatten(), index=self._df.index, dtype=s.dtype)
        return s_scaled_col

    def train_validation_split(self, train_frac: float, random_seed: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
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
        """
        df_train = self._df.sample(frac=train_frac, random_state=random_seed)
        df_validation = self._df.drop(labels=df_train.index)
        return df_train, df_validation
