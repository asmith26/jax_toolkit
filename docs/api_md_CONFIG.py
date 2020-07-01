import pandas_toolkit.ml
import pandas_toolkit.nn

ROOT_GITHUB_URL = "https://github.com/asmith26/pandas_toolkit/blob/master/pandas_toolkit/"
PACKAGE_NAME = "pandas_toolkit"

accessors = {
    "df.ml.": [
        pandas_toolkit.ml.MachineLearningAccessor.standard_scaler,
        pandas_toolkit.ml.MachineLearningAccessor.train_validation_split,
    ],
    "df.nn.": [
        pandas_toolkit.nn.NeuralNetworkAccessor.init,
        pandas_toolkit.nn.NeuralNetworkAccessor.get_model,
        pandas_toolkit.nn.NeuralNetworkAccessor.hvplot_losses,
        pandas_toolkit.nn.NeuralNetworkAccessor.update,
        pandas_toolkit.nn.NeuralNetworkAccessor.predict,
        pandas_toolkit.nn.NeuralNetworkAccessor.evaluate,
    ],}
