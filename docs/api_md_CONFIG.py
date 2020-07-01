import jax_toolkit.ml

ROOT_GITHUB_URL = "https://github.com/asmith26/jax_toolkit/blob/master/jax_toolkit/"
PACKAGE_NAME = "jax_toolkit"

accessors = {
    "df.ml.": [
        jax_toolkit.ml.MachineLearningAccessor.standard_scaler,
        jax_toolkit.ml.MachineLearningAccessor.train_validation_split,
    ],
}
