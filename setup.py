import os

from setuptools import find_packages, setup

from pandas_toolkit import __version__

_here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(_here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pandas_toolkit",
    version=__version__,
    description="A collection of pandas accessors to help with common machine learning related functionality.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="asmith26",
    url="https://github.com/asmith26/pandas_toolkit.git",
    license="Apache-2.0",
    include_package_data=True,
    packages=find_packages(include=["pandas_toolkit", "pandas_toolkit.*"]),
    install_requires=["dm-haiku", "jax", "jaxlib", "pandas", "scikit-learn"],
    extras_require={"streamz": ["hvplot", "streamz"]},
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
    ],
)
