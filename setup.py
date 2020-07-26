import os

from setuptools import find_packages, setup

from jax_toolkit import __version__

_here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(_here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="jax_toolkit",
    version=__version__,
    description="A collection of jax functions to help with common machine/deep learning related functionality.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="asmith26",
    url="https://github.com/asmith26/jax_toolkit.git",
    license="Apache-2.0",
    include_package_data=True,
    packages=find_packages(include=["jax_toolkit", "jax_toolkit.*"]),
    install_requires=["jax", "jaxlib"],
    extras_require={"losses_utils": ["dm-haiku"]},
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
    ],
)
