# @Author: Carlos Duque <carlos>
# @Date:   2023-07-10T15:50:36+02:00
# @Email:  carlosmduquej@gmail.com
# @Filename: __init__.py
# @Last modified by:   carlos
# @Last modified time: 2023-07-10T17:26:00+02:00


import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "vertexmodelpy",
    version = "0.0.1",
    author = "carlosmduquej",
    author_email = "carlosmduquej@gmail.com",
    description = "A Python repository of a free boundary implementing of the tissue mechanics vertex model",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/carlosmduquej/vertex-model-python",
    packages=setuptools.find_packages(),
    # packages=['vertexmodelpy'],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires = '>=3.6',
)

from setuptools import setup, find_packages
