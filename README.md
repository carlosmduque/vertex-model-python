# Vertex Model

<!--- ![](fig_vertex_model.png) -->

A repository holding the structure for the 'vertexmodelpy' package as well examples and galleries.

## Introduction

The module 'vertexmodelpy' is a pure python implementation of the vertex model originally introduced by [Farhadifar et. al.](https://www.sciencedirect.com/science/article/pii/S0960982207023342) and is commonly used to simulate the mechanics of epithelia. At the moment, this implementation is restricted to tissue packings with open and free to deform boundaries. Periodic boundary condition (PBC) or the ability to simulate shear stress will be introduced in a future release.

## Installation

Via PyPi

```sh
pip install vertexmodelpy
```

Via Cloning

```sh
git clone https://github.com/carlosmduque/vertex-model-python.git .

pip install ./vertexmodelpy/
```

## Usage

For theory and algorithm details see publication: [Farhadifar et. al.](https://www.sciencedirect.com/science/article/pii/S0960982207023342)

```sh
import numpy as np
```

The package also allows you to directly

```text


```

## Requirements

```sh
numpy==1.21.0

scipy==1.7.3

pandas==1.5.3
```

## Acknowledgement

`vertexmodelpy` written by Carlos Duque