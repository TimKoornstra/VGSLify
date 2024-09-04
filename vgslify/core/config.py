# Imports

# > Standard library
from dataclasses import dataclass


@dataclass
class Conv2DConfig:
    activation: str
    kernel_size: tuple
    strides: tuple
    filters: int


@dataclass
class Pooling2DConfig:
    pool_size: tuple
    strides: tuple


@dataclass
class DenseConfig:
    activation: str
    units: int


@dataclass
class RNNConfig:
    units: int
    return_sequences: bool
    go_backwards: bool
    dropout: float
    recurrent_dropout: float
    rnn_type: str = None


@dataclass
class DropoutConfig:
    rate: float


@dataclass
class OutputLayerConfig:
    dimensionality: int
    activation: str
    units: int


@dataclass
class ReshapeConfig:
    target_shape: tuple


@dataclass
class InputConfig:
    batch_size: int
    depth: int
    height: int
    width: int
    channels: int
