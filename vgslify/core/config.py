# Imports

# > Standard library
from dataclasses import dataclass


@dataclass
class Conv2DConfig:
    """
    Configuration for 2D Convolutional layer.

    Parameters
    ----------
    activation : str
        Activation function to use.
    kernel_size : tuple
        Size of the convolution kernels.
    strides : tuple
        Stride length of the convolution.
    filters : int
        Number of output filters in the convolution.
    """
    activation: str
    kernel_size: tuple
    strides: tuple
    filters: int


@dataclass
class Pooling2DConfig:
    """
    Configuration for 2D Pooling layer.

    Parameters
    ----------
    pool_type : str
        Type of pooling operation (e.g., 'max', 'average').
    pool_size : tuple
        Size of the pooling window.
    strides : tuple
        Stride length of the pooling operation.
    """
    pool_type: str
    pool_size: tuple
    strides: tuple


@dataclass
class DenseConfig:
    """
    Configuration for Dense (Fully Connected) layer.

    Parameters
    ----------
    activation : str
        Activation function to use.
    units : int
        Number of neurons in the dense layer.
    """
    activation: str
    units: int


@dataclass
class RNNConfig:
    """
    Configuration for Recurrent Neural Network layer.

    Parameters
    ----------
    units : int
        Number of RNN units.
    return_sequences : bool
        Whether to return the last output or the full sequence.
    go_backwards : bool
        If True, process the input sequence backwards.
    dropout : float
        Fraction of the units to drop for the linear transformation of the inputs.
    recurrent_dropout : float
        Fraction of the units to drop for the linear transformation of the recurrent state.
    rnn_type : str, optional
        Type of RNN (e.g., 'simple', 'lstm', 'gru').
    bidirectional : bool, optional
        If True, create a bidirectional RNN.
    """
    units: int
    return_sequences: bool
    go_backwards: bool
    dropout: float
    recurrent_dropout: float
    rnn_type: str = None
    bidirectional: bool = False

@dataclass
class DropoutConfig:
    """
    Configuration for Dropout layer.

    Parameters
    ----------
    rate : float
        Fraction of the input units to drop.
    """
    rate: float


@dataclass
class ReshapeConfig:
    """
    Configuration for Reshape layer.

    Parameters
    ----------
    target_shape : tuple
        Target shape of the output.
    """
    target_shape: tuple


@dataclass
class InputConfig:
    """
    Configuration for Input layer.

    Parameters
    ----------
    batch_size : int
        Size of the batches of data.
    depth : int
        Depth of the input (for 3D inputs).
    height : int
        Height of the input.
    width : int
        Width of the input.
    channels : int
        Number of channels in the input.
    """
    batch_size: int
    depth: int
    height: int
    width: int
    channels: int

@dataclass
class ActivationConfig:
    """
    Configuration for Activation layer.

    Parameters
    ----------
    activation : str
        Activation function to use.
    """
    activation: str
