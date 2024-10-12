# Imports

# > Standard library
from typing import Tuple

# > Third-party dependencies
import torch
import torch.nn as nn

# > Internal dependencies
from vgslify.core.factory import LayerFactory
from vgslify.core.config import (Conv2DConfig, Pooling2DConfig, DenseConfig,
                                 RNNConfig, DropoutConfig, ReshapeConfig,
                                 InputConfig)
from vgslify.torch.reshape import Reshape

class TorchLayerFactory(LayerFactory):
    """
    TorchLayerFactory is responsible for creating PyTorch-specific layers based on parsed
    VGSL (Variable-size Graph Specification Language) specifications.

    This factory handles the creation of various types of layers, including convolutional layers,
    pooling layers, RNN layers, dense layers, activation layers, and more.

    Attributes
    ----------
    layers : list
        A list of PyTorch layers that have been added to the factory.
    shape : tuple of int
        The current shape of the tensor, excluding the batch size.
    _input_shape : tuple of int or None
        The original input shape provided during initialization.
    """

    def __init__(self, input_shape: Tuple[int, ...] = None):
        """
        Initialize the TorchLayerFactory.

        Parameters
        ----------
        input_shape : tuple of int, optional
            The input shape for the model, excluding batch size.
        """
        super().__init__(input_shape, data_format='channels_first')

    def build(self, name: str = "VGSL_Model") -> nn.Module:
        """
        Build the final model using the accumulated layers.

        Parameters
        ----------
        name : str, optional
            The name of the model, by default "VGSL_Model"

        Returns
        -------
        torch.nn.Module
            The constructed PyTorch model.

        Raises
        ------
        ValueError
            If no layers have been added to the model.
        ValueError
            If no input shape has been specified for the model.
        """
        if not self.layers:
            raise ValueError("No layers added to the model.")
        if not self._input_shape:
            raise ValueError("No input shape specified for the model.")

        # model = VGSLModel(self.layers)
        # TODO: Implement VGSLModel class
        model = nn.Sequential(*self.layers)
        model.__class__.__name__ = name
        return model

    # Layer creation methods
    def _input(self, config: InputConfig, input_shape: Tuple[int, ...]):
        """
        Create a PyTorch input layer (placeholder method).

        Parameters
        ----------
        config : InputConfig
            Configuration object (unused in PyTorch).
        input_shape : tuple of int
            The input shape for the layer.

        Returns
        -------
        None
            PyTorch doesn't require a separate input layer.
        """
        return None

    def _conv2d(self, config: Conv2DConfig):
        """
        Create a PyTorch Conv2d layer.

        Parameters
        ----------
        config : Conv2DConfig
            Configuration object for the Conv2D layer.

        Returns
        -------
        torch.nn.Conv2d
            The created Conv2d layer.
        """
        padding = 'same' if torch.__version__ >= '1.7' else self._compute_same_padding(
            config.kernel_size, config.strides)
        return nn.Conv2d(
            in_channels=self.shape[0],
            out_channels=config.filters,
            kernel_size=config.kernel_size,
            stride=config.strides,
            padding=padding
        )

    def _pooling2d(self, config: Pooling2DConfig):
        """
        Create a PyTorch Pooling2d layer.

        Parameters
        ----------
        config : Pooling2DConfig
            Configuration object for the Pooling2D layer.

        Returns
        -------
        torch.nn.Module
            The created Pooling2d layer (either MaxPool2d or AvgPool2d).
        """
        padding = self._compute_same_padding(config.pool_size, config.strides)
        pool_layer = nn.MaxPool2d if config.pool_type == 'max' else nn.AvgPool2d
        return pool_layer(
            kernel_size=config.pool_size,
            stride=config.strides,
            padding=padding
        )

    def _dense(self, config: DenseConfig):
        """
        Create a PyTorch Linear (Dense) layer.

        Parameters
        ----------
        config : DenseConfig
            Configuration object for the Dense layer.

        Returns
        -------
        torch.nn.Linear
            The created Linear layer.
        """
        return nn.Linear(self.shape[-1], config.units)

    def _rnn(self, config: RNNConfig):
        """
        Create a PyTorch RNN layer (LSTM or GRU), either unidirectional or bidirectional.

        Parameters
        ----------
        config : RNNConfig
            Configuration object for the RNN layer.

        Returns
        -------
        torch.nn.Module
            The created RNN layer (either LSTM or GRU, unidirectional or bidirectional).

        Raises
        ------
        ValueError
            If an unsupported RNN type is specified.
        """
        if config.rnn_type.upper() == 'L':
            rnn_class = nn.LSTM
        elif config.rnn_type.upper() == 'G':
            rnn_class = nn.GRU
        else:
            raise ValueError(f"Unsupported RNN type: {config.rnn_type}")

        return rnn_class(
            input_size=self.shape[-1],
            hidden_size=config.units,
            num_layers=1,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=config.bidirectional
        )

    def _batchnorm(self):
        """
        Create a PyTorch BatchNorm layer.

        Returns
        -------
        torch.nn.Module
            The created BatchNorm layer (either BatchNorm1d or BatchNorm2d).

        Raises
        ------
        ValueError
            If the input shape is not supported for BatchNorm.
        """
        if len(self.shape) == 3:
            return nn.BatchNorm2d(self.shape[0])
        elif len(self.shape) == 2:
            return nn.BatchNorm1d(self.shape[0])
        else:
            raise ValueError("Unsupported input shape for BatchNorm layer.")

    def _dropout(self, config: DropoutConfig):
        """
        Create a PyTorch Dropout layer.

        Parameters
        ----------
        config : DropoutConfig
            Configuration object for the Dropout layer.

        Returns
        -------
        nn.Dropout
            The created Dropout layer.
        """
        return nn.Dropout(p=config.rate)

    def _activation(self, activation_function: str):
        """
        Create a PyTorch activation layer.

        Parameters
        ----------
        activation_function : str
            Name of the activation function. Supported values are 'softmax', 'tanh', 'relu',
            'linear', 'sigmoid'.

        Returns
        -------
        nn.Module
            The created activation layer.

        Raises
        ------
        ValueError
            If the activation function is not supported.
        """
        activations = {
            'softmax': nn.Softmax(dim=1),
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'linear': nn.Identity(),
            'sigmoid': nn.Sigmoid(),
        }
        if activation_function in activations:
            return activations[activation_function]
        else:
            raise ValueError(f"Unsupported activation: {activation_function}")

    def _reshape(self, config: ReshapeConfig):
        """
        Create a PyTorch Reshape layer.

        Parameters
        ----------
        config : ReshapeConfig
            Configuration object for the Reshape layer.

        Returns
        -------
        nn.Module
            The created Reshape layer.
        """
        return Reshape(*config.target_shape)

    def _flatten(self):
        """
        Create a PyTorch Flatten layer.

        Returns
        -------
        nn.Flatten
            The created Flatten layer.
        """
        return nn.Flatten()

    # Helper methods
    def _compute_same_padding(self, kernel_size, stride):
        """
        Compute the padding size to achieve 'same' padding.

        Parameters
        ----------
        kernel_size : int or tuple
            Size of the kernel.
        stride : int or tuple
            Stride of the convolution.

        Returns
        -------
        tuple
            Padding size for height and width dimensions.
        """
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        padding = []
        for k, s in zip(kernel_size, stride):
            p = ((k - 1) // 2)
            padding.append(p)
        return tuple(padding)

    def _get_activation_layer(self, activation_name: str):
        """
        Return a PyTorch activation layer based on the activation name.

        Parameters
        ----------
        activation_name : str
            Name of the activation function.

        Returns
        -------
        torch.nn.Module
            The activation layer.

        Raises
        ------
        ValueError
            If the activation_name is not recognized.
        """
        activations = {
            'softmax': nn.Softmax(dim=1),
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'linear': nn.Identity(),
            'sigmoid': nn.Sigmoid(),
        }
        if activation_name in activations:
            return activations[activation_name]
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")
