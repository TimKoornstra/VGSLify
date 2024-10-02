# Imports

# > Standard library
from typing import Tuple

# > Third-party dependencies
import torch
import torch.nn as nn

# > Internal dependencies
from vgslify.core.factory import LayerFactory
from vgslify.core.parser import (parse_conv2d_spec, parse_pooling2d_spec,
                                 parse_dense_spec, parse_rnn_spec,
                                 parse_input_spec)


class TorchLayerFactory(LayerFactory):
    """
    TorchLayerFactory is responsible for creating PyTorch-specific layers based on parsed
    VGSL (Variable-size Graph Specification Language) specifications. This factory handles the
    creation of various types of layers, including convolutional layers, pooling layers, RNN layers,
    dense layers, activation layers, and more.

    This class maintains an internal state to track the shape of the tensor as layers are added.

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
        super().__init__(input_shape, data_format='channels_last')

    def build_final_model(self, name: str = "VGSL_Model") -> nn.Module:
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

        Examples
        --------
        >>> from vgslify.torch.layers import TorchLayerFactory
        >>> factory = TorchLayerFactory(input_shape=(3, 32, 32))
        >>> factory.conv2d("Cr3,3,64")
        >>> factory.maxpooling2d("Mp2,2,2,2")
        >>> model = factory.build_final_model()
        >>> print(model)
        VGSL_Model(
          (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=same)
          (1): ReLU()
          (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=same)
        )
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

    def _create_conv2d_layer(self, config):
        padding = 'same' if torch.__version__ >= '1.7' else self._compute_same_padding(
            config.kernel_size, config.strides)
        return nn.Conv2d(
            in_channels=self.shape[0],
            out_channels=config.filters,
            kernel_size=config.kernel_size,
            stride=config.strides,
            padding=padding
        )

    def _create_pooling2d_layer(self, config):
        padding = self._compute_same_padding(config.pool_size, config.strides)
        pool_layer = nn.MaxPool2d if config.pool_type == 'max' else nn.AvgPool2d
        return pool_layer(
            kernel_size=config.pool_size,
            stride=config.strides,
            padding=padding
        )

    def _create_dense_layer(self, config):
        return nn.Linear(self.shape[0], config.units)

    def _create_rnn_layer(self, config):
        if config.rnn_type == 'L':
            return nn.LSTM(
                input_size=self.shape[-1],
                hidden_size=config.units,
                num_layers=1,
                batch_first=True,
                dropout=config.dropout,
                bidirectional=False
            )
        elif config.rnn_type == 'G':
            return nn.GRU(
                input_size=self.shape[-1],
                hidden_size=config.units,
                num_layers=1,
                batch_first=True,
                dropout=config.dropout,
                bidirectional=False
            )
        else:
            raise ValueError(f"Unsupported RNN type: {config.rnn_type}")

    def _create_bidirectional_layer(self, config):
        rnn_layer = nn.LSTM if config.rnn_type == 'L' else nn.GRU

        return rnn_layer(
            input_size=self.shape[-1],
            hidden_size=config.units,
            num_layers=1,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=True
        )

    def _create_input_layer(self, config, input_shape: Tuple[int, ...]):
        # PyTorch doesn't require a separate input layer.
        # This is only for reference and compatibility.
        return None

    def _create_batchnorm_layer(self):
        if len(self.shape) == 3:
            return nn.BatchNorm2d(self.shape[0])
        elif len(self.shape) == 2:
            return nn.BatchNorm1d(self.shape[0])
        else:
            raise ValueError("Unsupported input shape for BatchNorm layer.")

    def _create_dropout_layer(self, rate: float):
        """
        Create a PyTorch Dropout layer.

        Parameters
        ----------
        rate : float
            Dropout rate, between 0 and 1.

        Returns
        -------
        nn.Dropout
            The created Dropout layer.
        """
        return nn.Dropout(p=rate)

    def _create_activation_layer(self, activation_function: str):
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

    def _create_reshape_layer(self, target_shape: Tuple[int, ...]):
        """
        Create a PyTorch Reshape layer.

        Parameters
        ----------
        target_shape : tuple
            The target shape to reshape to, excluding the batch size.

        Returns
        -------
        nn.Module
            The created Reshape layer.
        """
        return self.Reshape(*target_shape)

    class Reshape(nn.Module):
        def __init__(self, *args):
            """
            PyTorch custom Reshape layer.

            Parameters
            ----------
            *args : int
                Dimensions of the target shape excluding the batch size.
            """
            super().__init__()
            self.target_shape = args

        def forward(self, x):
            """
            Forward pass for reshaping the input tensor.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor to reshape.

            Returns
            -------
            torch.Tensor
                Reshaped tensor.
            """
            return x.view(x.size(0), *self.target_shape)

    def _create_flatten_layer(self):
        """
        Create a PyTorch Flatten layer.

        Returns
        -------
        nn.Flatten
            The created Flatten layer.
        """
        return nn.Flatten()
