# Imports

# > Standard Libraries
from abc import ABC, abstractmethod
from typing import Any, Tuple
import math

# > Internal dependencies
from vgslify.core.parser import (parse_dropout_spec, parse_activation_spec,
                                 parse_reshape_spec)


class LayerFactory(ABC):
    """
    Abstract base class for creating neural network layers from VGSL
    specifications. This class defines the interface that must be implemented
    by concrete factories for different frameworks (e.g., TensorFlow, PyTorch).

    It also provides common methods for output shape calculations to be used by
    subclasses.
    """

    def __init__(self, input_shape: Tuple[int, ...] = None):
        self.layers = []

        # Make sure the input shape is valid
        if input_shape is not None and not all(isinstance(dim, int) for dim in input_shape):
            raise ValueError("Input shape must be a tuple of integers.")

        # Set the input shape if provided
        self.shape = input_shape
        self._input_shape = input_shape

    def dropout(self, spec: str):
        """
        Create a Dropout layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Dropout layer.

        Returns
        -------
        Any
            The created Dropout layer.
        """
        config = parse_dropout_spec(spec)
        layer = self._create_dropout_layer(config.rate)
        self.layers.append(layer)
        # Shape remains the same
        return layer

    def activation(self, spec: str):
        """
        Create an Activation layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Activation layer.

        Returns
        -------
        Any
            The created Activation layer.
        """
        activation_function = parse_activation_spec(spec)
        layer = self._create_activation_layer(activation_function)
        self.layers.append(layer)
        # Shape remains the same
        return layer

    def reshape(self, spec: str):
        """
        Create a Reshape layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            VGSL specification string for the Reshape layer. Can be:
            - 'Rc(2|3)': Collapse spatial dimensions (height, width, and channels).
            - 'R<x>,<y>,<z>': Reshape to the specified target shape.

        Returns
        -------
        Any
            The created Reshape layer.
        """
        if self.shape is None:
            raise ValueError("Input shape must be set before adding layers.")

        # Handle 'Rc' (collapse spatial dimensions) specification
        if spec.startswith('Rc'):
            if spec == 'Rc2':
                # Flatten to (batch_size, -1)
                layer = self._create_flatten_layer()
                self.layers.append(layer)
                self.shape = (int(self._compute_flatten_shape(self.shape)),)
                return layer

            elif spec == 'Rc3':
                # Reshape to (batch_size, seq_length, features)
                if len(self.shape) != 3:
                    raise ValueError(
                        f"Expected a 3D input shape for 'Rc3', got {self.shape}")

                C, H, W = self.shape
                seq_length = H * W
                features = C
                layer = self._create_reshape_layer((seq_length, features))
                self.layers.append(layer)
                self.shape = (seq_length, features)
                return layer

            else:
                raise ValueError(f"Unsupported Rc variant: {spec}")

        # Handle regular reshape (e.g., 'R64,64,3')
        config = parse_reshape_spec(spec)
        layer = self._create_reshape_layer(config.target_shape)
        self.layers.append(layer)
        self.shape = config.target_shape
        return layer

    def flatten(self, spec: str):
        """
        Create a Flatten layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Flatten layer.

        Returns
        -------
        Any
            The created Flatten layer.
        """
        if spec != "Flt":
            raise ValueError(
                f"Flatten layer spec '{spec}' is incorrect. Expected 'Flt'.")

        layer = self._create_flatten_layer()
        self.layers.append(layer)
        # Update shape
        self.shape = (self._compute_flatten_shape(self.shape),)
        return layer

    @abstractmethod
    def conv2d(self, spec: str):
        pass

    @abstractmethod
    def maxpooling2d(self, spec: str):
        pass

    @abstractmethod
    def avgpool2d(self, spec: str):
        pass

    @abstractmethod
    def dense(self, spec: str):
        pass

    @abstractmethod
    def lstm(self, spec: str):
        pass

    @abstractmethod
    def gru(self, spec: str):
        pass

    @abstractmethod
    def bidirectional(self, spec: str):
        pass

    @abstractmethod
    def batchnorm(self, spec: str):
        pass

    @abstractmethod
    def input(self, spec: str):
        pass

    @abstractmethod
    def build_final_model(self, name):
        pass

    @abstractmethod
    def _create_dropout_layer(self, rate: float):
        """
        Abstract method to create a Dropout layer.

        Parameters
        ----------
        rate : float
            Dropout rate.

        Returns
        -------
        Any
            The created Dropout layer.
        """
        pass

    @abstractmethod
    def _create_activation_layer(self, activation_function: str):
        """
        Abstract method to create an Activation layer.

        Parameters
        ----------
        activation_function : str
            Name of the activation function.

        Returns
        -------
        Any
            The created Activation layer.
        """
        pass

    @abstractmethod
    def _create_reshape_layer(self, target_shape: Tuple[int, ...]):
        """
        Abstract method to create a Reshape layer.

        Parameters
        ----------
        target_shape : tuple
            The target shape to reshape to.

        Returns
        -------
        Any
            The created Reshape layer.
        """
        pass

    @abstractmethod
    def _create_flatten_layer(self):
        """
        Abstract method to create a Flatten layer.

        Returns
        -------
        Any
            The created Flatten layer.
        """
        pass

    def _compute_conv_output_shape(self, input_shape, config, data_format='channels_last'):
        """
        Computes the output shape of a convolutional layer.

        Parameters
        ----------
        input_shape : tuple
            The input shape.
        config : Conv2DConfig
            The configuration object returned by parse_conv2d_spec.
        data_format : str
            One of 'channels_first' or 'channels_last'.

        Returns
        -------
        tuple
            The output shape after the convolution.
        """
        if data_format == 'channels_last':
            H_in, W_in, C_in = input_shape
        else:
            C_in, H_in, W_in = input_shape

        # Compute output dimensions based on padding and strides
        # Adjust calculations based on the backend's handling of padding

        # Example computation for 'same' padding
        H_out = math.ceil(H_in / config.strides[0]) \
            if H_in is not None else None
        W_out = math.ceil(W_in / config.strides[1]) \
            if W_in is not None else None
        C_out = config.filters

        if data_format == 'channels_last':
            return (H_out, W_out, C_out)
        else:
            return (C_out, H_out, W_out)

    def _compute_pool_output_shape(self,
                                   input_shape: Tuple[int, ...],
                                   config: Any,
                                   data_format: str = 'channels_last') -> Tuple[int, ...]:
        """
        Computes the output shape of a pooling layer.

        Parameters
        ----------
        input_shape : tuple
            The input shape.
        config : Any
            The configuration object returned by parse_pooling2d_spec.
        data_format : str, optional
            One of 'channels_first' or 'channels_last'. Defaults to 'channels_last'.

        Returns
        -------
        tuple
            The output shape after pooling.
        """
        if data_format == 'channels_last':
            H_in, W_in, C_in = input_shape
        else:
            C_in, H_in, W_in = input_shape

        # Compute output dimensions based on pooling size and strides
        H_out = (H_in + config.strides[0] - 1) // config.strides[0] \
            if H_in is not None else None
        W_out = (W_in + config.strides[1] - 1) // config.strides[1] if \
            W_in is not None else None

        if data_format == 'channels_last':
            return (H_out, W_out, C_in)
        else:
            return (C_in, H_out, W_out)

    def _compute_flatten_shape(self, shape: Tuple[int, ...]) -> int:
        """
        Computes the shape after flattening the input.

        Parameters
        ----------
        shape : tuple
            The input shape.

        Returns
        -------
        int
            The product of all dimensions.
        """
        from functools import reduce
        from operator import mul
        return reduce(mul, shape)
