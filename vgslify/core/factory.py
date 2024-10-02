# Imports

# > Standard Libraries
from abc import ABC, abstractmethod
from typing import Any, Tuple
import math

# > Internal dependencies
from vgslify.core.parser import (parse_dropout_spec, parse_activation_spec,
                                 parse_reshape_spec, parse_conv2d_spec,
                                 parse_pooling2d_spec, parse_dense_spec,
                                 parse_rnn_spec, parse_input_spec)


class LayerFactory(ABC):
    """
    Abstract base class for creating neural network layers from VGSL
    specifications. This class defines the interface that must be implemented
    by concrete factories for different frameworks (e.g., TensorFlow, PyTorch).

    It also provides common methods for output shape calculations to be used by
    subclasses.
    """

    def __init__(self, input_shape: Tuple[int, ...] = None, data_format: str = 'channels_last'):
        self.layers = []
        self.data_format = data_format

        # Make sure the input shape is valid
        if input_shape is not None and not all(isinstance(dim, int) for dim in input_shape):
            raise ValueError("Input shape must be a tuple of integers.")

        # Set the input shape if provided
        self.shape = input_shape
        self._input_shape = input_shape

    def conv2d(self, spec: str):
        config = parse_conv2d_spec(spec)
        self._validate_input_shape()

        conv_layer = self._create_conv2d_layer(config)
        self._add_layer(conv_layer)

        # Add activation if needed
        # TODO: Only do this when we have PyTorch backend
        if config.activation:
            self.add_activation_layer(config.activation)

        # Update shape
        new_shape = self._compute_conv_output_shape(self.shape, config)
        self._update_shape(new_shape)

        return conv_layer

    def pooling2d(self, spec: str):
        config = parse_pooling2d_spec(spec)
        self._validate_input_shape()

        pool_layer = self._create_pooling2d_layer(config)
        self._add_layer(pool_layer)

        # Update shape
        new_shape = self._compute_pool_output_shape(self.shape, config)
        self._update_shape(new_shape)

        return pool_layer

    def dense(self, spec: str):
        config = parse_dense_spec(spec)
        self._validate_input_shape()

        dense_layer = self._create_dense_layer(config)
        self._add_layer(dense_layer)

        # Add activation if needed
        if config.activation:
            self.add_activation_layer(config.activation)

        # Update shape
        self._update_shape((config.units,))

        return dense_layer

    def rnn(self, spec: str):
        config = parse_rnn_spec(spec)
        self._validate_input_shape()

        rnn_layer = self._create_rnn_layer(config)
        self._add_layer(rnn_layer)

        # Update shape
        if config.return_sequences:
            self._update_shape((self.shape[0], config.units))
        else:
            self._update_shape((config.units,))

        return rnn_layer

    def bidirectional(self, spec: str):
        config = parse_rnn_spec(spec)
        self._validate_input_shape()

        bidirectional_layer = self._create_bidirectional_layer(config)
        self._add_layer(bidirectional_layer)

        # Update shape
        if config.return_sequences:
            self._update_shape((self.shape[0], config.units * 2))
        else:
            self._update_shape((config.units * 2,))

        return bidirectional_layer

    def batchnorm(self, spec: str):
        if spec != 'Bn':
            raise ValueError(
                f"BatchNormalization layer spec '{spec}' is incorrect. Expected 'Bn'.")

        self._validate_input_shape()

        batchnorm_layer = self._create_batchnorm_layer()
        self._add_layer(batchnorm_layer)

        # Shape remains the same
        return batchnorm_layer

    def input(self, spec: str):
        config = parse_input_spec(spec)

        # Adjust input shape based on the parsed dimensions
        if config.channels is not None and config.depth is not None:
            # 4D input: shape = (depth, height, width, channels)
            input_shape = (config.depth, config.height,
                           config.width, config.channels)
        elif config.channels is not None:
            # 3D input: shape = (height, width, channels)
            input_shape = (config.height, config.width, config.channels)
        elif config.height is not None:
            # 2D input: shape = (height, width)
            input_shape = (config.height, config.width)
        else:
            # 1D input: shape = (width,)
            input_shape = (config.width,)

        # Adjust for data format
        if self.data_format == 'channels_first':
            if len(input_shape) == 3:
                input_shape = (input_shape[2], input_shape[0], input_shape[1])
            elif len(input_shape) == 4:
                input_shape = (input_shape[3], input_shape[0],
                               input_shape[1], input_shape[2])

        self.shape = input_shape
        self._input_shape = input_shape

        input_layer = self._create_input_layer(config, input_shape)
        if input_layer is not None:
            # Some backends may not return the layer
            self._add_layer(input_layer)

        return input_layer

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
    def _create_conv2d_layer(self, config):
        pass

    @abstractmethod
    def _create_pooling2d_layer(self, config):
        pass

    @abstractmethod
    def _create_dense_layer(self, config):
        pass

    @abstractmethod
    def _create_rnn_layer(self, config):
        """Create an RNN layer (LSTM or GRU) based on the configuration."""
        pass

    @abstractmethod
    def _create_bidirectional_layer(self, config):
        """Create a bidirectional RNN layer based on the configuration."""
        pass

    @abstractmethod
    def _create_batchnorm_layer(self):
        pass

    @abstractmethod
    def _create_input_layer(self, config, input_shape):
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

    def _compute_conv_output_shape(self,
                                   input_shape: Tuple[int, ...],
                                   config: Any) -> Tuple[int, ...]:
        """
        Computes the output shape of a convolutional layer.

        Parameters
        ----------
        input_shape : tuple
            The input shape.
        config : Conv2DConfig
            The configuration object returned by parse_conv2d_spec.

        Returns
        -------
        tuple
            The output shape after the convolution.
        """
        if self.data_format == 'channels_last':
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

        if self.data_format == 'channels_last':
            return (H_out, W_out, C_out)
        else:
            return (C_out, H_out, W_out)

    def _compute_pool_output_shape(self,
                                   input_shape: Tuple[int, ...],
                                   config: Any) -> Tuple[int, ...]:
        """
        Computes the output shape of a pooling layer.

        Parameters
        ----------
        input_shape : tuple
            The input shape.
        config : Any
            The configuration object returned by parse_pooling2d_spec.

        Returns
        -------
        tuple
            The output shape after pooling.
        """
        if self.data_format == 'channels_last':
            H_in, W_in, C_in = input_shape
        else:
            C_in, H_in, W_in = input_shape

        # Compute output dimensions based on pooling size and strides
        H_out = (H_in + config.strides[0] - 1) // config.strides[0] \
            if H_in is not None else None
        W_out = (W_in + config.strides[1] - 1) // config.strides[1] if \
            W_in is not None else None

        if self.data_format == 'channels_last':
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

    def _validate_input_shape(self):
        if self.shape is None:
            raise ValueError("Input shape must be set before adding layers.")

    def _add_layer(self, layer: Any):
        self.layers.append(layer)

    def _update_shape(self, new_shape: Tuple[int, ...]):
        self.shape = new_shape
