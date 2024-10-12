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
from vgslify.core.config import (Conv2DConfig, Pooling2DConfig, DenseConfig,
                                 RNNConfig, DropoutConfig, ReshapeConfig,
                                 InputConfig)


class LayerFactory(ABC):
    """
    Abstract base class for creating neural network layers from VGSL specifications.

    This class defines the interface that must be implemented by concrete factories
    for different frameworks (e.g., TensorFlow, PyTorch). It also provides common
    methods for output shape calculations to be used by subclasses.

    Parameters
    ----------
    input_shape : tuple of int, optional
        The initial input shape for the model.
    data_format : str, default 'channels_last'
        The data format for the input tensor. Either 'channels_last' or 'channels_first'.

    Attributes
    ----------
    layers : list
        A list to store the created layers.
    data_format : str
        The data format for the input tensor.
    shape : tuple of int
        The current shape of the output tensor.
    _input_shape : tuple of int
        The initial input shape for the model.

    Notes
    -----
    This is an abstract base class. Use a concrete implementation like 
    `TensorFlowLayerFactory` or `PyTorchLayerFactory` in your code.

    This class uses a naming convention where public methods for creating layers
    (e.g., conv2d) have corresponding private methods with an underscore prefix
    (e.g., _conv2d) that handle the actual layer creation.

    Examples
    --------
    >>> # Assuming we have a TensorFlowLayerFactory implementation
    >>> factory = TensorFlowLayerFactory(input_shape=(224, 224, 3))
    >>> factory.conv2d('Cr3,3,32')
    >>> factory.pooling2d('Mp2,2,2,2')
    >>> factory.flatten('Flt')
    >>> factory.dense('Fs128')
    >>> model = factory.build('my_model')
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

    @abstractmethod
    def build(self, name: str):
        """
        Abstract method to build the final model using the created layers.

        Parameters
        ----------
        name : str
            The name of the model.

        Returns
        -------
        Any
            The final built model.

        Examples
        --------
        >>> # Using a hypothetical concrete implementation
        >>> factory = SomeConcreteLayerFactory(input_shape=(28, 28, 1))
        >>> factory.conv2d('Cr3,3,32')
        >>> factory.flatten('Flt')
        >>> factory.dense('Fs10')
        >>> model = factory.build('my_model')
        """
        pass

    # Layer creation methods
    def input(self, spec: str):
        """
        Create an Input layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Input layer.

        Returns
        -------
        Any
            The created Input layer.

        Examples
        --------
        >>> # Using a hypothetical concrete implementation
        >>> factory = SomeConcreteLayerFactory()
        >>> factory.input('1,28,28,1')
        """
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

        input_layer = self._input(config, input_shape)
        if input_layer is not None:
            # Some backends may not return the layer
            self._add_layer(input_layer)

        return input_layer

    def conv2d(self, spec: str):
        """
        Create a 2D Convolutional layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Conv2D layer.

        Returns
        -------
        Any
            The created Conv2D layer.

        Examples
        --------
        >>> # Using a hypothetical concrete implementation
        >>> factory = SomeConcreteLayerFactory(input_shape=(28, 28, 1))
        >>> factory.conv2d('Cr3,3,32')
        """
        config = parse_conv2d_spec(spec)
        self._validate_input_shape()

        conv_layer = self._conv2d(config)
        self._add_layer(conv_layer)

        # Add activation if needed
        if config.activation:
            activation_layer = self._activation(config.activation)
            self._add_layer(activation_layer)

        # Update shape
        new_shape = self._compute_conv_output_shape(self.shape, config)
        self._update_shape(new_shape)

        return conv_layer

    def pooling2d(self, spec: str):
        """
        Create a 2D Pooling layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Pooling2D layer.

        Returns
        -------
        Any
            The created Pooling2D layer.

        Examples
        --------
        >>> # Using a hypothetical concrete implementation
        >>> factory = SomeConcreteLayerFactory(input_shape=(28, 28, 32))
        >>> factory.pooling2d('Mp2,2,2,2')
        """
        config = parse_pooling2d_spec(spec)
        self._validate_input_shape()

        pool_layer = self._pooling2d(config)
        self._add_layer(pool_layer)

        # Update shape
        new_shape = self._compute_pool_output_shape(self.shape, config)
        self._update_shape(new_shape)

        return pool_layer

    def dense(self, spec: str):
        """
        Create a Dense (Fully Connected) layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Dense layer.

        Returns
        -------
        Any
            The created Dense layer.

        Examples
        --------
        >>> # Using a hypothetical concrete implementation
        >>> factory = SomeConcreteLayerFactory(input_shape=(7*7*32,))
        >>> factory.dense('Fs128')
        """
        config = parse_dense_spec(spec)
        self._validate_input_shape()

        dense_layer = self._dense(config)
        self._add_layer(dense_layer)

        # Add activation if needed
        if config.activation:
            activation_layer = self._activation(config.activation)
            self._add_layer(activation_layer)

        # Update shape
        self._update_shape((config.units,))

        return dense_layer

    def rnn(self, spec: str):
        """
        Create an RNN layer (LSTM or GRU), either unidirectional or bidirectional, based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the RNN layer.

        Returns
        -------
        Any
            The created RNN layer (either unidirectional or bidirectional).

        Examples
        --------
        >>> # Using a hypothetical concrete implementation
        >>> factory = SomeConcreteLayerFactory(input_shape=(28, 28))
        >>> factory.rnn('Ls128')  # Unidirectional LSTM
        >>> factory.rnn('Bl128')  # Bidirectional LSTM
        """
        config = parse_rnn_spec(spec)
        self._validate_input_shape()

        rnn_layer = self._rnn(config)
        self._add_layer(rnn_layer)

        # Update shape
        if config.return_sequences:
            if config.bidirectional:
                self._update_shape((self.shape[0], config.units * 2))
            else:
                self._update_shape((self.shape[0], config.units))
        else:
            if config.bidirectional:
                self._update_shape((config.units * 2,))
            else:
                self._update_shape((config.units,))

        return rnn_layer

    def batchnorm(self, spec: str):
        """
        Create a BatchNormalization layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the BatchNormalization layer.

        Returns
        -------
        Any
            The created BatchNormalization layer.

        Examples
        --------
        >>> # Using a hypothetical concrete implementation
        >>> factory = SomeConcreteLayerFactory(input_shape=(28, 28, 32))
        >>> factory.batchnorm('Bn')
        """
        if spec != 'Bn':
            raise ValueError(
                f"BatchNormalization layer spec '{spec}' is incorrect. Expected 'Bn'.")

        self._validate_input_shape()

        batchnorm_layer = self._batchnorm()
        self._add_layer(batchnorm_layer)

        # Shape remains the same
        return batchnorm_layer

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

        Examples
        --------
        >>> # Using a hypothetical concrete implementation
        >>> factory = SomeConcreteLayerFactory(input_shape=(128,))
        >>> factory.dropout('D50')
        """
        config = parse_dropout_spec(spec)
        layer = self._dropout(config)
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

        Examples
        --------
        >>> # Using a hypothetical concrete implementation
        >>> factory = SomeConcreteLayerFactory(input_shape=(128,))
        >>> factory.activation('Ar')
        """
        activation_function = parse_activation_spec(spec)
        layer = self._activation(activation_function)
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

        Examples
        --------
        >>> # Using a hypothetical concrete implementation
        >>> factory = SomeConcreteLayerFactory(input_shape=(28, 28, 1))
        >>> factory.reshape('Rc3')
        """
        if self.shape is None:
            raise ValueError("Input shape must be set before adding layers.")

        # Handle 'Rc' (collapse spatial dimensions) specification
        if spec.startswith('Rc'):
            if spec == 'Rc2':
                # Flatten to (batch_size, -1)
                layer = self._flatten()
                self.layers.append(layer)
                self.shape = (int(self._compute_flatten_shape(self.shape)),)
                return layer

            elif spec == 'Rc3':
                # Reshape to (batch_size, seq_length, features)
                if len(self.shape) != 3:
                    raise ValueError(
                        f"Expected a 3D input shape for 'Rc3', got {self.shape}")

                if self.data_format == 'channels_first':
                    C, H, W = self.shape
                else:  # channels_last
                    H, W, C = self.shape

                # Handle variable height
                if H is None:
                    seq_length = None
                else:
                    seq_length = H * W if W is not None else None

                features = C
                config = ReshapeConfig(target_shape=(seq_length, features))
                layer = self._reshape(config)
                self.layers.append(layer)
                self.shape = (seq_length, features)
                return layer

            else:
                raise ValueError(f"Unsupported Rc variant: {spec}")

        # Handle regular reshape (e.g., 'R64,64,3')
        config = parse_reshape_spec(spec)
        layer = self._reshape(config)
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

        Examples
        --------
        >>> # Using a hypothetical concrete implementation
        >>> factory = SomeConcreteLayerFactory(input_shape=(7, 7, 64))
        >>> factory.flatten('Flt')
        """
        if spec != "Flt":
            raise ValueError(
                f"Flatten layer spec '{spec}' is incorrect. Expected 'Flt'.")

        layer = self._flatten()
        self.layers.append(layer)
        # Update shape
        self.shape = (self._compute_flatten_shape(self.shape),)
        return layer

    # Abstract methods
    @abstractmethod
    def _input(self, config: InputConfig, input_shape: Tuple[int, ...]):
        """
        Abstract method to create an Input layer.

        Parameters
        ----------
        config : InputConfig
            The configuration object returned by parse_input_spec.
        input_shape : tuple of int
            The input shape.

        Returns
        -------
        Any
            The created Input layer.
        """
        pass

    @abstractmethod
    def _conv2d(self, config: Conv2DConfig):
        """
        Abstract method to create a Conv2D layer.

        Parameters
        ----------
        config : Conv2DConfig
            The configuration object returned by parse_conv2d_spec.

        Returns
        -------
        Any
            The created Conv2D layer.
        """
        pass

    @abstractmethod
    def _pooling2d(self, config: Pooling2DConfig):
        """
        Abstract method to create a Pooling2D layer.

        Parameters
        ----------
        config : Pooling2DConfig
            The configuration object returned by parse_pooling2d_spec.

        Returns
        -------
        Any
            The created Pooling2D layer.
        """
        pass

    @abstractmethod
    def _dense(self, config: DenseConfig):
        """
        Abstract method to create a Dense (Fully Connected) layer.

        Parameters
        ----------
        config : DenseConfig
            The configuration object returned by parse_dense_spec.

        Returns
        -------
        Any
            The created Dense layer.
        """
        pass

    @abstractmethod
    def _rnn(self, config: RNNConfig):
        """
        Abstract method to create an RNN layer (LSTM or GRU).

        Parameters
        ----------
        config : RNNConfig
            The configuration object returned by parse_rnn_spec.

        Returns
        -------
        Any
            The created RNN layer.
        """
        pass

    @abstractmethod
    def _batchnorm(self):
        """
        Abstract method to create a BatchNormalization layer.

        Returns
        -------
        Any
            The created BatchNormalization layer.
        """
        pass

    @abstractmethod
    def _dropout(self, config: DropoutConfig):
        """
        Abstract method to create a Dropout layer.

        Parameters
        ----------
        config : DropoutConfig
            The configuration object returned by parse_dropout_spec.

        Returns
        -------
        Any
            The created Dropout layer.
        """
        pass

    @abstractmethod
    def _activation(self, activation_function: str):
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
    def _reshape(self, config: ReshapeConfig):
        """
        Abstract method to create a Reshape layer.

        Parameters
        ----------
        config : ReshapeConfig
            The configuration object returned by parse_reshape_spec.

        Returns
        -------
        Any
            The created Reshape layer.
        """
        pass

    @abstractmethod
    def _flatten(self):
        """
        Abstract method to create a Flatten layer.

        Returns
        -------
        Any
            The created Flatten layer.
        """
        pass

    # Helper methods
    def _compute_conv_output_shape(self,
                                   input_shape: Tuple[int, ...],
                                   config: Conv2DConfig) -> Tuple[int, ...]:
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
                                   config: Pooling2DConfig) -> Tuple[int, ...]:
        """
        Computes the output shape of a pooling layer.

        Parameters
        ----------
        input_shape : tuple
            The input shape.
        config : Pooling2DConfig
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
        """
        Validates that the input shape has been set before adding layers.

        Raises
        ------
        ValueError
            If the input shape has not been set.
        """
        if self.shape is None:
            raise ValueError("Input shape must be set before adding layers.")

    def _add_layer(self, layer: Any):
        """
        Adds a layer to the list of layers.

        Parameters
        ----------
        layer : Any
            The layer to be added.
        """
        self.layers.append(layer)

    def _update_shape(self, new_shape: Tuple[int, ...]):
        """
        Updates the current shape of the output tensor.

        Parameters
        ----------
        new_shape : tuple of int
            The new shape to set.
        """
        self.shape = new_shape