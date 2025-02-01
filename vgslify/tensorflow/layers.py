# Imports

# > Standard library
import inspect
from typing import Callable, Tuple

# > Third-party dependencies
import tensorflow as tf

from vgslify.core.config import (
    Conv2DConfig,
    DenseConfig,
    DropoutConfig,
    InputConfig,
    Pooling2DConfig,
    ReshapeConfig,
    RNNConfig,
)

# > Internal dependencies
from vgslify.core.factory import LayerFactory


class TensorFlowLayerFactory(LayerFactory):
    """
    TensorFlowLayerFactory is responsible for creating TensorFlow-specific layers based on parsed
    VGSL (Variable-size Graph Specification Language) specifications.

    This factory handles the creation of various types of layers, including convolutional layers,
    pooling layers, RNN layers, dense layers, activation layers, and more.

    Attributes
    ----------
    layers : list
        A list of TensorFlow layers that have been added to the factory.
    shape : tuple of int
        The current shape of the tensor, excluding the batch size.
    _input_shape : tuple of int or None
        The original input shape provided during initialization.
    """

    # A class-level dictionary that holds {prefix -> callable} for custom layers
    _custom_layer_registry = {}

    @classmethod
    def register(cls, prefix: str, builder_fn):
        """
        Register a custom layer builder function under a given spec prefix.

        Parameters
        ----------
        prefix : str
            The VGSL spec prefix that triggers this custom layer (e.g. "Xsw").
        builder_fn : callable
            A function with signature `builder_fn(self, spec: str) -> layer`
            that, given the VGSL spec string, returns the framework-specific layer.
        """
        if prefix in cls._custom_layer_registry:
            raise ValueError(f"Prefix '{prefix}' is already registered.")

        # Inspect the builder function’s signature
        sig = inspect.signature(builder_fn)
        params = list(sig.parameters.values())

        # Check that we have exactly two parameters
        if len(params) != 2:
            raise ValueError(
                "Custom layer builder_fn must define exactly two parameters: "
                "(factory_self, spec)."
            )

        cls._custom_layer_registry[prefix] = builder_fn

    @classmethod
    def get_custom_layer_registry(cls):
        """Return the dict of all registered custom layers for this factory class."""
        return cls._custom_layer_registry

    def __init__(self, input_shape: Tuple[int, ...] = None):
        """
        Initialize the TensorFlowLayerFactory.

        Parameters
        ----------
        input_shape : tuple of int, optional
            The input shape for the model, excluding batch size.
        """
        super().__init__(input_shape, data_format="channels_last")

    def build(self, name: str = "VGSL_Model") -> tf.keras.models.Model:
        """
        Build the final model using the accumulated layers.

        Parameters
        ----------
        name : str, optional
            The name of the model, by default "VGSL_Model"

        Returns
        -------
        tf.keras.models.Model
            The constructed TensorFlow model.

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

        # If we do not have an input layer, add one
        if not tf.keras.backend.is_keras_tensor(self.layers[0]):
            input_layer = tf.keras.Input(shape=self._input_shape)
            self.layers.insert(0, input_layer)

        inputs = self.layers[0]
        outputs = inputs
        for layer in self.layers[1:]:
            outputs = layer(outputs)
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name)
        return model

    # Layer creation methods
    def _input(self, config: InputConfig, input_shape: Tuple[int, ...]):
        """
        Create a TensorFlow Input layer.

        Parameters
        ----------
        config : InputConfig
            Configuration object for the Input layer.
        input_shape : tuple of int
            The input shape for the layer.

        Returns
        -------
        tf.keras.layers.Input
            The created Input layer.
        """
        return tf.keras.Input(shape=input_shape, batch_size=config.batch_size)

    def _conv2d(self, config: Conv2DConfig):
        """
        Create a TensorFlow Conv2D layer.

        Parameters
        ----------
        config : Conv2DConfig
            Configuration object for the Conv2D layer.

        Returns
        -------
        tf.keras.layers.Conv2D
            The created Conv2D layer.
        """
        return tf.keras.layers.Conv2D(
            filters=config.filters,
            kernel_size=config.kernel_size,
            strides=config.strides,
            padding="same",
            activation=None,
        )

    def _pooling2d(self, config: Pooling2DConfig):
        """
        Create a TensorFlow Pooling2D layer.

        Parameters
        ----------
        config : Pooling2DConfig
            Configuration object for the Pooling2D layer.

        Returns
        -------
        tf.keras.layers.Layer
            The created Pooling2D layer (either MaxPooling2D or AveragePooling2D).
        """
        if config.pool_type == "max":
            return tf.keras.layers.MaxPooling2D(
                pool_size=config.pool_size, strides=config.strides, padding="same"
            )
        if config.pool_type == "avg":
            return tf.keras.layers.AveragePooling2D(
                pool_size=config.pool_size, strides=config.strides, padding="same"
            )

    def _dense(self, config: DenseConfig):
        """
        Create a TensorFlow Dense layer.

        Parameters
        ----------
        config : DenseConfig
            Configuration object for the Dense layer.

        Returns
        -------
        tf.keras.layers.Dense
            The created Dense layer.
        """
        return tf.keras.layers.Dense(units=config.units, activation=None)

    def _rnn(self, config: RNNConfig):
        """
        Create a TensorFlow RNN layer (LSTM or GRU), either unidirectional or bidirectional.

        Parameters
        ----------
        config : RNNConfig
            Configuration object for the RNN layer.

        Returns
        -------
        tf.keras.layers.Layer
            The created RNN layer (either LSTM or GRU, unidirectional or bidirectional).

        Raises
        ------
        ValueError
            If an unsupported RNN type is specified.
        """
        if config.rnn_type.upper() == "L":
            rnn_class = tf.keras.layers.LSTM
        elif config.rnn_type.upper() == "G":
            rnn_class = tf.keras.layers.GRU
        else:
            raise ValueError(f"Unsupported RNN type: {config.rnn_type}")

        rnn_layer = rnn_class(
            units=config.units,
            return_sequences=config.return_sequences,
            dropout=config.dropout,
            recurrent_dropout=config.recurrent_dropout,
        )

        if config.bidirectional:
            return tf.keras.layers.Bidirectional(rnn_layer, merge_mode="concat")
        else:
            return rnn_layer

    def _batchnorm(self):
        """
        Create a TensorFlow BatchNormalization layer.

        Returns
        -------
        tf.keras.layers.BatchNormalization
            The created BatchNormalization layer.
        """
        return tf.keras.layers.BatchNormalization()

    def _dropout(self, config: DropoutConfig):
        """
        Create a TensorFlow Dropout layer.

        Parameters
        ----------
        config : DropoutConfig
            Configuration object for the Dropout layer.

        Returns
        -------
        tf.keras.layers.Dropout
            The created Dropout layer.
        """
        return tf.keras.layers.Dropout(rate=config.rate)

    def _activation(self, activation_function: str):
        """
        Create a TensorFlow activation layer.

        Parameters
        ----------
        activation_function : str
            Name of the activation function.

        Returns
        -------
        tf.keras.layers.Activation
            The created activation layer.
        """
        return tf.keras.layers.Activation(activation=activation_function)

    def _reshape(self, config: ReshapeConfig):
        """
        Create a TensorFlow Reshape layer.

        Parameters
        ----------
        config : ReshapeConfig
            Configuration object for the Reshape layer.

        Returns
        -------
        tf.keras.layers.Reshape
            The created Reshape layer.
        """
        return tf.keras.layers.Reshape(target_shape=config.target_shape)

    def _flatten(self):
        """
        Create a TensorFlow Flatten layer.

        Returns
        -------
        tf.keras.layers.Flatten
            The created Flatten layer.
        """
        return tf.keras.layers.Flatten()


def register_custom_layer(prefix: str) -> Callable:
    """
    Decorator to register a custom layer builder function for TensorFlowLayerFactory.

    This allows users to easily extend TensorFlowLayerFactory with custom layer types by
    defining a function that constructs a TensorFlow layer from a VGSL spec string.

    Parameters
    ----------
    prefix : str
        The VGSL spec prefix that triggers this custom layer (e.g. "Xsw").

    Returns
    -------
    Callable
        A decorator that registers the provided function as a builder for the given prefix.

    Raises
    ------
    ValueError
        If a builder for the prefix is already registered or if the function signature is invalid.

    Examples
    --------
    >>> from vgslify.tensorflow.layers import register_custom_layer
    >>> @register_custom_layer("Xsw")
    ... def build_custom_layer(factory, spec):
    ...     # Custom layer building logic
    ...     return tf.keras.layers.Dense(10)
    """

    def decorator(fn: Callable) -> Callable:
        TensorFlowLayerFactory.register(prefix, fn)
        return fn

    return decorator
