# Imports

# > Standard Library
import inspect
from typing import Callable, Dict, Type, Union

# > Third-Party Dependencies
import tensorflow as tf

# > Internal
from vgslify.core.config import (
    ActivationConfig,
    Conv2DConfig,
    DenseConfig,
    DropoutConfig,
    InputConfig,
    Pooling2DConfig,
    ReshapeConfig,
    RNNConfig,
)
from vgslify.model_parsers.base import BaseModelParser


class TensorFlowModelParser(BaseModelParser):
    """
    Parser for converting TensorFlow Keras models into VGSL (Variable-size Graph Specification Language) spec strings.

    This class extends the BaseModelParser to provide specific functionality for TensorFlow Keras models.
    It uses configuration dataclasses to represent different layer types and converts them into
    VGSL spec strings.

    Attributes
    ----------
    layer_parsers : Dict[Type[tf.keras.layers.Layer], Callable]
        A dictionary mapping TensorFlow Keras layer types to their corresponding parsing methods.

    Notes
    -----
    This parser supports a wide range of TensorFlow Keras layers and can be extended to support
    additional layer types by adding new parsing methods and updating the layer_parsers dictionary.
    """

    # A class-level dictionary: {layer_class -> parser_function}
    _custom_layer_parsers: Dict[Type[tf.keras.layers.Layer], Callable] = {}

    @classmethod
    def register(cls, layer_cls: Type[tf.keras.layers.Layer], parser_fn: Callable):
        """
        Registers a custom parser function for a given TF layer class.

        Parameters
        ----------
        layer_cls : Type[tf.keras.layers.Layer]
            The TF layer class this parser function can handle.
        parser_fn : Callable
            A function with signature parser_fn(layer) -> str
            that returns a VGSL spec string for the given layer.
        """
        if layer_cls in cls._custom_layer_parsers:
            raise ValueError(
                f"A parser is already registered for {layer_cls.__name__}."
            )

        # Check signature to ensure `parser_fn` is (layer) -> str
        sig = inspect.signature(parser_fn)
        params = list(sig.parameters.values())
        if len(params) != 1:
            raise ValueError(
                "Custom parser function must define exactly one parameter: (layer)."
            )

        cls._custom_layer_parsers[layer_cls] = parser_fn

    @classmethod
    def get_custom_parsers(cls):
        """Return the dict of all registered custom parser functions."""
        return cls._custom_layer_parsers

    def __init__(self):
        # Initialize the layer parsers mapping
        self.layer_parsers: Dict[Type[tf.keras.layers.Layer], Callable] = {
            tf.keras.layers.InputLayer: self.parse_input,
            tf.keras.layers.Conv2D: self.parse_conv2d,
            tf.keras.layers.Dense: self.parse_dense,
            tf.keras.layers.LSTM: self.parse_rnn,
            tf.keras.layers.GRU: self.parse_rnn,
            tf.keras.layers.Bidirectional: self.parse_rnn,
            tf.keras.layers.MaxPooling2D: self.parse_pooling,
            tf.keras.layers.AveragePooling2D: self.parse_pooling,
            tf.keras.layers.BatchNormalization: self.parse_batchnorm,
            tf.keras.layers.Dropout: self.parse_dropout,
            tf.keras.layers.Reshape: self.parse_reshape,
            tf.keras.layers.Flatten: self.parse_flatten,
            tf.keras.layers.Activation: self.parse_activation,
        }

        # Merge in any custom user-registered parsers from the class-level registry
        for layer_cls, parse_fn in self.get_custom_parsers().items():
            self.layer_parsers[layer_cls] = parse_fn

    def parse_model(self, model: tf.keras.models.Model) -> str:
        """
        Parse a TensorFlow Keras model into a VGSL spec string.

        Parameters
        ----------
        model : tf.keras.models.Model
            Keras model to be converted.

        Returns
        -------
        str
            VGSL spec string.

        Raises
        ------
        ValueError
            If the model contains unsupported layers or if the input shape is invalid.
        """
        configs = []

        # Handle InputLayer
        if not isinstance(model.layers[0], tf.keras.layers.InputLayer):
            input_layer = tf.keras.layers.InputLayer(
                input_shape=model.input_shape[1:], batch_size=model.input_shape[0]
            )
            input_config = self.parse_input(input_layer)
            configs.append(input_config)

        # Iterate through all layers in the model
        for idx, layer in enumerate(model.layers):
            layer_type = type(layer)
            parser_func = self.layer_parsers.get(layer_type, None)

            if parser_func:
                # Parse the layer
                config = parser_func(layer)

                # Append the config if not None
                if config:
                    configs.append(config)
            else:
                raise ValueError(
                    f"Unsupported layer type {layer_type.__name__} at position {idx}."
                )

        # Generate VGSL spec string from configs
        return self.generate_vgsl(configs)

    # Parser methods for different layer types

    def parse_input(self, layer: tf.keras.layers.InputLayer) -> InputConfig:
        """
        Parse an InputLayer into an InputConfig dataclass.

        Parameters
        ----------
        layer : tf.keras.layers.InputLayer
            The InputLayer to parse.

        Returns
        -------
        InputConfig
            The configuration for the input layer.
        """
        input_shape = layer.output.shape
        # Assuming input_shape: (batch_size, depth, height, width, channels)
        if len(input_shape) == 5:
            batch_size, depth, height, width, channels = input_shape
        elif len(input_shape) == 4:
            batch_size, height, width, channels = input_shape
            depth = -1
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")

        return InputConfig(
            batch_size=batch_size,
            depth=depth,
            height=height,
            width=width,
            channels=channels,
        )

    def parse_conv2d(self, layer: tf.keras.layers.Conv2D) -> Conv2DConfig:
        """
        Parse a Conv2D layer into a Conv2DConfig dataclass.

        Parameters
        ----------
        layer : tf.keras.layers.Conv2D
            The Conv2D layer to parse.

        Returns
        -------
        Conv2DConfig
            The configuration for the Conv2D layer.
        """
        activation = self._extract_activation(layer)
        return Conv2DConfig(
            activation=activation,
            kernel_size=layer.kernel_size,
            strides=layer.strides,
            filters=layer.filters,
        )

    def parse_dense(self, layer: tf.keras.layers.Dense) -> DenseConfig:
        """
        Parse a Dense layer into a DenseConfig dataclass.

        Parameters
        ----------
        layer : tf.keras.layers.Dense
            The Dense layer to parse.

        Returns
        -------
        DenseConfig
            The configuration for the Dense layer.
        """
        activation = self._extract_activation(layer)
        return DenseConfig(activation=activation, units=layer.units)

    def parse_rnn(
        self,
        layer: Union[
            tf.keras.layers.LSTM, tf.keras.layers.GRU, tf.keras.layers.Bidirectional
        ],
    ) -> RNNConfig:
        """
        Parse an RNN layer (LSTM, GRU, or Bidirectional) into an RNNConfig dataclass.

        Parameters
        ----------
        layer : Union[tf.keras.layers.LSTM, tf.keras.layers.GRU, tf.keras.layers.Bidirectional]
            The RNN layer to parse.

        Returns
        -------
        RNNConfig
            The configuration for the RNN layer.
        """
        if isinstance(layer, tf.keras.layers.Bidirectional):
            wrapped_layer = layer.forward_layer
            bidirectional = True
        else:
            wrapped_layer = layer
            bidirectional = False

        if isinstance(wrapped_layer, tf.keras.layers.LSTM):
            rnn_type = "lstm"
        elif isinstance(wrapped_layer, tf.keras.layers.GRU):
            rnn_type = "gru"
        else:
            raise ValueError(
                f"Unsupported RNN layer type {type(wrapped_layer).__name__}."
            )

        return RNNConfig(
            units=wrapped_layer.units,
            return_sequences=wrapped_layer.return_sequences,
            go_backwards=wrapped_layer.go_backwards if not bidirectional else False,
            dropout=wrapped_layer.dropout,
            recurrent_dropout=wrapped_layer.recurrent_dropout,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
        )

    def parse_pooling(
        self,
        layer: Union[tf.keras.layers.MaxPooling2D, tf.keras.layers.AveragePooling2D],
    ) -> Pooling2DConfig:
        """
        Parse a Pooling layer into a Pooling2DConfig dataclass.

        Parameters
        ----------
        layer : tf.keras.layers.MaxPooling2D or tf.keras.layers.AveragePooling2D
            The Pooling layer to parse.

        Returns
        -------
        Pooling2DConfig
            The configuration for the Pooling layer.
        """
        if isinstance(layer, tf.keras.layers.MaxPooling2D):
            pool_type = "max"
        elif isinstance(layer, tf.keras.layers.AveragePooling2D):
            pool_type = "average"

        return Pooling2DConfig(
            pool_type=pool_type,
            pool_size=layer.pool_size,
            strides=layer.strides if layer.strides else layer.pool_size,
        )

    def parse_batchnorm(self, layer: tf.keras.layers.BatchNormalization) -> None:
        """
        Parse a BatchNormalization layer.
        Since BatchNormalization does not require a VGSL spec beyond 'Bn', return a placeholder.

        Parameters
        ----------
        layer : tf.keras.layers.BatchNormalization
            The BatchNormalization layer to parse.

        Returns
        -------
        None
            Indicates that the VGSL spec should include 'Bn'.
        """
        return "Bn"

    def parse_dropout(self, layer: tf.keras.layers.Dropout) -> DropoutConfig:
        """
        Parse a Dropout layer into a DropoutConfig dataclass.

        Parameters
        ----------
        layer : tf.keras.layers.Dropout
            The Dropout layer to parse.

        Returns
        -------
        DropoutConfig
            The configuration for the Dropout layer.
        """
        return DropoutConfig(rate=layer.rate)

    def parse_flatten(self, layer: tf.keras.layers.Flatten) -> None:
        """
        Parse a Flatten layer.
        Since Flatten does not require a VGSL spec beyond 'Flatten', return a placeholder.

        Parameters
        ----------
        layer : tf.keras.layers.Flatten
            The Flatten layer to parse.

        Returns
        -------
        None
            Indicates that the VGSL spec should include 'Flatten'.
        """
        return "Flt"

    def parse_reshape(self, layer: tf.keras.layers.Reshape) -> ReshapeConfig:
        """
        Parse a Reshape layer into a ReshapeConfig dataclass.

        Parameters
        ----------
        layer : tf.keras.layers.Reshape
            The Reshape layer to parse.

        Returns
        -------
        ReshapeConfig
            The configuration for the Reshape layer.
        """
        target_shape = layer.target_shape
        return ReshapeConfig(target_shape=target_shape)

    def parse_activation(self, layer: tf.keras.layers.Activation) -> ActivationConfig:
        """
        Parse an Activation layer.

        Parameters
        ----------
        layer : tf.keras.layers.Activation
            The Activation layer to parse.

        Returns
        -------
        ActivationConfig
            The configuration for the Activation layer.
        """
        activation = self._extract_activation(layer)
        return ActivationConfig(activation=activation)

    # Helper methods
    def _extract_activation(self, layer: tf.keras.layers.Layer) -> str:
        """
        Extract the activation function from a TensorFlow Keras layer.

        Parameters
        ----------
        layer : tf.keras.layers.Layer
            The layer from which to extract the activation.

        Returns
        -------
        str
            The activation function name.
        """
        if hasattr(layer, "activation") and callable(layer.activation):
            activation = layer.activation.__name__
        elif isinstance(layer, tf.keras.layers.Activation):
            activation = layer.activation.__name__
        else:
            activation = "linear"
        return activation


def register_custom_parser(layer_cls: Type[tf.keras.layers.Layer]) -> Callable:
    """
    Decorator to register a custom parser function for a given TensorFlow Keras layer class.

    This allows users to extend `TensorFlowModelParser` by defining a function that
    converts a TensorFlow Keras layer into a VGSL specification.

    Parameters
    ----------
    layer_cls : Type[tf.keras.layers.Layer]
        The TensorFlow Keras layer class to associate with the parser function.

    Returns
    -------
    Callable
        A decorator that registers the provided function as a parser for `layer_cls`.

    Raises
    ------
    ValueError
        If a parser for `layer_cls` is already registered or if the function does not
        accept exactly one argument (the layer instance).

    Examples
    --------
    Registering a custom parser for a `MyCustomLayer`:

    >>> from vgslify.model_parsers.tensorflow import register_custom_parser
    >>> import tensorflow as tf
    >>> class MyCustomLayer(tf.keras.layers.Layer):
    ...     def __init__(self, units: int):
    ...         super().__init__()
    ...         self.units = units
    ...
    >>> @register_custom_parser(MyCustomLayer)
    ... def parse_my_custom_layer(layer: MyCustomLayer):
    ...     return f"MyCustomSpec({layer.units})"
    ...
    >>> # Now the parser is automatically registered inside TensorFlowModelParser
    """

    def decorator(fn: Callable) -> Callable:
        TensorFlowModelParser.register(layer_cls, fn)
        return fn

    return decorator
