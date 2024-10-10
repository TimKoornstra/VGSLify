import tensorflow as tf
from typing import Callable, Dict, Type, Union
from vgslify.core.config import (
    ActivationConfig,
    Conv2DConfig,
    Pooling2DConfig,
    DenseConfig,
    RNNConfig,
    DropoutConfig,
    ReshapeConfig,
    InputConfig
)
from vgslify.parsers.base_parser import BaseModelParser

class TensorFlowModelParser(BaseModelParser):
    """
    Parser for TensorFlow Keras models to convert them into VGSL spec strings using configuration dataclasses.
    """

    def __init__(self):
        # Initialize the layer parsers mapping
        self.layer_parsers: Dict[Type[tf.keras.layers.Layer], Callable] = {
            tf.keras.layers.InputLayer: self.parse_input_layer,
            tf.keras.layers.Conv2D: self.parse_conv2d,
            tf.keras.layers.Dense: self.parse_dense,
            tf.keras.layers.LSTM: lambda layer: self.parse_rnn(layer, "lstm"),
            tf.keras.layers.GRU: lambda layer: self.parse_rnn(layer, "gru"),
            tf.keras.layers.Bidirectional: self.parse_bidirectional,
            tf.keras.layers.MaxPooling2D: lambda layer: self.parse_pooling(layer, "max"),
            tf.keras.layers.AveragePooling2D: lambda layer: self.parse_pooling(layer, "average"),
            tf.keras.layers.BatchNormalization: self.parse_batchnorm,
            tf.keras.layers.Dropout: self.parse_dropout,
            tf.keras.layers.Reshape: self.parse_reshape,
            tf.keras.layers.Flatten: self.parse_flatten,
            tf.keras.layers.Activation: self.parse_activation
        }

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
                input_shape=model.input_shape[1:],
                batch_size=model.input_shape[0]
            )
            input_config = self.parse_input_layer(input_layer)
            configs.append(input_config)

        # Iterate through all layers in the model
        for idx, layer in enumerate(model.layers):
            layer_type = type(layer)
            parser_func = self._get_parser(layer_type)

            if parser_func:
                # Parse the layer
                if isinstance(layer, tf.keras.layers.Reshape):
                    # Reshape may need previous layer info
                    prev_layer = model.layers[idx - 1] if idx > 0 else None
                    config = parser_func(layer, prev_layer)
                elif isinstance(layer, tf.keras.layers.Bidirectional):
                    config = parser_func(layer)
                else:
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

    def _get_parser(self, layer_type: Type[tf.keras.layers.Layer]) -> Callable:
        """
        Retrieve the parser function for a given layer type.

        Parameters
        ----------
        layer_type : Type[tf.keras.layers.Layer]
            The type of the layer.

        Returns
        -------
        Callable
            The corresponding parser function.
        """
        return self.layer_parsers.get(layer_type, None)

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
        if hasattr(layer, 'activation') and callable(layer.activation):
            activation = layer.activation.__name__
        elif isinstance(layer, tf.keras.layers.Activation):
            activation = layer.activation.__name__
        else:
            activation = 'linear'
        return activation

    # Parser methods for different layer types

    def parse_input_layer(self, layer: tf.keras.layers.InputLayer) -> InputConfig:
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
            channels=channels
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
            filters=layer.filters
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
        return DenseConfig(
            activation=activation,
            units=layer.units
        )

    def parse_rnn(self, layer: Union[tf.keras.layers.LSTM, tf.keras.layers.GRU], rnn_type: str) -> RNNConfig:
        """
        Parse an RNN layer (LSTM or GRU) into an RNNConfig dataclass.

        Parameters
        ----------
        layer : tf.keras.layers.LSTM or tf.keras.layers.GRU
            The RNN layer to parse.
        rnn_type : str
            The type identifier ('lstm' or 'gru').

        Returns
        -------
        RNNConfig
            The configuration for the RNN layer.
        """
        return RNNConfig(
            units=layer.units,
            return_sequences=layer.return_sequences,
            go_backwards=layer.go_backwards,
            dropout=layer.dropout,
            recurrent_dropout=layer.recurrent_dropout,
            rnn_type=rnn_type
        )

    def parse_bidirectional(self, layer: tf.keras.layers.Bidirectional) -> RNNConfig:
        """
        Parse a Bidirectional layer into an RNNConfig dataclass.

        Parameters
        ----------
        layer : tf.keras.layers.Bidirectional
            The Bidirectional layer to parse.

        Returns
        -------
        RNNConfig
            The configuration for the Bidirectional RNN layer.
        """
        wrapped_layer = layer.forward_layer
        if isinstance(wrapped_layer, tf.keras.layers.LSTM):
            rnn_type = 'lstm'
        elif isinstance(wrapped_layer, tf.keras.layers.GRU):
            rnn_type = 'gru'
        else:
            raise ValueError(f"Unsupported wrapped layer type {type(wrapped_layer).__name__} in Bidirectional layer.")

        config = self.parse_rnn(wrapped_layer, rnn_type)
        # Adjust for bidirectionality
        config.go_backwards = False  # Assuming forward; adjust if necessary
        # Optionally, modify units or other parameters if needed
        return config

    def parse_pooling(self, layer: Union[tf.keras.layers.MaxPooling2D, tf.keras.layers.AveragePooling2D], pool_type: str) -> Pooling2DConfig:
        """
        Parse a Pooling layer into a Pooling2DConfig dataclass.

        Parameters
        ----------
        layer : tf.keras.layers.MaxPooling2D or tf.keras.layers.AveragePooling2D
            The Pooling layer to parse.
        pool_type : str
            Type of pooling ('max' or 'average').

        Returns
        -------
        Pooling2DConfig
            The configuration for the Pooling layer.
        """
        return Pooling2DConfig(
            pool_type=pool_type,
            pool_size=layer.pool_size,
            strides=layer.strides if layer.strides else layer.pool_size
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
        return DropoutConfig(
            rate=layer.rate
        )

    def parse_reshape(self, layer: tf.keras.layers.Reshape, prev_layer: tf.keras.layers.Layer) -> ReshapeConfig:
        """
        Parse a Reshape layer into a ReshapeConfig dataclass.

        Parameters
        ----------
        layer : tf.keras.layers.Reshape
            The Reshape layer to parse.
        prev_layer : tf.keras.layers.Layer
            The previous layer in the model.

        Returns
        -------
        ReshapeConfig
            The configuration for the Reshape layer.
        """
        target_shape = layer.target_shape
        return ReshapeConfig(
            target_shape=target_shape
        )

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

    def parse_activation(self, layer: tf.keras.layers.Activation) -> None:
        """
        Parse an Activation layer.
        
        Parameters
        ----------
        layer : tf.keras.layers.Activation
            The Activation layer to parse.

        Returns
        -------
        str
            The activation function name.
        """
        activation = self._extract_activation(layer)
        return ActivationConfig(activation=activation)
