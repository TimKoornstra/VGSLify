# Imports

# > Standard library
from typing import Tuple

# > Third-party dependencies
import tensorflow as tf

# > Internal dependencies
from vgslify.core.factory import LayerFactory
from vgslify.core.parser import (parse_conv2d_spec, parse_pooling2d_spec,
                                 parse_dense_spec, parse_rnn_spec,
                                 parse_input_spec)


class TensorFlowLayerFactory(LayerFactory):
    """
    TensorFlowLayerFactory is responsible for creating TensorFlow-specific layers based on parsed
    VGSL (Variable-size Graph Specification Language) specifications. This factory handles the
    creation of various types of layers, including convolutional layers, pooling layers, RNN layers,
    dense layers, activation layers, and more.

    This class maintains an internal state to track the shape of the tensor as layers are added.

    Attributes
    ----------
    layers : list
        A list of TensorFlow layers that have been added to the factory.
    shape : tuple of int
        The current shape of the tensor, excluding the batch size.
    _input_shape : tuple of int or None
        The original input shape provided during initialization.
    """

    def __init__(self, input_shape: Tuple[int, ...] = None):
        super().__init__(input_shape, data_format='channels_first')

    def build_final_model(self, name: str = "VGSL_Model") -> tf.keras.models.Model:
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

        Examples
        --------
        >>> from vgslify.tensorflow.layers import TensorFlowLayerFactory
        >>> factory = TensorFlowLayerFactory(input_shape=(32, 32, 3))
        >>> factory.conv2d("Cr3,3,64")
        >>> factory.maxpooling2d("Mp2,2,2,2")
        >>> model = factory.build_final_model()
        >>> model.summary()
        Model: "VGSL_Model"
        ...
        """
        if not self.layers:
            raise ValueError("No layers added to the model.")
        if not self._input_shape:
            raise ValueError("No input shape specified for the model.")

        # If we do not have an input layer, add one
        if not isinstance(self.layers[0], tf.keras.layers.InputLayer):
            input_layer = tf.keras.Input(shape=self._input_shape)
            self.layers.insert(0, input_layer)

        inputs = self.layers[0]
        outputs = inputs
        for layer in self.layers[1:]:
            outputs = layer(outputs)
        model = tf.keras.models.Model(
            inputs=inputs, outputs=outputs, name=name)
        return model

    def _create_conv2d_layer(self, config):
        return tf.keras.layers.Conv2D(
            filters=config.filters,
            kernel_size=config.kernel_size,
            strides=config.strides,
            padding='same',
            activation=None
        )

    def _create_pooling2d_layer(self, config):
        if config.pool_type == 'max':
            return tf.keras.layers.MaxPooling2D(
                pool_size=config.pool_size,
                strides=config.strides,
                padding='same'
            )
        if config.pool_type == 'avg':
            return tf.keras.layers.AvgPool2D(
                pool_size=config.pool_size,
                strides=config.strides,
                padding='same'
            )

    def _create_dense_layer(self, config):
        return tf.keras.layers.Dense(
            units=config.units,
            activation=None
        )

    def _create_rnn_layer(self, config):
        if config.rnn_type == 'L':
            return tf.keras.layers.LSTM(
                units=config.units,
                return_sequences=config.return_sequences,
                go_backwards=config.go_backwards,
                dropout=config.dropout,
                recurrent_dropout=config.recurrent_dropout
            )
        elif config.rnn_type == 'G':
            return tf.keras.layers.GRU(
                units=config.units,
                return_sequences=config.return_sequences,
                go_backwards=config.go_backwards,
                dropout=config.dropout,
                recurrent_dropout=config.recurrent_dropout
            )
        else:
            raise ValueError(f"Unsupported RNN type: {config.rnn_type}")

    def _create_bidirectional_layer(self, config):
        rnn_layer_class = tf.keras.layers.LSTM if config.rnn_type == 'L' else tf.keras.layers.GRU

        return tf.keras.layers.Bidirectional(
            rnn_layer_class(
                units=config.units,
                return_sequences=True,
                dropout=config.dropout,
                recurrent_dropout=config.recurrent_dropout
            ),
            merge_mode='concat'
        )

    def _create_input_layer(self, config, input_shape: Tuple[int, ...]):
        # Create a TensorFlow Input layer with the given input shape.
        return tf.keras.Input(shape=input_shape, batch_size=config.batch_size)

    def _create_batchnorm_layer(self):
        """
        Create a TensorFlow BatchNormalization layer.

        Returns
        -------
        tf.keras.layers.BatchNormalization
            The created BatchNormalization layer.
        """
        return tf.keras.layers.BatchNormalization()

    def _create_dropout_layer(self, rate: float):
        """
        Create a TensorFlow Dropout layer.

        Parameters
        ----------
        rate : float
            Dropout rate, between 0 and 1.

        Returns
        -------
        tf.keras.layers.Dropout
            The created Dropout layer.
        """
        return tf.keras.layers.Dropout(rate=rate)

    def _create_activation_layer(self, activation_function: str):
        """
        Create a TensorFlow activation layer.

        Parameters
        ----------
        activation_function : str
            Name of the activation function. Supported values are 'softmax', 'tanh', 'relu',
            'linear', 'sigmoid', etc.

        Returns
        -------
        tf.keras.layers.Layer
            The created activation layer.
        """
        return tf.keras.layers.Activation(activation=activation_function)

    def _create_reshape_layer(self, target_shape: Tuple[int, ...]):
        """
        Create a TensorFlow Reshape layer.

        Parameters
        ----------
        target_shape : tuple
            The target shape to reshape to, excluding the batch size.

        Returns
        -------
        tf.keras.layers.Layer
            The created Reshape layer.
        """
        return tf.keras.layers.Reshape(target_shape=target_shape)

    def _create_flatten_layer(self):
        """
        Create a TensorFlow Flatten layer.

        Returns
        -------
        tf.keras.layers.Layer
            The created Flatten layer.
        """
        return tf.keras.layers.Flatten()
