# Imports

# > Standard library
from typing import Tuple

# > Third-party dependencies
import tensorflow as tf

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

    def __init__(self, input_shape: Tuple[int, ...] = None):
        """
        Initialize the TensorFlowLayerFactory.

        Parameters
        ----------
        input_shape : tuple of int, optional
            The input shape for the model, excluding batch size.
        """
        super().__init__(input_shape, data_format='channels_first')

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

    # Layer creation methods
    def _input(self, config, input_shape: Tuple[int, ...]):
        """
        Create a TensorFlow Input layer.

        Parameters
        ----------
        config : object
            Configuration object containing batch_size.
        input_shape : tuple of int
            The input shape for the layer.

        Returns
        -------
        tf.keras.layers.Input
            The created Input layer.
        """
        return tf.keras.Input(shape=input_shape, batch_size=config.batch_size)

    def _conv2d(self, config):
        """
        Create a TensorFlow Conv2D layer.

        Parameters
        ----------
        config : object
            Configuration object containing filters, kernel_size, and strides.

        Returns
        -------
        tf.keras.layers.Conv2D
            The created Conv2D layer.
        """
        return tf.keras.layers.Conv2D(
            filters=config.filters,
            kernel_size=config.kernel_size,
            strides=config.strides,
            padding='same',
            activation=None
        )

    def _pooling2d(self, config):
        """
        Create a TensorFlow Pooling2D layer.

        Parameters
        ----------
        config : object
            Configuration object containing pool_type, pool_size, and strides.

        Returns
        -------
        tf.keras.layers.Layer
            The created Pooling2D layer (either MaxPooling2D or AveragePooling2D).
        """
        if config.pool_type == 'max':
            return tf.keras.layers.MaxPooling2D(
                pool_size=config.pool_size,
                strides=config.strides,
                padding='same'
            )
        if config.pool_type == 'avg':
            return tf.keras.layers.AveragePooling2D(
                pool_size=config.pool_size,
                strides=config.strides,
                padding='same'
            )

    def _dense(self, config):
        """
        Create a TensorFlow Dense layer.

        Parameters
        ----------
        config : object
            Configuration object containing units.

        Returns
        -------
        tf.keras.layers.Dense
            The created Dense layer.
        """
        return tf.keras.layers.Dense(
            units=config.units,
            activation=None
        )

    def _rnn(self, config):
        """
        Create a TensorFlow RNN layer (LSTM or GRU).

        Parameters
        ----------
        config : object
            Configuration object containing rnn_type, units, return_sequences,
            go_backwards, dropout, and recurrent_dropout.

        Returns
        -------
        tf.keras.layers.Layer
            The created RNN layer (either LSTM or GRU).

        Raises
        ------
        ValueError
            If an unsupported RNN type is specified.
        """
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

    def _bidirectional(self, config):
        """
        Create a TensorFlow Bidirectional RNN layer.

        Parameters
        ----------
        config : object
            Configuration object containing rnn_type, units, dropout, and recurrent_dropout.

        Returns
        -------
        tf.keras.layers.Bidirectional
            The created Bidirectional RNN layer.
        """
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

    def _batchnorm(self):
        """
        Create a TensorFlow BatchNormalization layer.

        Returns
        -------
        tf.keras.layers.BatchNormalization
            The created BatchNormalization layer.
        """
        return tf.keras.layers.BatchNormalization()

    def _dropout(self, rate: float):
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

    def _reshape(self, target_shape: Tuple[int, ...]):
        """
        Create a TensorFlow Reshape layer.

        Parameters
        ----------
        target_shape : tuple of int
            The target shape to reshape to, excluding the batch size.

        Returns
        -------
        tf.keras.layers.Reshape
            The created Reshape layer.
        """
        return tf.keras.layers.Reshape(target_shape=target_shape)

    def _flatten(self):
        """
        Create a TensorFlow Flatten layer.

        Returns
        -------
        tf.keras.layers.Flatten
            The created Flatten layer.
        """
        return tf.keras.layers.Flatten()