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
        super().__init__(input_shape)

    def conv2d(self, spec: str):
        """
        Create a Conv2D layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Conv2D layer.

        Returns
        -------
        tf.keras.layers.Layer
            The created Conv2D layer.

        Examples
        --------
        >>> from vgslify.tensorflow.layers import TensorFlowLayerFactory
        >>> factory = TensorFlowLayerFactory(input_shape=(32, 32, 3))
        >>> conv_layer = factory.conv2d("Cr3,3,64")
        >>> print(conv_layer)
        <keras.src.layers.convolutional.conv2d.Conv2D object at ...>
        """
        config = parse_conv2d_spec(spec)
        if self.shape is None:
            raise ValueError("Input shape must be set before adding layers.")

        conv_layer = tf.keras.layers.Conv2D(
            filters=config.filters,
            kernel_size=config.kernel_size,
            strides=config.strides,
            padding='same',
            activation=config.activation
        )

        self.layers.append(conv_layer)
        # Update shape
        self.shape = self._compute_conv_output_shape(
            self.shape, config, data_format='channels_last')

        return conv_layer

    def maxpooling2d(self, spec: str) -> tf.keras.layers.Layer:
        """
        Create a MaxPooling2D layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the MaxPooling2D layer.

        Returns
        -------
        tf.keras.layers.Layer
            The created MaxPooling2D layer.

        Examples
        --------
        >>> from vgslify.tensorflow.layers import TensorFlowLayerFactory
        >>> factory = TensorFlowLayerFactory(input_shape=(32, 32, 3))
        >>> maxpool_layer = factory.maxpooling2d("Mp2,2,2,2")
        >>> print(maxpool_layer)
        <keras.src.layers.pooling.max_pooling2d.MaxPooling2D object at ...>
        """
        config = parse_pooling2d_spec(spec)
        layer = tf.keras.layers.MaxPooling2D(
            pool_size=config.pool_size,
            strides=config.strides,
            padding='same'
        )
        self.layers.append(layer)
        # Update shape
        self.shape = self._compute_pool_output_shape(
            self.shape, config, data_format='channels_last')
        return layer

    def avgpool2d(self, spec: str) -> tf.keras.layers.Layer:
        """
        Create an AvgPool2D layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the AvgPool2D layer.

        Returns
        -------
        tf.keras.layers.Layer
            The created AvgPool2D layer.

        Examples
        --------
        >>> from vgslify.tensorflow.layers import TensorFlowLayerFactory
        >>> factory = TensorFlowLayerFactory(input_shape=(32, 32, 3))
        >>> avgpool_layer = factory.avgpool2d("Ap2,2,2,2")
        >>> print(avgpool_layer)
        <keras.src.layers.pooling.average_pooling2d.AveragePooling2D object at ...>
        """
        config = parse_pooling2d_spec(spec)
        layer = tf.keras.layers.AvgPool2D(
            pool_size=config.pool_size,
            strides=config.strides,
            padding='same'
        )
        self.layers.append(layer)
        # Update shape
        self.shape = self._compute_pool_output_shape(
            self.shape, config, data_format='channels_last')
        return layer

    def dense(self, spec: str) -> tf.keras.layers.Layer:
        """
        Create a Dense layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Dense layer.

        Returns
        -------
        tf.keras.layers.Layer
            The created Dense layer.

        Examples
        --------
        >>> from vgslify.tensorflow.layers import TensorFlowLayerFactory
        >>> factory = TensorFlowLayerFactory(input_shape=(32,))
        >>> dense_layer = factory.dense("Dr64")
        >>> print(dense_layer)
        <keras.src.layers.core.dense.Dense object at ...>
        """
        config = parse_dense_spec(spec)
        if self.shape is None:
            raise ValueError("Input shape must be set before adding layers.")

        dense_layer = tf.keras.layers.Dense(
            units=config.units,
            activation=config.activation
        )
        self.layers.append(dense_layer)

        # Update shape
        self.shape = (config.units,)

        return dense_layer

    def lstm(self, spec: str) -> tf.keras.layers.Layer:
        """
        Create an LSTM layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the LSTM layer.

        Returns
        -------
        tf.keras.layers.Layer
            The created LSTM layer.

        Examples
        --------
        >>> from vgslify.tensorflow.layers import TensorFlowLayerFactory
        >>> factory = TensorFlowLayerFactory(input_shape=(10, 32))
        >>> lstm_layer = factory.lstm("Lf64")
        >>> print(lstm_layer)
        <keras.src.layers.rnn.lstm.LSTM object at ...>
        """
        config = parse_rnn_spec(spec)
        if self.shape is None:
            raise ValueError("Input shape must be set before adding layers.")

        lstm_layer = tf.keras.layers.LSTM(
            units=config.units,
            return_sequences=config.return_sequences,
            go_backwards=config.go_backwards,
            dropout=config.dropout,
            recurrent_dropout=config.recurrent_dropout
        )
        self.layers.append(lstm_layer)

        # Update shape
        if config.return_sequences:
            self.shape = (self.shape[0], config.units)
        else:
            self.shape = (config.units,)

        return lstm_layer

    def gru(self, spec: str) -> tf.keras.layers.Layer:
        """
        Create a GRU layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the GRU layer.

        Returns
        -------
        tf.keras.layers.Layer
            The created GRU layer.

        Examples
        --------
        >>> from vgslify.tensorflow.layers import TensorFlowLayerFactory
        >>> factory = TensorFlowLayerFactory(input_shape=(10, 32))
        >>> gru_layer = factory.gru("Gf64")
        >>> print(gru_layer)
        <keras.src.layers.rnn.gru.GRU object at ...>
        """
        config = parse_rnn_spec(spec)
        if self.shape is None:
            raise ValueError("Input shape must be set before adding layers.")

        gru_layer = tf.keras.layers.GRU(
            units=config.units,
            return_sequences=config.return_sequences,
            go_backwards=config.go_backwards,
            dropout=config.dropout,
            recurrent_dropout=config.recurrent_dropout
        )
        self.layers.append(gru_layer)

        # Update shape
        if config.return_sequences:
            self.shape = (self.shape[0], config.units)
        else:
            self.shape = (config.units,)

        return gru_layer

    def bidirectional(self, spec: str) -> tf.keras.layers.Layer:
        """
        Create a Bidirectional RNN layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Bidirectional layer.

        Returns
        -------
        tf.keras.layers.Layer
            The created Bidirectional RNN layer.

        Examples
        --------
        >>> from vgslify.tensorflow.layers import TensorFlowLayerFactory
        >>> factory = TensorFlowLayerFactory(input_shape=(10, 32))
        >>> bidirectional_layer = factory.bidirectional("Bl64")
        >>> print(bidirectional_layer)
        <keras.src.layers.rnn.bidirectional.Bidirectional object at ...>
        """
        config = parse_rnn_spec(spec)
        if self.shape is None:
            raise ValueError("Input shape must be set before adding layers.")

        rnn_layer_class = tf.keras.layers.LSTM if config.rnn_type == 'l' else tf.keras.layers.GRU

        rnn_layer = rnn_layer_class(
            units=config.units,
            return_sequences=True,
            dropout=config.dropout,
            recurrent_dropout=config.recurrent_dropout
        )

        bidirectional_layer = tf.keras.layers.Bidirectional(
            rnn_layer,
            merge_mode='concat'
        )
        self.layers.append(bidirectional_layer)

        # Update shape
        self.shape = (self.shape[0], config.units * 2)

        return bidirectional_layer

    def batchnorm(self, spec: str) -> tf.keras.layers.Layer:
        """
        Create a BatchNormalization layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the BatchNormalization layer.

        Returns
        -------
        tf.keras.layers.Layer
            The created BatchNormalization layer.

        Examples
        --------
        >>> from vgslify.tensorflow.layers import TensorFlowLayerFactory
        >>> factory = TensorFlowLayerFactory(input_shape=(32, 32, 3))
        >>> batchnorm_layer = factory.batchnorm("Bn")
        >>> print(batchnorm_layer)
        <keras.src.layers.normalization.batch_normalization.BatchNormalization object at ...>
        """
        if spec != 'Bn':
            raise ValueError(
                f"BatchNormalization layer spec '{spec}' is incorrect. Expected 'Bn'.")

        layer = tf.keras.layers.BatchNormalization()
        self.layers.append(layer)
        # Shape remains the same
        return layer

    def input(self, spec: str) -> tf.keras.layers.Input:
        """
        Create an Input layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Input layer.

        Returns
        -------
        tf.keras.layers.Input
            The created Input layer.

        Examples
        --------
        >>> from vgslify.tensorflow.layers import TensorFlowLayerFactory
        >>> factory = TensorFlowLayerFactory()
        >>> input_layer = factory.input("32,32,3")
        >>> print(input_layer)
        <keras.src.engine.input_layer.InputLayer object at ...>
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

        self.shape = input_shape
        self._input_shape = input_shape
        input_layer = tf.keras.Input(
            shape=input_shape, batch_size=config.batch_size)
        self.layers.append(input_layer)
        return input_layer

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
