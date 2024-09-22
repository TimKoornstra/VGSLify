# Imports

# > Third-party dependencies
import tensorflow as tf

# > Internal dependencies
from vgslify.core.factory import LayerFactory
from vgslify.core.parser import (parse_conv2d_spec, parse_pooling2d_spec,
                                 parse_dense_spec, parse_rnn_spec,
                                 parse_dropout_spec, parse_activation_spec,
                                 parse_reshape_spec, parse_input_spec)


class TensorFlowLayerFactory(LayerFactory):
    """
    TensorFlowLayerFactory is responsible for creating TensorFlow-specific layers based on parsed
    VGSL (Variable-size Graph Specification Language) specifications. This factory handles the
    creation of various types of layers, including convolutional layers, pooling layers, RNN layers,
    dense layers, activation layers, and more.

    This class maintains an internal state to track the shape of the tensor as layers are added.
    """

    def __init__(self):
        super().__init__()
        self.layers = []
        self.shape = None  # Shape excluding batch size

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
        self.shape = self._compute_conv_output_shape(self.shape, config)

        return conv_layer

    def _compute_conv_output_shape(self, input_shape, config):
        """
        Computes the output shape of a convolutional layer.

        Parameters
        ----------
        input_shape : tuple
            The input shape (H, W, C).
        config : Any
            The configuration object returned by parse_conv2d_spec.

        Returns
        -------
        tuple
            The output shape after the convolution.
        """
        H_in, W_in, C_in = input_shape
        C_out = config.filters

        # For 'same' padding
        H_out = int((H_in + config.strides[0] - 1) //
                    config.strides[0]) if H_in is not None else None
        W_out = int((W_in + config.strides[1] - 1) //
                    config.strides[1]) if W_in is not None else None

        return (H_out, W_out, C_out)

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
        """
        config = parse_pooling2d_spec(spec)
        layer = tf.keras.layers.MaxPooling2D(
            pool_size=config.pool_size,
            strides=config.strides,
            padding='same'
        )
        self.layers.append(layer)
        # Update shape
        self.shape = self._compute_pool_output_shape(self.shape, config)
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
        """
        config = parse_pooling2d_spec(spec)
        layer = tf.keras.layers.AvgPool2D(
            pool_size=config.pool_size,
            strides=config.strides,
            padding='same'
        )
        self.layers.append(layer)
        # Update shape
        self.shape = self._compute_pool_output_shape(self.shape, config)
        return layer

    def _compute_pool_output_shape(self, input_shape, config):
        """
        Computes the output shape of a pooling layer.

        Parameters
        ----------
        input_shape : tuple
            The input shape (H, W, C).
        config : Any
            The configuration object returned by parse_pooling2d_spec.

        Returns
        -------
        tuple
            The output shape after pooling.
        """
        H_in, W_in, C_in = input_shape

        H_out = int((H_in + config.strides[0] - 1) //
                    config.strides[0]) if H_in is not None else None
        W_out = int((W_in + config.strides[1] - 1) //
                    config.strides[1]) if W_in is not None else None

        return (H_out, W_out, C_in)

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

    def dropout(self, spec: str) -> tf.keras.layers.Layer:
        """
        Create a Dropout layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Dropout layer.

        Returns
        -------
        tf.keras.layers.Layer
            The created Dropout layer.
        """
        config = parse_dropout_spec(spec)
        layer = tf.keras.layers.Dropout(rate=config.rate)
        self.layers.append(layer)
        # Shape remains the same
        return layer

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
        """
        if spec != 'Bn':
            raise ValueError(
                f"BatchNormalization layer spec '{spec}' is incorrect. Expected 'Bn'.")

        layer = tf.keras.layers.BatchNormalization()
        self.layers.append(layer)
        # Shape remains the same
        return layer

    def activation(self, spec: str) -> tf.keras.layers.Layer:
        """
        Create an Activation layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Activation layer.

        Returns
        -------
        tf.keras.layers.Layer
            The created Activation layer.
        """
        activation_function = parse_activation_spec(spec)
        layer = tf.keras.layers.Activation(activation=activation_function)
        self.layers.append(layer)
        # Shape remains the same
        return layer

    def reshape(self, spec: str) -> tf.keras.layers.Layer:
        """
        Create a Reshape layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            VGSL specification string for the Reshape layer.

        Returns
        -------
        tf.keras.layers.Layer
            The created Reshape layer.
        """
        if self.shape is None:
            raise ValueError("Input shape must be set before adding layers.")

        # Handle 'Rc' (collapse spatial dimensions) specification
        if spec.startswith('Rc'):
            if spec == 'Rc2':
                # Flatten to (batch_size, -1)
                layer = tf.keras.layers.Flatten()
                self.layers.append(layer)
                self.shape = (int(tf.reduce_prod(self.shape).numpy()),)
                return layer

            elif spec == 'Rc3':
                # Reshape to (batch_size, timesteps, features) for RNN compatibility
                H, W, C = self.shape
                timesteps = H * W
                features = C
                layer = tf.keras.layers.Reshape((timesteps, features))
                self.layers.append(layer)
                self.shape = (timesteps, features)
                return layer

            else:
                raise ValueError(f"Unsupported Rc variant: {spec}")

        # Handle regular reshape (e.g., 'R64,64,3')
        config = parse_reshape_spec(spec)
        layer = tf.keras.layers.Reshape(target_shape=config.target_shape)
        self.layers.append(layer)
        self.shape = config.target_shape
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
        input_layer = tf.keras.Input(
            shape=input_shape, batch_size=config.batch_size)
        self.layers.append(input_layer)
        return input_layer

    def flatten(self, spec: str) -> tf.keras.layers.Layer:
        """
        Create a Flatten layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Flatten layer.

        Returns
        -------
        tf.keras.layers.Layer
            The created Flatten layer.
        """
        if spec != "Flt":
            raise ValueError(
                f"Flatten layer spec '{spec}' is incorrect. Expected 'Flt'.")

        layer = tf.keras.layers.Flatten()
        self.layers.append(layer)
        # Update shape
        self.shape = (int(tf.reduce_prod(self.shape).numpy()),)
        return layer

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
        """
        inputs = self.layers[0]
        outputs = inputs
        for layer in self.layers[1:]:
            outputs = layer(outputs)
        model = tf.keras.models.Model(
            inputs=inputs, outputs=outputs, name=name)
        return model
