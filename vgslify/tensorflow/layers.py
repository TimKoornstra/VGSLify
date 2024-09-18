# Imports

# > Third-party dependencies
import tensorflow as tf

# > Internal dependencies
from vgslify.core.factory import LayerFactory
from vgslify.core.parser import (parse_conv2d_spec, parse_pooling2d_spec,
                                 parse_dense_spec, parse_rnn_spec,
                                 parse_dropout_spec, parse_output_layer_spec,
                                 parse_activation_spec, parse_reshape_spec,
                                 parse_input_spec)


class TensorFlowLayerFactory(LayerFactory):
    """
    TensorFlowLayerFactory is responsible for creating TensorFlow-specific layers based on parsed
    VGSL (Variable-size Graph Specification Language) specifications. This factory handles the
    creation of various types of layers, including convolutional layers, pooling layers, RNN layers,
    dense layers, activation layers, and more.

    This class abstracts the layer creation logic, allowing the VGSLModelGenerator to dynamically
    build models  without needing to know the specifics of TensorFlow operations.
    """
    @staticmethod
    def conv2d(spec: str):
        """
        Create a Conv2D layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Conv2D layer.

        Returns
        -------
        tf.keras.layers.Conv2D
            The created Conv2D layer.

        Raises
        ------
        ValueError
            If the provided VGSL spec string does not match the expected format.

        Examples
        --------
        >>> from vgslify.tensorflow.layers import TensorFlowLayerFactory
        >>> conv_layer = TensorFlowLayerFactory.conv2d("Cr3,3,64")
        >>> print(conv_layer)
        <keras.src.layers.convolutional.conv2d.Conv2D object at 0x7f8b1c0b1d30>
        """
        config = parse_conv2d_spec(spec)
        return tf.keras.layers.Conv2D(
            filters=config.filters,
            kernel_size=config.kernel_size,
            strides=config.strides,
            padding='same',
            activation=config.activation,
        )

    @staticmethod
    def maxpooling2d(spec: str) -> tf.keras.layers.MaxPooling2D:
        """
        Create a MaxPooling2D layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the MaxPooling2D layer.

        Returns
        -------
        tf.keras.layers.MaxPooling2D
            The created MaxPooling2D layer.

        Raises
        ------
        ValueError
            If the provided VGSL spec string does not match the expected format.

        Examples
        --------
        >>> from vgslify.tensorflow.layers import TensorFlowLayerFactory
        >>> maxpool_layer = TensorFlowLayerFactory.maxpooling2d("Mp2,2,2,2")
        >>> print(maxpool_layer)
        <keras.src.layers.pooling.max_pooling2d.MaxPooling2D object at 0x7f8b1c0b1d30>
        """
        config = parse_pooling2d_spec(spec)
        return tf.keras.layers.MaxPooling2D(
            pool_size=config.pool_size,
            strides=config.strides,
            padding='same'
        )

    @staticmethod
    def avgpool2d(spec: str) -> tf.keras.layers.AvgPool2D:
        """
        Create an AvgPool2D layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the AvgPool2D layer.

        Returns
        -------
        tf.keras.layers.AvgPool2D
            The created AvgPool2D layer.

        Raises
        ------
        ValueError
            If the provided VGSL spec string does not match the expected format.

        Examples
        --------
        >>> from vgslify.tensorflow.layers import TensorFlowLayerFactory
        >>> avgpool_layer = TensorFlowLayerFactory.avgpool2d("Ap2,2,2,2")
        >>> print(avgpool_layer)
        <keras.src.layers.pooling.average_pooling2d.AveragePooling2D object at 0x7f8b1c0b1d30>
        """
        config = parse_pooling2d_spec(spec)
        return tf.keras.layers.AvgPool2D(
            pool_size=config.pool_size,
            strides=config.strides,
            padding='same'
        )

    @staticmethod
    def dense(spec: str) -> tf.keras.layers.Dense:
        """
        Create a Dense layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Dense layer.

        Returns
        -------
        tf.keras.layers.Dense
            The created Dense layer.

        Raises
        ------
        ValueError
            If the provided VGSL spec string does not match the expected format.

        Examples
        --------
        >>> from vgslify.tensorflow.layers import TensorFlowLayerFactory
        >>> dense_layer = TensorFlowLayerFactory.dense("Fr64")
        >>> print(dense_layer)
        <keras.src.layers.core.dense.Dense object at 0x7f8b1c0b1d30>
        """
        config = parse_dense_spec(spec)
        return tf.keras.layers.Dense(
            units=config.units,
            activation=config.activation,
        )

    @staticmethod
    def lstm(spec: str) -> tf.keras.layers.LSTM:
        """
        Create an LSTM layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the LSTM layer.

        Returns
        -------
        tf.keras.layers.LSTM
            The created LSTM layer.

        Raises
        ------
        ValueError
            If the provided VGSL spec string does not match the expected format.

        Examples
        --------
        >>> from vgslify.tensorflow.layers import TensorFlowLayerFactory
        >>> lstm_layer = TensorFlowLayerFactory.lstm("Lr64,D20,Rd10")
        >>> print(lstm_layer)
        <keras.src.layers.recurrent.lstm.LSTM object at 0x7f8b1c0b1d30>
        """
        config = parse_rnn_spec(spec)
        return tf.keras.layers.LSTM(
            units=config.units,
            return_sequences=config.return_sequences,
            go_backwards=config.go_backwards,
            dropout=config.dropout,
            recurrent_dropout=config.recurrent_dropout
        )

    @staticmethod
    def gru(spec: str) -> tf.keras.layers.GRU:
        """
        Create a GRU layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the GRU layer.

        Returns
        -------
        tf.keras.layers.GRU
            The created GRU layer.

        Raises
        ------
        ValueError
            If the provided VGSL spec string does not match the expected format.

        Examples
        --------
        >>> from vgslify.tensorflow.layers import TensorFlowLayerFactory
        >>> gru_layer = TensorFlowLayerFactory.gru("Gr64,D20,Rd10")
        >>> print(gru_layer)
        <keras.src.layers.recurrent.gru.GRU object at 0x7f8b1c0b1d30>
        """
        config = parse_rnn_spec(spec)
        return tf.keras.layers.GRU(
            units=config.units,
            return_sequences=config.return_sequences,
            go_backwards=config.go_backwards,
            dropout=config.dropout,
            recurrent_dropout=config.recurrent_dropout
        )

    @staticmethod
    def bidirectional(spec: str) -> tf.keras.layers.Bidirectional:
        """
        Create a Bidirectional RNN layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Bidirectional layer.

        Returns
        -------
        tf.keras.layers.Bidirectional
            The created Bidirectional RNN layer.

        Raises
        ------
        ValueError
            If the provided VGSL spec string does not match the expected format.

        Examples
        --------
        >>> from vgslify.tensorflow.layers import TensorFlowLayerFactory
        >>> bidirectional_layer = TensorFlowLayerFactory.bidirectional("Blr64,D20,Rd10")
        >>> print(bidirectional_layer)
        <keras.src.layers.wrappers.bidirectional.Bidirectional object at 0x7f8b1c0b1d30>
        """
        config = parse_rnn_spec(spec)

        rnn_layer = tf.keras.layers.LSTM if config.rnn_type == 'l' else tf.keras.layers.GRU

        return tf.keras.layers.Bidirectional(
            rnn_layer(
                units=config.units,
                return_sequences=True,
                dropout=config.dropout,
                recurrent_dropout=config.recurrent_dropout
            ),
            merge_mode='concat'
        )

    @staticmethod
    def dropout(spec: str) -> tf.keras.layers.Dropout:
        """
        Create a Dropout layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Dropout layer.

        Returns
        -------
        tf.keras.layers.Dropout
            The created Dropout layer.

        Raises
        ------
        ValueError
            If the provided VGSL spec string does not match the expected format.

        Examples
        --------
        >>> from vgslify.tensorflow.layers import TensorFlowLayerFactory
        >>> dropout_layer = TensorFlowLayerFactory.dropout("D50")
        >>> print(dropout_layer)
        <keras.src.layers.core.dropout.Dropout object at 0x7f8b1c0b1d30>
        """
        config = parse_dropout_spec(spec)
        return tf.keras.layers.Dropout(rate=config.rate)

    @staticmethod
    def output(spec: str) -> tf.keras.layers.Dense:
        """
        Create an Output layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Output layer.

        Returns
        -------
        tf.keras.layers.Dense
            The created Output layer.

        Raises
        ------
        ValueError
            If the provided VGSL spec string does not match the expected format.

        Examples
        --------
        >>> from vgslify.tensorflow.layers import TensorFlowLayerFactory
        >>> output_layer = TensorFlowLayerFactory.output("O1s10")
        >>> print(output_layer)
        <keras.src.layers.core.dense.Dense object at 0x7f8b1c0b1d30>
        """
        config = parse_output_layer_spec(spec)
        return tf.keras.layers.Dense(
            units=config.units,
            activation=config.activation,
            kernel_initializer=tf.keras.initializers.GlorotNormal()
        )

    @staticmethod
    def batchnorm(spec: str) -> tf.keras.layers.BatchNormalization:
        """
        Create a BatchNormalization layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the BatchNormalization layer.

        Returns
        -------
        tf.keras.layers.BatchNormalization
            The created BatchNormalization layer.

        Raises
        ------
        ValueError
            If the provided VGSL spec string does not match the expected format.

        Examples
        --------
        >>> from vgslify.tensorflow.layers import TensorFlowLayerFactory
        >>> batchnorm_layer = TensorFlowLayerFactory.batchnorm("Bn")
        >>> print(batchnorm_layer)
        <keras.src.layers.normalization.batch_normalization.BatchNormalization object at 
        0x7f8b1c0b1d30>
        """
        if spec != 'Bn':
            raise ValueError(
                f"BatchNormalization layer spec '{spec}' is incorrect. Expected 'Bn'.")

        return tf.keras.layers.BatchNormalization()

    @staticmethod
    def activation(spec: str) -> tf.keras.layers.Activation:
        """
        Create an Activation layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Activation layer.

        Returns
        -------
        tf.keras.layers.Activation
            The created Activation layer.

        Examples
        --------
        >>> from vgslify.tensorflow.layers import TensorFlowLayerFactory
        >>> activation_layer = TensorFlowLayerFactory.activation("Ar")
        >>> print(activation_layer)
        <keras.src.layers.core.activation.Activation object at 0x7f8b1c0b1d30>
        """
        activation_function = parse_activation_spec(spec)
        return tf.keras.layers.Activation(activation=activation_function)

    @staticmethod
    def reshape(spec: str,
                prev_layer: tf.keras.layers.Layer = None) -> tf.keras.layers.Reshape:
        """
        Create a Reshape layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            VGSL specification string for the Reshape layer. Can be:
            - 'Rc': Collapse spatial dimensions (height and width).
            - 'R<x>,<y>,<z>': Reshape to the specified target shape.
        prev_layer : tf.keras.layers.Layer, optional
            The previous layer in the model, used for spatial collapsing, by default None

        Returns
        -------
        tf.keras.layers.Reshape
            The created Reshape layer.

        Examples
        --------
        >>> from vgslify.tensorflow.layers import TensorFlowLayerFactory
        >>> reshape_layer = TensorFlowLayerFactory.reshape("R10,10,3")
        >>> print(reshape_layer)
        <keras.src.layers.core.reshape.Reshape object at 0x7f8b1c0b1d30>
        """
        # Handle 'Rc' (collapse spatial dimensions) specification
        if spec == 'Rc':
            if prev_layer is None:
                raise ValueError(
                    "Previous layer is required for spatial collapsing. None provided.")
            prev_shape = prev_layer.shape  # Get shape of the previous layer
            if len(prev_shape) < 4:
                raise ValueError(
                    f"Previous layer shape {prev_shape} is incompatible for spatial collapsing. "
                    "Expected at least 4 dimensions."
                )
            # Get height and width dimensions
            height, width = prev_shape[-3] or 1, prev_shape[-2] or 1
            num_channels = prev_shape[-1]
            return tf.keras.layers.Reshape((-1, height * width * num_channels))

        # Handle regular reshape (e.g., 'R64,64,3')
        config = parse_reshape_spec(spec)
        return tf.keras.layers.Reshape(target_shape=config.target_shape)

    @staticmethod
    def input(spec: str) -> tf.keras.layers.Input:
        """
        Create an Input layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Input layer. Supported formats:
            - 1D: `<batch_size>,<width>`
            - 2D: `<batch_size>,<height>,<width>`
            - 3D: `<batch_size>,<height>,<width>,<channels>`
            - 4D: `<batch_size>,<depth>,<height>,<width>,<channels>`

        Returns
        -------
        tf.keras.layers.Input
            The created Input layer.

        Raises
        ------
        ValueError
            If the provided VGSL spec string does not match the expected format.

        Examples
        --------
        >>> from vgslify.tensorflow.layers import TensorFlowLayerFactory
        >>> input_layer = TensorFlowLayerFactory.input("None,32,3,32")
        >>> print(input_layer)
        <keras.src.layers.core.input.Input object at 0x7f8b1c0b1d30>
        >>> input_layer = TensorFlowLayerFactory.input("None,128")
        >>> print(input_layer)
        <keras.src.layers.core.input.Input object at 0x7f8b1c0b1d30>
        >>> input_layer = TensorFlowLayerFactory.input("None,16,64,64,3")
        >>> print(input_layer)
        <keras.src.layers.core.input.Input object at 0x7f8b1c0b1d30>
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

        return tf.keras.Input(shape=input_shape, batch_size=config.batch_size)

    @staticmethod
    def flatten(spec: str) -> tf.keras.layers.Flatten:
        """
        Create a Flatten layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Flatten layer. Expected format: 'Flt'.

        Returns
        -------
        tf.keras.layers.Flatten
            The created Flatten layer.

        Raises
        ------
        ValueError
            If the provided VGSL spec string does not match the expected format.

        Examples
        --------
        >>> from vgslify.tensorflow.layers import TensorFlowLayerFactory
        >>> flatten_layer = TensorFlowLayerFactory.flatten("Flt")
        >>> print(flatten_layer)
        <keras.src.layers.core.flatten.Flatten object at 0x7f8b1c0b1d30>
        """
        if spec != "Flt":
            raise ValueError(
                f"Flatten layer spec '{spec}' is incorrect. Expected 'Flt'.")

        return tf.keras.layers.Flatten()

    @staticmethod
    def build_final_model(inputs: tf.keras.layers.Input, outputs: tf.keras.layers.Layer,
                          name: str = "VGSL_Model") \
            -> tf.keras.models.Model:
        """
        Build the final model using the provided input and output layers.

        Parameters
        ----------
        inputs : tf.keras.layers.Input
            The input layer of the model.
        outputs : tf.keras.layers.Layer
            The output layer (or the final processed layer before output) of the model.
        name : str, optional
            The name of the model, by default "VGSL_Model

        Returns
        -------
        tf.keras.models.Model
            The constructed TensorFlow model.

        Examples
        --------
        >>> from vgslify.tensorflow.layers import TensorFlowLayerFactory
        >>> inputs = TensorFlowLayerFactory.input("None,32,3,32")
        >>> outputs = TensorFlowLayerFactory.dense("Fr64")(inputs)
        >>> model = TensorFlowLayerFactory.build_final_model(inputs, outputs)
        >>> print(model)
        <keras.engine.training.Model object at 0x7f8b1c0b1d30>
        """
        model = tf.keras.models.Model(
            inputs=inputs, outputs=outputs, name=name)

        return model
