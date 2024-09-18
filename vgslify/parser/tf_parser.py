import tensorflow as tf


def tf_to_spec(model: tf.keras.models.Model) -> str:
    """
    Convert a Keras model to a VGSL spec string.

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

    # Map activation functions to VGSL codes
    ACTIVATION_MAP = {
        'softmax': 's', 'tanh': 't', 'relu': 'r',
        'elu': 'e', 'linear': 'l', 'sigmoid': 'm'
    }

    # Helper functions for constructing spec parts
    def get_activation(layer) -> str:
        activation = layer.get_config().get("activation", "linear")
        return ACTIVATION_MAP.get(activation, None)

    def get_dropout_spec(layer) -> str:
        dropout = getattr(layer, 'dropout', 0)
        recurrent_dropout = getattr(layer, 'recurrent_dropout', 0)
        return (f",D{int(dropout*100)}" if dropout > 0 else "") + \
               (f",Rd{int(recurrent_dropout*100)}" if recurrent_dropout > 0 else "")

    def get_stride_spec(layer) -> str:
        strides = getattr(layer, 'strides', (1, 1))
        return f",{strides[0]},{strides[1]}" if strides != (1, 1) else ""

    # Layer parsing functions
    def parse_input_layer(layer):
        input_shape = layer.output.shape if isinstance(
            layer, tf.keras.layers.InputLayer) else layer.input_shape
        layer_str = ",".join([str(dim) for dim in input_shape])
        return layer_str

    def parse_conv2d(layer):
        act = get_activation(layer)
        return f"C{act}{layer.kernel_size[0]},{layer.kernel_size[1]},{layer.filters}" \
            f"{get_stride_spec(layer)}"

    def parse_dense(layer, is_output=False):
        act = get_activation(layer)
        prefix = "O1" if is_output else "F"
        return f"{prefix}{act}{layer.units}"

    def parse_rnn(layer, rnn_type):
        direction = 'r' if layer.go_backwards else 'f'
        return_sequences = 's' if layer.return_sequences else ''
        return f"{rnn_type}{direction}{return_sequences}{layer.units}{get_dropout_spec(layer)}"

    def parse_bidirectional(layer):
        wrapped_layer = layer.forward_layer
        cell_type = 'l' if isinstance(
            wrapped_layer, tf.keras.layers.LSTM) else 'g'
        return f"B{cell_type}{wrapped_layer.units}{get_dropout_spec(wrapped_layer)}"

    def parse_pooling(layer, pool_type):
        return f"{pool_type}{layer.pool_size[0]},{layer.pool_size[1]},{layer.strides[0]}," \
            f"{layer.strides[1]}"

    def parse_batchnorm(_):
        return "Bn"

    def parse_dropout(layer):
        return f"D{int(layer.rate * 100)}"

    def parse_reshape(layer, prev_layer):
        # Get the previous layer shape
        prev_shape = prev_layer.output.shape

        # Replace None dimensions with 1
        prev_shape = tuple(
            dim if dim is not None else 1 for dim in prev_shape)

        # Get the target shape from the Reshape layer
        target_shape = layer.target_shape

        # Replace None dimensions with 1
        target_shape = tuple(
            dim if dim is not None else 1 for dim in target_shape)

        print(prev_shape, target_shape)

        # Determine if this is a spatial collapse
        if (len(prev_shape) == 4
                and target_shape == (-1, prev_shape[-3] * prev_shape[-2] * prev_shape[-1])):
            # This is a spatial collapse
            return "Rc"

        # This is a general reshape, format it as Rx,y,z
        reshape_dims = ",".join([str(dim) for dim in target_shape])
        return f"R{reshape_dims}"

    # Mapping layer types to their corresponding parsing functions
    LAYER_PARSERS = {
        tf.keras.layers.InputLayer: parse_input_layer,
        tf.keras.layers.Conv2D: parse_conv2d,
        tf.keras.layers.Dense: parse_dense,
        tf.keras.layers.LSTM: lambda l: parse_rnn(l, "L"),
        tf.keras.layers.GRU: lambda l: parse_rnn(l, "G"),
        tf.keras.layers.Bidirectional: parse_bidirectional,
        tf.keras.layers.MaxPooling2D: lambda l: parse_pooling(l, "Mp"),
        tf.keras.layers.AveragePooling2D: lambda l: parse_pooling(l, "Ap"),
        tf.keras.layers.BatchNormalization: parse_batchnorm,
        tf.keras.layers.Dropout: parse_dropout,
        tf.keras.layers.Reshape: parse_reshape,
        tf.keras.layers.Activation: lambda l: None
    }

    # Parse the model
    vgsl_parts = []
    if type(model.layers[0]) != tf.keras.layers.InputLayer:
        input_layer = tf.keras.layers.InputLayer(input_shape=model.input_shape[1:],
                                                 batch_size=model.input_shape[0])
        vgsl_parts.append(parse_input_layer(input_layer))

    for idx, layer in enumerate(model.layers):
        layer_type = type(layer)
        if layer_type in LAYER_PARSERS:
            # Call the corresponding parser function
            prev_layer = model.layers[idx - 1] if idx > 0 else None
            spec = LAYER_PARSERS[layer_type](layer, prev_layer) \
                if layer_type == tf.keras.layers.Reshape else LAYER_PARSERS[layer_type](layer)
            if spec:
                is_output_layer = isinstance(
                    layer, tf.keras.layers.Dense) and idx >= len(model.layers) - 1
                if is_output_layer:
                    spec = parse_dense(layer, is_output=True)
                vgsl_parts.append(spec)
        else:
            raise ValueError(
                f"Unsupported layer type {layer_type.__name__} at position {idx}.")

    # Return the VGSL string
    return " ".join(vgsl_parts)
