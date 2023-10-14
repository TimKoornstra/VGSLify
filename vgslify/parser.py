import tensorflow as tf


def parse_tf_model(model: tf.keras.models.Model) -> str:
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
        If the model contains unsupported layers or if the input shape is
        invalid.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from vgslify.parser import parse_tf_model
    >>> model = tf.keras.models.load_model("model.h5")
    >>> parse_tf_model(model)
    """

    def get_dropout(dropout: float, recurrent_dropout: int = 0) -> str:
        """Helper function to generate dropout specifications."""

        dropout_spec = f",D{int(dropout*100)}" if dropout > 0 else ""
        recurrent_dropout_spec = f",Rd{int(recurrent_dropout*100)}" \
            if recurrent_dropout > 0 else ""

        return f"{dropout_spec}{recurrent_dropout_spec}"

    def get_stride_spec(strides: tuple) -> str:
        """Helper function to generate stride specifications."""

        return f",{strides[0]},{strides[1]}" if strides != (1, 1) else ""

    vgsl_parts = []
    activation_map = {'softmax': 's', 'tanh': 't', 'relu': 'r',
                      'elu': 'e', 'linear': 'l', 'sigmoid': 'm'}

    # Map Input layer
    # If the first layer is an InputLayer, get the input shape from the
    # second layer
    # This is only the case where we have a model created with the Keras
    # functional API
    if isinstance(model.layers[0], tf.keras.layers.InputLayer):
        input_shape = model.layers[0].input_shape[0]
        start_idx = 1
    else:
        input_shape = model.layers[0].input_shape
        start_idx = 0

    if not (len(input_shape) == 4 and
            all(isinstance(dim, (int, type(None)))
                for dim in input_shape)):
        raise ValueError(f"Invalid input shape {input_shape}. Input shape "
                         "must be of the form (None, height, width, "
                         "channels).")

    vgsl_parts.append(
        f"{input_shape[0]},{input_shape[2]},{input_shape[1]},"
        f"{input_shape[3]}")

    # Loop through and map the rest of the layers
    for idx in range(start_idx, len(model.layers)):
        layer = model.layers[idx]
        if isinstance(layer, tf.keras.layers.Conv2D):
            act = activation_map[layer.get_config()["activation"]]
            if act is None:
                raise ValueError(
                    "Unsupported activation function "
                    f"{layer.get_config()['activation']} in layer "
                    f"{type(layer).__name__} at position {idx}.")

            vgsl_parts.append(
                f"C{act}{layer.kernel_size[0]},{layer.kernel_size[1]},"
                f"{layer.filters}{get_stride_spec(layer.strides)}")

        elif isinstance(layer, tf.keras.layers.Dense):
            act = activation_map[layer.get_config()["activation"]]
            if act is None:
                raise ValueError(
                    "Unsupported activation function "
                    f"{layer.get_config()['activation']} in layer "
                    f"{type(layer).__name__} at position {idx}.")
            prefix = "O1" if idx == len(model.layers) - 1 or isinstance(
                model.layers[idx + 1], tf.keras.layers.Activation) else "F"

            vgsl_parts.append(f"{prefix}{act}{layer.units}")

        elif isinstance(layer, (tf.keras.layers.LSTM,
                                tf.keras.layers.GRU)):
            act = 'L' if isinstance(layer, tf.keras.layers.LSTM) else 'G'
            direction = 'r' if layer.go_backwards else 'f'
            return_sequences = "s" if layer.return_sequences else ""

            vgsl_parts.append(
                f"{act}{direction}{return_sequences}{layer.units}"
                f"{get_dropout(layer.dropout, layer.recurrent_dropout)}")

        elif isinstance(layer, tf.keras.layers.Bidirectional):
            wrapped_layer = layer.layer
            cell_type = 'l' if isinstance(
                wrapped_layer, tf.keras.layers.LSTM) else 'g'
            dropout = get_dropout(wrapped_layer.dropout,
                                  wrapped_layer.recurrent_dropout)

            vgsl_parts.append(
                f"B{cell_type}{wrapped_layer.units}{dropout}")

        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            vgsl_parts.append("Bn")

        elif isinstance(layer, tf.keras.layers.MaxPooling2D):
            vgsl_parts.append(
                f"Mp{layer.pool_size[0]},{layer.pool_size[1]},"
                f"{layer.strides[0]},{layer.strides[1]}")

        elif isinstance(layer, tf.keras.layers.AveragePooling2D):
            vgsl_parts.append(
                f"Ap{layer.pool_size[0]},{layer.pool_size[1]},"
                f"{layer.strides[0]},{layer.strides[1]}")

        elif isinstance(layer, tf.keras.layers.Dropout):
            vgsl_parts.append(f"D{int(layer.rate*100)}")

        elif isinstance(layer, tf.keras.layers.Reshape):
            vgsl_parts.append("Rc")

        elif isinstance(layer, tf.keras.layers.Activation):
            # Activation layers are not included in the VGSL spec
            # but is handled in the output layer
            continue

        else:
            # If an unsupported layer type is encountered
            raise ValueError(
                f"Unsupported layer type {type(layer).__name__} at "
                f"position {idx}.")

    return " ".join(vgsl_parts)
