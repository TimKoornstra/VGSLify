try:
    import tensorflow as tf
except ImportError:
    tf = None
try:
    from torch import nn
except ImportError:
    nn = None


def model_to_spec(model) -> str:
    """
    Convert a deep learning model (TensorFlow or PyTorch) to a VGSL spec string.

    Parameters
    ----------
    model : Model
        The deep learning model to be converted. Can be a TensorFlow model (tf.keras.models.Model)
        or a PyTorch model (torch.nn.Module).

    Returns
    -------
    str
        VGSL spec string.

    Raises
    ------
    ValueError
        If the model is not supported or cannot be parsed.

    Examples
    --------
    >>> from vgslify import model_to_spec
    >>> import tensorflow as tf
    >>> model = tf.keras.models.load_model("path_to_model.h5")
    >>> spec_string = model_to_spec(model)
    >>> print(spec_string)
    """

    # Check if it's a TensorFlow model
    if tf and isinstance(model, tf.keras.Model):
        from vgslify.model_parsers import TensorFlowModelParser

        parser = TensorFlowModelParser()

    # Check if it's a PyTorch model
    if nn and isinstance(model, nn.Module):
        from vgslify.model_parsers import TorchModelParser

        parser = TorchModelParser()

    # Raise an error if the model is not recognized
    if not parser:
        raise ValueError(
            f"Unsupported model type: {type(model).__name__}. Expected TensorFlow "
            "or PyTorch model."
        )

    return parser.parse_model(model)
