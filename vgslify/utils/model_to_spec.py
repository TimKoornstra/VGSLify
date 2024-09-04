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
    >>> from vgslify.utils import model_to_spec
    >>> import tensorflow as tf
    >>> model = tf.keras.models.load_model("path_to_model.h5")
    >>> spec_string = model_to_spec(model)
    >>> print(spec_string)
    """

    # Check if it's a TensorFlow model
    if tf and isinstance(model, tf.keras.Model):
        from vgslify.parser.tf_parser import tf_to_spec
        return tf_to_spec(model)

    # Check if it's a PyTorch model
    if nn and isinstance(model, nn.Module):
        raise NotImplementedError("PyTorch models are not supported yet.")

    # Raise an error if the model is not recognized
    raise ValueError(
        f"Unsupported model type: {type(model).__name__}. Expected TensorFlow "
        "or PyTorch model.")
