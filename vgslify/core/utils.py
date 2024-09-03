def get_activation_function(activation_char: str) -> str:
    """
    Maps a VGSL activation character to the corresponding Keras activation function.

    Parameters
    ----------
    activation_char : str
        The character representing the activation function in the VGSL spec.

    Returns
    -------
    str
        The name of the Keras activation function.

    Raises
    ------
    ValueError
        If the provided activation character is not recognized.

    Examples
    --------
    >>> activation = get_activation_function('r')
    >>> print(activation)
    'relu'
    """
    activation_map = {
        's': 'softmax',
        't': 'tanh',
        'r': 'relu',
        'l': 'linear',
        'm': 'sigmoid'
    }

    if activation_char not in activation_map:
        raise ValueError(f"Invalid activation character '{activation_char}'.")

    return activation_map[activation_char]
