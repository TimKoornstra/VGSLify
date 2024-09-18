import re
from vgslify.core.config import (Conv2DConfig, Pooling2DConfig, DenseConfig,
                                 RNNConfig, DropoutConfig, OutputLayerConfig,
                                 ReshapeConfig, InputConfig)
from vgslify.core.utils import get_activation_function


def parse_spec(model_spec: str) -> list:
    """
    Parse the full model spec string into a list of individual layer specs.

    Parameters
    ----------
    model_spec : str
        The VGSL specification string defining the model architecture.

    Returns
    -------
    list
        A list of layer specification strings.
    """
    return model_spec.split()


def parse_conv2d_spec(spec: str) -> Conv2DConfig:
    """
    Parses a VGSL specification string for a Conv2D layer and returns the parsed configuration.

    Parameters
    ----------
    spec : str
        VGSL specification for the convolutional layer. Expected format:
        `C(s|t|r|l|m)<x>,<y>,[<s_x>,<s_y>,]<d>`
        - (s|t|r|l|m): Activation type.
        - <x>,<y>: Kernel size.
        - <s_x>,<s_y>: Optional strides (defaults to (1, 1) if not provided).
        - <d>: Number of filters (depth).

    Returns
    -------
    Conv2DConfig
        Parsed configuration for the Conv2D layer.

    Raises
    ------
    ValueError:
        If the provided VGSL spec string does not match the expected format.

    Examples
    --------
    >>> from vgslify.core.parser import parse_conv2d_spec
    >>> config = parse_conv2d_spec("Cr3,3,64")
    >>> print(config)
    Conv2DConfig(activation='relu', kernel_size=(3, 3), strides=(1, 1), filters=64)
    """

    # Extract convolutional parameters
    conv_filter_params = [int(match) for match in re.findall(r'\d+', spec)]

    # Check if the layer format is as expected
    if len(conv_filter_params) < 3:
        raise ValueError(f"Conv layer {spec} has too few parameters. "
                         "Expected format: C(s|t|r|l|m)<x>,<y>,<d> or "
                         "C(s|t|r|l|m)<x>,<y>,<s_x>,<s_y>,<d>")
    if len(conv_filter_params) > 5:
        raise ValueError(f"Conv layer {spec} has too many parameters. "
                         "Expected format: C(s|t|r|l|m)<x>,<y>,<d> or "
                         "C(s|t|r|l|m)<x>,<y>,<s_x>,<s_y>,<d>")

    # Extract activation function
    try:
        activation = get_activation_function(spec[1])
    except ValueError:
        activation = None  # Fall back to default activation

    # Check parameter length and assign kernel size, strides, and filters
    if len(conv_filter_params) == 3:
        x, y, d = conv_filter_params
        strides = (1, 1)  # Default stride
    elif len(conv_filter_params) == 5:
        x, y, s_x, s_y, d = conv_filter_params
        strides = (s_x, s_y)
    else:
        raise ValueError(f"Invalid number of parameters in {spec}")

    kernel_size = (y, x)

    # Return the parsed configuration
    return Conv2DConfig(
        activation=activation,
        kernel_size=kernel_size,
        strides=strides,
        filters=d
    )


def parse_pooling2d_spec(spec: str) -> Pooling2DConfig:
    """
    Parses a VGSL specification string for a Pooling2D layer and returns the parsed configuration.

    Parameters
    ----------
    spec : str
        VGSL specification for the pooling layer. Expected format:
        `Mp<x>,<y>,<s_x>,<s_y>` or `Ap<x>,<y>,<s_x>,<s_y>`
        - <x>,<y>: Pool size.
        - <s_x>,<s_y>: Strides.

    Returns
    -------
    Pooling2DConfig
        Parsed configuration for the Pooling2D layer.

    Raises
    ------
    ValueError:
        If the provided VGSL spec string does not match the expected format.

    Examples
    --------
    >>> config = parse_pooling2d_spec("Mp2,2,2,2")
    >>> print(config)
    Pooling2DConfig(pool_size=(2, 2), strides=(2, 2))
    """

    # Extract pooling and stride parameters
    pool_stride_params = [int(match) for match in re.findall(r'-?\d+', spec)]

    # Check if the parameters are as expected
    if len(pool_stride_params) != 4:
        raise ValueError(f"Pooling layer {spec} does not have the expected number of parameters. "
                         "Expected format: <p><x>,<y>,<stride_x>,<stride_y>")

    pool_x, pool_y, stride_x, stride_y = pool_stride_params

    # Check if pool and stride values are valid
    if pool_x <= 0 or pool_y <= 0 or stride_x <= 0 or stride_y <= 0:
        raise ValueError(f"Invalid values for pooling or stride in {spec}. "
                         "All values should be positive integers.")

    return Pooling2DConfig(pool_size=(pool_x, pool_y), strides=(stride_x, stride_y))


def parse_dense_spec(spec: str) -> DenseConfig:
    """
    Parses a VGSL specification string for a Dense layer and returns the parsed configuration.

    Parameters
    ----------
    spec : str
        VGSL specification for the dense layer. Expected format: `F(s|t|r|l|m)<d>`
        - `(s|t|r|l|m)`: Non-linearity type. One of sigmoid, tanh, relu,
          linear, or softmax.
        - `<d>`: Number of outputs (units).

    Returns
    -------
    DenseConfig
        Parsed configuration for the Dense layer.

    Raises
    ------
    ValueError
        If the provided VGSL spec string does not match the expected format.

    Examples
    --------
    >>> config = parse_dense_spec("Fr64")
    >>> print(config)
    DenseConfig(activation='relu', units=64)
    """

    # Ensure the layer string format is as expected
    if not re.match(r'^F[a-z]-?\d+$', spec):
        raise ValueError(
            f"Dense layer {spec} is of unexpected format. Expected format: F(s|t|r|l|m)<d>."
        )

    # Extract the activation function
    try:
        activation = get_activation_function(spec[1])
    except ValueError as e:
        raise ValueError(
            f"Invalid activation function '{spec[1]}' for Dense layer {spec}. "
            "Expected one of 's', 't', 'r', 'l', or 'm'.") from e

    # Extract the number of neurons (units)
    units = int(spec[2:])
    if units <= 0:
        raise ValueError(
            f"Invalid number of neurons {units} for Dense layer {spec}.")

    # Return the parsed configuration
    return DenseConfig(
        activation=activation,
        units=units
    )


def parse_rnn_spec(spec: str) -> RNNConfig:
    """
    Parses a VGSL specification string for an RNN layer (LSTM, GRU, Bidirectional)
    and returns the parsed configuration.

    Parameters
    ----------
    spec : str
        VGSL specification for the RNN layer. Expected format:
        For LSTM/GRU: `L(f|r)[s]<n>[,D<rate>,Rd<rate>]`
        For Bidirectional: `B(g|l)<n>[,D<rate>,Rd<rate>]`

    Returns
    -------
    RNNConfig
        Parsed configuration for the RNN layer.

    Raises
    ------
    ValueError
        If the provided VGSL spec string does not match the expected format.

    Examples
    --------
    >>> config = parse_rnn_spec("Lf64,D50,Rd25")
    >>> print(config)
    RNNConfig(units=64, return_sequences=True, go_backwards=False, dropout=0.5,
              recurrent_dropout=0.25)
    """

    match = re.match(
        r'([LGB])([frgl])(s?)(-?\d+),?(D-?\d+)?,?(Rd-?\d+)?$', spec)
    if not match:
        raise ValueError(
            f"RNN layer {spec} is of unexpected format. Expected format: "
            "L(f|r)[s]<n>[,D<rate>,Rd<rate>], G(f|r)[s]<n>[,D<rate>,Rd<rate>], "
            "or B(g|l)<n>[,D<rate>,Rd<rate>]."
        )

    layer_type, rnn_type, summarize, units, dropout, recurrent_dropout = match.groups()

    units = int(units)
    dropout = 0 if dropout is None else int(dropout.replace('D', "")) / 100
    recurrent_dropout = 0 if recurrent_dropout is None else int(
        recurrent_dropout.replace("Rd", "")) / 100

    # Validation
    if units <= 0:
        raise ValueError(
            f"Invalid number of units {units} for RNN layer {spec}.")
    if dropout < 0 or dropout > 1:
        raise ValueError("Dropout rate must be between 0 and 1.")
    if recurrent_dropout < 0 or recurrent_dropout > 1:
        raise ValueError("Recurrent dropout rate must be between 0 and 1.")

    # Return RNNConfig with parsed parameters
    return RNNConfig(
        units=units,
        return_sequences=bool(summarize) if layer_type == 'L' else True,
        go_backwards=rnn_type == 'r',
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        rnn_type=rnn_type
    )


def parse_dropout_spec(spec: str) -> DropoutConfig:
    """
    Parses a VGSL specification string for a Dropout layer and returns the parsed configuration.

    Parameters
    ----------
    spec : str
        VGSL specification for the Dropout layer. Expected format:
        `D<rate>` where <rate> is the dropout percentage (0-100).

    Returns
    -------
    DropoutConfig
        Parsed configuration for the Dropout layer.

    Raises
    ------
    ValueError
        If the provided VGSL spec string does not match the expected format.

    Examples
    --------
    >>> config = parse_dropout_spec("D50")
    >>> print(config)
    DropoutConfig(rate=0.5)
    """

    match = re.match(r'D(-?\d+)$', spec)
    if not match:
        raise ValueError(
            f"Dropout layer {spec} is of unexpected format. Expected format: D<rate>."
        )

    dropout_rate = int(match.group(1))

    if dropout_rate < 0 or dropout_rate > 100:
        raise ValueError("Dropout rate must be in the range [0, 100].")

    return DropoutConfig(rate=dropout_rate / 100)


def parse_output_layer_spec(spec: str) -> OutputLayerConfig:
    """
    Parses a VGSL specification string for an Output layer and returns the parsed configuration.

    Parameters
    ----------
    spec : str
        VGSL specification for the Output layer. Expected format:
        `O(2|1|0)(l|s)<n>`
        - `(2|1|0)`: Dimensionality of the output.
        - `(l|s)`: Non-linearity type.
        - `<n>`: Number of output classes.

    Returns
    -------
    OutputLayerConfig
        Parsed configuration for the Output layer.

    Raises
    ------
    ValueError
        If the provided VGSL spec string does not match the expected format.

    Examples
    --------
    >>> config = parse_output_layer_spec("O1s10")
    >>> print(config)
    OutputLayerConfig(dimensionality=1, activation='softmax', units=10)
    """

    match = re.match(r'O([210])([a-z])(\d+)$', spec)
    if not match:
        raise ValueError(
            f"Output layer {spec} is of unexpected format. Expected format: O[210](l|s)<n>."
        )

    dimensionality, activation_char, units = match.groups()
    units = int(units)

    activation = get_activation_function(activation_char)

    return OutputLayerConfig(
        dimensionality=int(dimensionality),
        activation=activation,
        units=units
    )


def parse_activation_spec(spec: str) -> str:
    """
    Parses a VGSL specification string for an Activation layer and returns the activation function.

    Parameters
    ----------
    spec : str
        VGSL specification for the Activation layer. Expected format: `A(s|t|r|l|m)`
        - `s`: softmax
        - `t`: tanh
        - `r`: relu
        - `l`: linear
        - `m`: sigmoid

    Returns
    -------
    str
        The activation function name.

    Raises
    ------
    ValueError
        If the provided VGSL spec string does not match the expected format.

    Examples
    --------
    >>> activation = parse_activation_spec("Ar")
    >>> print(activation)
    'relu'
    """

    match = re.match(r'A([strlm])$', spec)
    if not match:
        raise ValueError(
            f"Activation layer spec '{spec}' is incorrect. Expected format: A(s|t|r|l|m).")

    return get_activation_function(match.group(1))


def parse_reshape_spec(spec: str) -> ReshapeConfig:
    """
    Parses a VGSL specification string for a Reshape layer and returns the target shape.

    Parameters
    ----------
    spec : str
        VGSL specification for the Reshape layer. Expected format: `R<x>,<y>,<z>`

    Returns
    -------
    ReshapeConfig
        Parsed configuration for the Reshape layer.

    Raises
    ------
    ValueError
        If the provided VGSL spec string does not match the expected format.

    Examples
    --------
    >>> config = parse_reshape_spec("R64,64,3")
    >>> print(config)
    ReshapeConfig(target_shape=(64, 64, 3))
    """

    match = re.match(r'R(-?\d+),(-?\d+)(?:,(-?\d+))?$', spec)
    if not match:
        raise ValueError(
            f"Reshape layer spec '{spec}' is incorrect. Expected format: R<x>,<y>[,<z>].")

    target_shape = tuple(int(x) for x in match.groups() if x)
    return ReshapeConfig(target_shape=target_shape)


def parse_input_spec(spec: str) -> InputConfig:
    """
    Parses a VGSL specification string for an Input layer and returns the parsed configuration.

    Parameters
    ----------
    spec : str
        VGSL specification for the Input layer. Supported format: 
        `<batch_size>,<depth>,<height>,<width>,<channels>` for 4D inputs,
        `<batch_size>,<height>,<width>,<channels>` for 3D inputs,
        `<batch_size>,<height>,<width>` for 2D inputs,
        `<batch_size>,<width>` for 1D inputs.

    Returns
    -------
    InputConfig
        Parsed configuration for the Input layer.

    Raises
    ------
    ValueError
        If the provided VGSL spec string does not match the expected format.
    """
    try:
        dims = spec.split(",")
        if len(dims) == 5:
            batch, depth, height, width, channels = dims
        elif len(dims) == 4:
            batch, height, width, channels = dims
            depth = None
        elif len(dims) == 3:
            batch, height, width = dims
            depth, channels = None, None
        elif len(dims) == 2:
            batch, width = dims
            height, depth, channels = None, None, None
        else:
            raise ValueError(f"Invalid input spec: {spec}")

        return InputConfig(
            batch_size=None if batch == "None" else int(batch),
            width=None if width == "None" else int(width),
            depth=None if depth == "None" else int(depth) if depth else None,
            height=None if height == "None" else int(
                height) if height else None,
            channels=None if channels == "None" else int(
                channels) if channels else None
        )
    except ValueError as e:
        raise ValueError(
            f"Invalid input string format '{spec}'. Expected valid VGSL format.") from e
