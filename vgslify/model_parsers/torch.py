# Imports

# > Standard Library
import inspect
import warnings
from typing import Callable, Dict, Type, Union

# > Third-Party Dependencies
from torch import nn

# > Internal
from vgslify.core.config import (
    ActivationConfig,
    Conv2DConfig,
    DenseConfig,
    DropoutConfig,
    InputConfig,
    Pooling2DConfig,
    ReshapeConfig,
    RNNConfig,
)
from vgslify.model_parsers.base import BaseModelParser
from vgslify.torch.layers import Reshape


class TorchModelParser(BaseModelParser):
    """
    Parser for converting PyTorch models into VGSL (Variable-size Graph Specification Language) spec strings.

    This class extends the BaseModelParser to provide specific functionality for PyTorch models.
    It uses configuration dataclasses to represent different layer types and converts them into
    VGSL spec strings.

    Attributes
    ----------
    layer_parsers : Dict[Type[nn.Module], Callable]
        A dictionary mapping PyTorch layer types to their corresponding parsing methods.

    Notes
    -----
    This parser supports a wide range of PyTorch layers and can be extended to support
    additional layer types by adding new parsing methods and updating the layer_parsers dictionary.
    """

    # A class-level dictionary: {layer_class -> parser_function}
    _custom_layer_parsers: Dict[Type[nn.Module], Callable] = {}

    @classmethod
    def register(cls, layer_cls: Type[nn.Module], parser_fn: Callable):
        """
        Registers a custom parser function for a given TF layer class.

        Parameters
        ----------
        layer_cls : Type[tf.keras.layers.Layer]
            The TF layer class this parser function can handle.
        parser_fn : Callable
            A function with signature parser_fn(layer) -> str
            that returns a VGSL spec string for the given layer.
        """
        if layer_cls in cls._custom_layer_parsers:
            raise ValueError(
                f"A parser is already registered for {layer_cls.__name__}."
            )

        # Check signature to ensure `parser_fn` is (layer) -> str
        sig = inspect.signature(parser_fn)
        params = list(sig.parameters.values())
        if len(params) != 1:
            raise ValueError(
                "Custom parser function must define exactly one parameter: (layer)."
            )

        cls._custom_layer_parsers[layer_cls] = parser_fn

    @classmethod
    def get_custom_parsers(cls):
        """Return the dict of all registered custom parser functions."""
        return cls._custom_layer_parsers

    def __init__(self):
        # Initialize the layer parsers mapping
        self.layer_parsers: Dict[Type[nn.Module], Callable] = {
            nn.Conv2d: self.parse_conv2d,
            nn.Linear: self.parse_dense,
            nn.LSTM: self.parse_rnn,
            nn.GRU: self.parse_rnn,
            nn.MaxPool2d: self.parse_pooling,
            nn.AvgPool2d: self.parse_pooling,
            nn.BatchNorm2d: self.parse_batchnorm,
            nn.Dropout: self.parse_dropout,
            nn.Flatten: self.parse_flatten,
            nn.ReLU: self.parse_activation,
            nn.Sigmoid: self.parse_activation,
            nn.Tanh: self.parse_activation,
            nn.Identity: self.parse_activation,
            nn.Softmax: self.parse_activation,
            Reshape: self.parse_reshape,
        }

        # Merge in any custom user-registered parsers from the class-level registry
        for layer_cls, parse_fn in self.get_custom_parsers().items():
            self.layer_parsers[layer_cls] = parse_fn

    def parse_model(self, model: nn.Module) -> str:
        """
        Parse a PyTorch model into a VGSL spec string.

        Parameters
        ----------
        model : nn.Module
            PyTorch model to be converted.

        Returns
        -------
        str
            VGSL spec string.

        Raises
        ------
        ValueError
            If the model contains unsupported layers or if the input shape is invalid.
        """
        configs = []

        # Extract input shape from the first layer
        first_layer = next(model.children())
        input_config = self.parse_input(first_layer)
        if input_config:
            configs.append(input_config)

        # Iterate through all layers in the model
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Sequential):
                continue

            layer_type = type(layer)
            parser_func = self.layer_parsers.get(layer_type, None)

            if parser_func:
                # Parse the layer
                config = parser_func(layer)
                if isinstance(config, ReshapeConfig) or config == "Flt":
                    warnings.warn(
                        "Warning: The model contains a Flatten or Reshape layer. This may cause VGSLify to "
                        "misinterpret the model's input shape. It is recommended to manually verify and "
                        "adjust the input shape if necessary to ensure accuracy."
                    )

                # Append the config if not None
                if config:
                    configs.append(config)
            else:
                raise ValueError(
                    f"Unsupported layer type {layer_type.__name__} at {name}."
                )

        # Generate VGSL spec string from configs
        return self.generate_vgsl(configs)

    def parse_input(self, layer: nn.Module) -> InputConfig:
        """
        Parse the input shape from the first layer of the model.

        Parameters
        ----------
        layer : nn.Module
            The first layer of the PyTorch model.

        Returns
        -------
        InputConfig
            The configuration for the input layer.

        Raises
        ------
        ValueError
            If the input shape cannot be determined.
        """
        batch_size = None  # Placeholder for dynamic batch size
        depth, height, width, channels = -1, -1, -1, -1

        if hasattr(layer, "in_channels"):
            # Conv2d, Conv3d, BatchNorm2d, etc.
            channels = layer.in_channels
        elif hasattr(layer, "in_features"):
            # Linear, LSTM, GRU, etc.
            channels = layer.in_features
        elif hasattr(layer, "input_size"):
            # Some RNN layers
            channels = layer.input_size
        elif hasattr(layer, "num_features"):
            # Some normalization layers
            channels = layer.num_features

        # Try to infer spatial dimensions if available
        if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d, nn.AvgPool2d)):
            height, width = None, None
        elif isinstance(layer, (nn.Conv3d, nn.BatchNorm3d, nn.MaxPool3d, nn.AvgPool3d)):
            depth, height, width = None, None, None
        elif isinstance(layer, (nn.Linear, nn.LSTM, nn.GRU)):
            depth, height, width = None, None, None

        if channels == -1:
            raise ValueError("Unable to determine input shape from the first layer.")

        return InputConfig(
            batch_size=batch_size,
            depth=depth,
            height=height,
            width=width,
            channels=channels,
        )

    # Parser methods for different layer types
    def parse_conv2d(self, layer: nn.Conv2d) -> Conv2DConfig:
        """
        Parse a Conv2d layer into a Conv2DConfig dataclass.

        Parameters
        ----------
        layer : nn.Conv2d
            The Conv2d layer to parse.

        Returns
        -------
        Conv2DConfig
            The configuration for the Conv2D layer.
        """
        return Conv2DConfig(
            activation="linear",  # PyTorch typically separates activation
            kernel_size=layer.kernel_size,
            strides=layer.stride,
            filters=layer.out_channels,
        )

    def parse_dense(self, layer: nn.Linear) -> DenseConfig:
        """
        Parse a Linear layer into a DenseConfig dataclass.

        Parameters
        ----------
        layer : nn.Linear
            The Linear layer to parse.

        Returns
        -------
        DenseConfig
            The configuration for the Dense layer.
        """
        return DenseConfig(
            activation="linear",  # PyTorch typically separates activation
            units=layer.out_features,
        )

    def parse_rnn(self, layer: Union[nn.LSTM, nn.GRU]) -> RNNConfig:
        """
        Parse an RNN layer (LSTM or GRU) into an RNNConfig dataclass.

        Parameters
        ----------
        layer : Union[nn.LSTM, nn.GRU]
            The RNN layer to parse.

        Returns
        -------
        RNNConfig
            The configuration for the RNN layer.
        """
        if isinstance(layer, nn.LSTM):
            rnn_type = "lstm"
        elif isinstance(layer, nn.GRU):
            rnn_type = "gru"
        else:
            raise ValueError(f"Unsupported RNN layer type {type(layer).__name__}.")

        return RNNConfig(
            units=layer.hidden_size,
            return_sequences=True,  # PyTorch RNNs always return sequences by default
            go_backwards=False,  # PyTorch doesn't have a direct equivalent
            dropout=layer.dropout,
            recurrent_dropout=0,  # PyTorch doesn't have recurrent dropout
            rnn_type=rnn_type,
            bidirectional=layer.bidirectional,
        )

    def parse_pooling(
        self, layer: Union[nn.MaxPool2d, nn.AvgPool2d]
    ) -> Pooling2DConfig:
        """
        Parse a Pooling layer into a Pooling2DConfig dataclass.

        Parameters
        ----------
        layer : nn.MaxPool2d or nn.AvgPool2d
            The Pooling layer to parse.

        Returns
        -------
        Pooling2DConfig
            The configuration for the Pooling layer.
        """
        if isinstance(layer, nn.MaxPool2d):
            pool_type = "max"
        elif isinstance(layer, nn.AvgPool2d):
            pool_type = "average"

        return Pooling2DConfig(
            pool_type=pool_type, pool_size=layer.kernel_size, strides=layer.stride
        )

    def parse_batchnorm(self, layer: nn.BatchNorm2d) -> str:
        """
        Parse a BatchNorm2d layer.

        Parameters
        ----------
        layer : nn.BatchNorm2d
            The BatchNorm2d layer to parse.

        Returns
        -------
        str
            Indicates that the VGSL spec should include 'Bn'.
        """
        return "Bn"

    def parse_dropout(self, layer: nn.Dropout) -> DropoutConfig:
        """
        Parse a Dropout layer into a DropoutConfig dataclass.

        Parameters
        ----------
        layer : nn.Dropout
            The Dropout layer to parse.

        Returns
        -------
        DropoutConfig
            The configuration for the Dropout layer.
        """
        return DropoutConfig(rate=layer.p)

    def parse_flatten(self, layer: nn.Flatten) -> str:
        """
        Parse a Flatten layer.

        Parameters
        ----------
        layer : nn.Flatten
            The Flatten layer to parse.

        Returns
        -------
        str
            Indicates that the VGSL spec should include 'Flatten'.
        """
        return "Flt"

    def parse_reshape(self, layer: Reshape) -> ReshapeConfig:
        """
        Parse a Reshape layer into a ReshapeConfig dataclass.

        Parameters
        ----------
        layer : Reshape
            The custom Reshape layer to parse.

        Returns
        -------
        ReshapeConfig
            The configuration for the Reshape layer.
        """
        target_shape = layer.target_shape
        return ReshapeConfig(target_shape=target_shape)

    def parse_activation(self, layer: nn.Module) -> ActivationConfig:
        """
        Parse an activation function.

        Parameters
        ----------
        layer : nn.Module
            The activation layer to parse.

        Returns
        -------
        ActivationConfig
            The configuration for the Activation layer.
        """
        if isinstance(layer, nn.ReLU):
            activation = "relu"
        elif isinstance(layer, nn.Sigmoid):
            activation = "sigmoid"
        elif isinstance(layer, nn.Tanh):
            activation = "tanh"
        elif isinstance(layer, nn.Identity):
            activation = "linear"
        elif isinstance(layer, nn.Softmax):
            activation = "softmax"
        else:
            activation = "linear"

        return ActivationConfig(activation=activation)


def register_custom_parser(layer_cls: Type[nn.Module]):
    """
    Decorator to register a custom parser function for a given PyTorch layer class.

    This allows users to easily extend the TorchModelParser with custom layer types
    by defining a function that converts a PyTorch layer into a VGSL specification.

    Parameters
    ----------
    layer_cls : Type[nn.Module]
        The PyTorch layer class to associate with the parser function.

    Returns
    -------
    Callable
        A decorator that registers the provided function as a parser for `layer_cls`.

    Raises
    ------
    ValueError
        If a parser for `layer_cls` is already registered or if the function does not
        accept exactly one argument (the layer instance).

    Examples
    --------
    Registering a custom parser for a `MyCustomLayer`:

    >>> from vgslify.model_parsers.torch import register_custom_parser
    >>> from torch import nn
    >>> class MyCustomLayer(nn.Module):
    ...     def __init__(self, param: int):
    ...         super().__init__()
    ...         self.param = param
    ...
    >>> @register_custom_parser(MyCustomLayer)
    ... def parse_my_custom_layer(layer: MyCustomLayer):
    ...     return f"MyCustomSpec({layer.param})"
    ...
    >>> # Now the parser is automatically registered inside TorchModelParser
    """

    def decorator(fn: Callable):
        TorchModelParser.register(layer_cls, fn)
        return fn

    return decorator
