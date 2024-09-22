# Imports

# > Third-party dependencies
import torch
import torch.nn as nn

# > Internal dependencies
from vgslify.core.factory import LayerFactory
from vgslify.core.parser import (parse_conv2d_spec, parse_pooling2d_spec,
                                 parse_dense_spec, parse_rnn_spec,
                                 parse_dropout_spec, parse_activation_spec,
                                 parse_reshape_spec, parse_input_spec)


class TorchLayerFactory(LayerFactory):
    """
    TorchLayerFactory is responsible for creating PyTorch-specific layers based on parsed
    VGSL (Variable-size Graph Specification Language) specifications. This factory handles the
    creation of various types of layers, including convolutional layers, pooling layers, RNN layers,
    dense layers, activation layers, and more.

    This class maintains an internal state to track the shape of the tensor as layers are added.
    """

    def __init__(self):
        super().__init__()

    def _get_activation_layer(self, activation_name: str):
        """
        Return a PyTorch activation layer based on the activation name.

        Parameters
        ----------
        activation_name : str
            Name of the activation function.

        Returns
        -------
        torch.nn.Module
            The activation layer.

        Raises
        ------
        ValueError
            If the activation_name is not recognized.
        """
        activations = {
            'softmax': nn.Softmax(dim=1),
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'linear': nn.Identity(),
            'sigmoid': nn.Sigmoid(),
        }
        if activation_name in activations:
            return activations[activation_name]
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")

    def conv2d(self, spec: str):
        """
        Create a Conv2D layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Conv2D layer.

        Returns
        -------
        torch.nn.Module
            The created Conv2D layer.

        Raises
        ------
        ValueError
            If the provided VGSL spec string does not match the expected format.

        Examples
        --------
        >>> from vgslify.torch.layers import TorchLayerFactory
        >>> factory = TorchLayerFactory()
        >>> factory.set_input_shape((3, 32, 32))
        >>> conv_layer = factory.conv2d("Cr3,3,64")
        >>> print(conv_layer)
        Sequential(
          (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=same)
          (1): ReLU()
        )
        """
        config = parse_conv2d_spec(spec)
        if self.shape is None:
            raise ValueError("Input shape must be set before adding layers.")

        in_channels = self.shape[0]  # Assuming channels-first

        # Check if padding='same' is supported (PyTorch >=1.7)
        padding = 'same' if torch.__version__ >= '1.7' else self._compute_same_padding(
            config.kernel_size, config.strides)

        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=config.filters,
            kernel_size=config.kernel_size,
            stride=config.strides,
            padding=padding
        )

        self.layers.append(conv_layer)

        # Handle activation
        if config.activation:
            activation_layer = self._get_activation_layer(config.activation)
            self.layers.append(activation_layer)

        # Update shape
        self.shape = self._compute_conv_output_shape(self.shape, config)
        return conv_layer, activation_layer or None

    def _compute_same_padding(self, kernel_size, stride):
        """
        Compute the padding size to achieve 'same' padding.

        Parameters
        ----------
        kernel_size : int or tuple
            Size of the kernel.
        stride : int or tuple
            Stride of the convolution.

        Returns
        -------
        tuple
            Padding size for height and width dimensions.
        """
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        padding = []
        for k, s in zip(kernel_size, stride):
            p = ((k - 1) // 2)
            padding.append(p)
        return tuple(padding)

    def _compute_conv_output_shape(self, input_shape, config):
        """
        Computes the output shape of a convolutional layer.

        Parameters
        ----------
        input_shape : tuple
            The input shape (C, H, W).
        config : Any
            The configuration object returned by parse_conv2d_spec.

        Returns
        -------
        tuple
            The output shape after the convolution.
        """
        C_in, H_in, W_in = input_shape
        C_out = config.filters

        # For 'same' padding
        H_out = int((H_in + config.strides[0] - 1) // config.strides[0])
        W_out = int((W_in + config.strides[1] - 1) // config.strides[1])

        return (C_out, H_out, W_out)

    def maxpooling2d(self, spec: str) -> nn.Module:
        """
        Create a MaxPooling2D layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the MaxPooling2D layer.

        Returns
        -------
        torch.nn.Module
            The created MaxPooling2D layer.
        """
        config = parse_pooling2d_spec(spec)
        padding = self._compute_same_padding(config.pool_size, config.strides)
        layer = nn.MaxPool2d(
            kernel_size=config.pool_size,
            stride=config.strides,
            padding=padding
        )
        self.layers.append(layer)
        # Update shape
        self.shape = self._compute_pool_output_shape(self.shape, config)
        return layer

    def avgpool2d(self, spec: str) -> nn.Module:
        """
        Create an AvgPool2D layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the AvgPool2D layer.

        Returns
        -------
        torch.nn.Module
            The created AvgPool2D layer.
        """
        config = parse_pooling2d_spec(spec)
        padding = self._compute_same_padding(config.pool_size, config.strides)
        layer = nn.AvgPool2d(
            kernel_size=config.pool_size,
            stride=config.strides,
            padding=padding
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
            The input shape (C, H, W).
        config : Any
            The configuration object returned by parse_pooling2d_spec.

        Returns
        -------
        tuple
            The output shape after pooling.
        """
        C_in, H_in, W_in = input_shape
        C_out = C_in

        H_out = int((H_in + config.strides[0] - 1) // config.strides[0])
        W_out = int((W_in + config.strides[1] - 1) // config.strides[1])

        return (C_out, H_out, W_out)

    def dense(self, spec: str) -> nn.Module:
        """
        Create a Dense layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Dense layer.

        Returns
        -------
        torch.nn.Module
            The created Dense layer.
        """
        config = parse_dense_spec(spec)
        if self.shape is None:
            raise ValueError("Input shape must be set before adding layers.")

        in_features = int(torch.prod(torch.tensor(self.shape)).item())
        linear_layer = nn.Linear(
            in_features=in_features,
            out_features=config.units,
        )

        self.layers.append(linear_layer)

        # Handle activation
        if config.activation:
            activation_layer = self._get_activation_layer(config.activation)
            self.layers.append(activation_layer)

        # Update shape
        self.shape = (config.units,)
        return linear_layer, activation_layer or None

    def lstm(self, spec: str):
        """
        Create an LSTM layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the LSTM layer.

        Returns
        -------
        torch.nn.LSTM
            The created LSTM layer.
        """
        config = parse_rnn_spec(spec)
        if self.shape is None:
            raise ValueError("Input shape must be set before adding layers.")

        input_size = self.shape[-1]
        layer = nn.LSTM(
            input_size=input_size,
            hidden_size=config.units,
            num_layers=1,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=False
        )
        self.layers.append(layer)

        # Update shape
        self.shape = (self.shape[0], config.units)
        return layer

    def gru(self, spec: str):
        """
        Create a GRU layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the GRU layer.

        Returns
        -------
        torch.nn.GRU
            The created GRU layer.
        """
        config = parse_rnn_spec(spec)
        if self.shape is None:
            raise ValueError("Input shape must be set before adding layers.")

        input_size = self.shape[-1]
        layer = nn.GRU(
            input_size=input_size,
            hidden_size=config.units,
            num_layers=1,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=False
        )
        self.layers.append(layer)

        # Update shape
        self.shape = (self.shape[0], config.units)
        return layer

    def bidirectional(self, spec: str) -> nn.Module:
        """
        Create a Bidirectional RNN layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Bidirectional layer.

        Returns
        -------
        torch.nn.Module
            The created Bidirectional RNN layer.
        """
        config = parse_rnn_spec(spec)
        if self.shape is None:
            raise ValueError("Input shape must be set before adding layers.")

        input_size = self.shape[-1]
        rnn_layer = nn.LSTM if config.rnn_type == 'l' else nn.GRU

        layer = rnn_layer(
            input_size=input_size,
            hidden_size=config.units,
            num_layers=1,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=True
        )
        self.layers.append(layer)

        # Update shape
        self.shape = (self.shape[0], config.units * 2)
        return layer

    def dropout(self, spec: str) -> nn.Dropout:
        """
        Create a Dropout layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Dropout layer.

        Returns
        -------
        torch.nn.Dropout
            The created Dropout layer.
        """
        config = parse_dropout_spec(spec)
        layer = nn.Dropout(p=config.rate)
        self.layers.append(layer)
        # Shape remains the same
        return layer

    def batchnorm(self, spec: str) -> nn.Module:
        """
        Create a BatchNormalization layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the BatchNormalization layer.

        Returns
        -------
        torch.nn.Module
            The created BatchNormalization layer.
        """
        if spec != 'Bn':
            raise ValueError(
                f"BatchNormalization layer spec '{spec}' is incorrect. Expected 'Bn'.")

        if self.shape is None:
            raise ValueError("Input shape must be set before adding layers.")

        num_features = self.shape[0]  # Assuming channels-first for Conv layers

        # Decide which BatchNorm layer to use based on the expected input dimensions
        if len(self.shape) == 3:
            layer = nn.BatchNorm2d(num_features)
        elif len(self.shape) == 2:
            layer = nn.BatchNorm1d(num_features)
        else:
            raise ValueError("Unsupported input shape for BatchNorm layer.")

        self.layers.append(layer)
        # Shape remains the same
        return layer

    def activation(self, spec: str) -> nn.Module:
        """
        Create an Activation layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Activation layer.

        Returns
        -------
        torch.nn.Module
            The created Activation layer.
        """
        activation_function = parse_activation_spec(spec)
        layer = self._get_activation_layer(activation_function)
        self.layers.append(layer)
        # Shape remains the same
        return layer

    def reshape(self, spec: str) -> nn.Module:
        """
        Create a Reshape layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            VGSL specification string for the Reshape layer. Can be:
            - 'Rc': Collapse spatial dimensions (height, width, and channels).
            - 'R<x>,<y>,<z>': Reshape to the specified target shape.

        Returns
        -------
        torch.nn.Module
            The created Reshape layer.
        """
        if self.shape is None:
            raise ValueError("Input shape must be set before adding layers.")

        # Handle 'Rc' (collapse spatial dimensions) specification
        if spec.startswith('Rc'):
            if spec == 'Rc2':
                # Flatten to (batch_size, -1)
                layer = nn.Flatten()
                self.layers.append(layer)
                self.shape = (
                    int(torch.prod(torch.tensor(self.shape)).item()),)
                return layer

            elif spec == 'Rc3':
                # Reshape to (batch_size, seq_length, features)
                C, H, W = self.shape
                seq_length = H * W
                features = C
                layer = self.Reshape(features, seq_length)
                self.layers.append(layer)
                self.shape = (seq_length, features)
                return layer

            else:
                raise ValueError(f"Unsupported Rc variant: {spec}")

        # Handle regular reshape (e.g., 'R64,64,3')
        config = parse_reshape_spec(spec)
        layer = self.Reshape(*config.target_shape)
        self.layers.append(layer)
        self.shape = config.target_shape
        return layer

    class Reshape(nn.Module):
        def __init__(self, *args):
            super().__init__()
            self.target_shape = args

        def forward(self, x):
            return x.view(x.size(0), *self.target_shape)

    def input(self, spec: str):
        """
        Parses the input specification and sets the initial shape.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Input layer.

        Returns
        -------
        tuple
            The input shape (excluding batch size).
        """
        config = parse_input_spec(spec)

        # Adjust input shape based on the parsed dimensions
        if config.channels is not None and config.depth is not None:
            # 4D input: shape = (channels, depth, height, width)
            input_shape = (config.channels, config.depth,
                           config.height, config.width)
        elif config.channels is not None:
            # 3D input: shape = (channels, height, width)
            input_shape = (config.channels, config.height, config.width)
        elif config.height is not None:
            # 2D input: shape = (height, width)
            input_shape = (config.height, config.width)
        else:
            # 1D input: shape = (width,)
            input_shape = (config.width,)

        self.shape = input_shape
        return input_shape

    def flatten(self, spec: str) -> nn.Flatten:
        """
        Create a Flatten layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Flatten layer. Expected format: 'Flt'.

        Returns
        -------
        torch.nn.Flatten
            The created Flatten layer.
        """
        if spec != "Flt":
            raise ValueError(
                f"Flatten layer spec '{spec}' is incorrect. Expected 'Flt'.")

        layer = nn.Flatten()
        self.layers.append(layer)
        # Update shape
        self.shape = (int(torch.prod(torch.tensor(self.shape)).item()),)
        return layer

    def build_final_model(self, name: str = "VGSL_Model") -> nn.Module:
        """
        Build the final model using the accumulated layers.

        Parameters
        ----------
        name : str, optional
            The name of the model, by default "VGSL_Model"

        Returns
        -------
        torch.nn.Module
            The constructed PyTorch model.
        """
        class VGSLModel(nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.model = nn.Sequential(*layers)

            def forward(self, x):
                return self.model(x)

        model = VGSLModel(self.layers)
        model.__class__.__name__ = name
        return model
