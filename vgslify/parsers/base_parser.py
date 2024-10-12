from abc import ABC, abstractmethod
from typing import List, Union
from vgslify.core.config import (
    ActivationConfig,
    Conv2DConfig,
    Pooling2DConfig,
    DenseConfig,
    RNNConfig,
    DropoutConfig,
    ReshapeConfig,
    InputConfig
)

class BaseModelParser(ABC):
    """
    Abstract base class for model parsers.
    Provides common utility methods for parsing different frameworks and generating VGSL spec strings.
    """

    def generate_vgsl(self, configs: List[Union[
        Conv2DConfig,
        Pooling2DConfig,
        DenseConfig,
        RNNConfig,
        DropoutConfig,
        ReshapeConfig,
        InputConfig,
        ActivationConfig
    ]]) -> str:
        """
        Convert a list of layer configuration dataclasses into a VGSL specification string.

        Parameters
        ----------
        configs : List[Union[Conv2DConfig, Pooling2DConfig, DenseConfig, RNNConfig,
                             DropoutConfig, ReshapeConfig, InputConfig, ActivationConfig]]
            List of layer configurations.

        Returns
        -------
        str
            VGSL specification string.
        """
        vgsl_parts = []
        i = len(configs) - 1  # Start from the end of the list to merge activations

        while i >= 0:
            config = configs[i]

            if isinstance(config, ActivationConfig):
                # Check if there is a preceding layer to merge with
                if i > 0:
                    preceding_config = configs[i - 1]
                    if isinstance(preceding_config, (Conv2DConfig, DenseConfig, RNNConfig)) and \
                            preceding_config.activation == 'linear':
                        # Merge the activation into the preceding layer
                        preceding_config.activation = config.activation
                        # Skip adding this ActivationConfig
                        i -= 1
                        continue
                # If cannot merge, add the activation spec
                vgsl_parts.append(self._vgsl_activation(config))
            else:
                # Handle non-activation layers and strings
                if isinstance(config, InputConfig):
                    vgsl_parts.append(self._vgsl_input(config))
                elif isinstance(config, Conv2DConfig):
                    vgsl_parts.append(self._vgsl_conv2d(config))
                elif isinstance(config, Pooling2DConfig):
                    vgsl_parts.append(self._vgsl_pooling2d(config))
                elif isinstance(config, DenseConfig):
                    vgsl_parts.append(self._vgsl_dense(config))
                elif isinstance(config, RNNConfig):
                    vgsl_parts.append(self._vgsl_rnn(config))
                elif isinstance(config, DropoutConfig):
                    vgsl_parts.append(self._vgsl_dropout(config))
                elif isinstance(config, ReshapeConfig):
                    vgsl_parts.append(self._vgsl_reshape(config))
                elif isinstance(config, str):
                    vgsl_parts.append(config)
                else:
                    raise ValueError(f"Unsupported configuration type: {type(config).__name__}")
            i -= 1  # Move to the previous config

        # Reverse to restore the original order
        return " ".join(vgsl_parts[::-1])

    @abstractmethod
    def parse_model(self, model) -> str:
        """Parse the model into a VGSL spec string."""
        pass

    @abstractmethod
    def parse_input(self, layer) -> InputConfig:
        """Parse the input layer into a InputConfig dataclass."""
        pass

    @abstractmethod
    def parse_conv2d(self, layer) -> Conv2DConfig:
        """Parse the Conv2D layer into a Conv2DConfig dataclass."""
        pass

    @abstractmethod
    def parse_dense(self, layer) -> DenseConfig:
        """Parse the Dense layer into a DenseConfig dataclass."""
        pass

    @abstractmethod
    def parse_rnn(self, layer) -> RNNConfig:
        """Parse the RNN layer into a RNNConfig dataclass."""
        pass

    @abstractmethod
    def parse_pooling(self, layer) -> Pooling2DConfig:
        """Parse the Pooling layer into a Pooling2DConfig dataclass."""
        pass

    @abstractmethod
    def parse_batchnorm(self, layer) -> str:
        """Parse the BatchNorm layer into a VGSL spec string."""
        pass

    @abstractmethod
    def parse_dropout(self, layer) -> DropoutConfig:
        """Parse the Dropout layer into a DropoutConfig dataclass."""
        pass

    @abstractmethod
    def parse_flatten(self, layer) -> str:
        """Parse the Flatten layer into a VGSL spec string."""
        pass

    @abstractmethod
    def parse_reshape(self, layer) -> ReshapeConfig:
        """Parse the Reshape layer into a ReshapeConfig dataclass."""
        pass

    @abstractmethod
    def parse_activation(self, layer) -> ActivationConfig:
        """Parse the Activation layer into a ActivationConfig dataclass."""
        pass

    # VGSL Generation Methods
    def _vgsl_input(self, config: InputConfig) -> str:
        """
        Generate VGSL string for input layer.

        Parameters
        ----------
        config : InputConfig
            Configuration for the input layer.

        Returns
        -------
        str
            VGSL string representation of the input layer.
        """
        return ",".join(map(str, filter(lambda x: x != -1, [
            config.batch_size,
            config.depth,
            config.height,
            config.width,
            config.channels
        ])))

    def _vgsl_conv2d(self, config: Conv2DConfig) -> str:
        """
        Generate VGSL string for Conv2D layer.

        Parameters
        ----------
        config : Conv2DConfig
            Configuration for the Conv2D layer.

        Returns
        -------
        str
            VGSL string representation of the Conv2D layer.
        """
        act = self._get_activation_code(config.activation)
        stride_spec = ",".join(map(str, config.strides)) if config.strides != (1, 1) else ""
        stride_str = f",{stride_spec}" if stride_spec else ""
        return f"C{act}{config.kernel_size[0]},{config.kernel_size[1]}{stride_str},{config.filters}"

    def _vgsl_pooling2d(self, config: Pooling2DConfig) -> str:
        """
        Generate VGSL string for Pooling2D layer.

        Parameters
        ----------
        config : Pooling2DConfig
            Configuration for the Pooling2D layer.

        Returns
        -------
        str
            VGSL string representation of the Pooling2D layer.
        """
        pool_type_code = 'Mp' if config.pool_type.lower() == 'max' else 'Ap'
        pool_size_str = ",".join(map(str, config.pool_size))
        strides_str = ",".join(map(str, config.strides)) if config.strides != config.pool_size else ""
        return f"{pool_type_code}{pool_size_str}{',' + strides_str if strides_str else ''}"

    def _vgsl_dense(self, config: DenseConfig) -> str:
        """
        Generate VGSL string for Dense layer.

        Parameters
        ----------
        config : DenseConfig
            Configuration for the Dense layer.

        Returns
        -------
        str
            VGSL string representation of the Dense layer.
        """
        act = self._get_activation_code(config.activation)
        return f"F{act}{config.units}"

    def _vgsl_rnn(self, config: RNNConfig) -> str:
        """
        Generate VGSL string for RNN layer.

        Parameters
        ----------
        config : RNNConfig
            Configuration for the RNN layer.

        Returns
        -------
        str
            VGSL string representation of the RNN layer.

        Raises
        ------
        ValueError
            If an unsupported RNN type is provided.
        """
        if config.bidirectional:
            layer_type = 'B'
            rnn_type = 'l' if config.rnn_type.lower() == 'lstm' else 'g'
        else:
            if config.rnn_type.lower() == 'lstm':
                layer_type = 'L'
            elif config.rnn_type.lower() == 'gru':
                layer_type = 'G'
            else:
                raise ValueError(f"Unsupported RNN type: {config.rnn_type}")
            rnn_type = 'r' if config.go_backwards else 'f'
        
        return_sequences = 's' if config.return_sequences and not config.bidirectional else ''
        
        spec = f"{layer_type}{rnn_type}{return_sequences}{config.units}"
        
        if config.dropout > 0:
            spec += f",D{int(config.dropout * 100)}"
        if config.recurrent_dropout > 0:
            spec += f",Rd{int(config.recurrent_dropout * 100)}"
        
        return spec

    def _vgsl_dropout(self, config: DropoutConfig) -> str:
        """
        Generate VGSL string for Dropout layer.

        Parameters
        ----------
        config : DropoutConfig
            Configuration for the Dropout layer.

        Returns
        -------
        str
            VGSL string representation of the Dropout layer.
        """
        return f"D{int(config.rate * 100)}"

    def _vgsl_reshape(self, config: ReshapeConfig) -> str:
        """
        Generate VGSL string for Reshape layer.

        Parameters
        ----------
        config : ReshapeConfig
            Configuration for the Reshape layer.

        Returns
        -------
        str
            VGSL string representation of the Reshape layer.
        """
        if len(config.target_shape) == 2 and (None in config.target_shape or -1 in config.target_shape):
            return "Rc3"
        else:
            reshape_dims = ",".join(map(lambda x: str(x) if x is not None else '-1', config.target_shape))
            return f"R{reshape_dims}"

    def _vgsl_activation(self, config: ActivationConfig) -> str:
        """
        Generate VGSL string for Activation layer.

        Parameters
        ----------
        config : ActivationConfig
            Configuration for the Activation layer.

        Returns
        -------
        str
            VGSL string representation of the Activation layer.
        """
        act = self._get_activation_code(config.activation)
        return f"A{act}"

    def _get_activation_code(self, activation: str) -> str:
        """
        Get the VGSL activation code for a given activation function.

        Parameters
        ----------
        activation : str
            Name of the activation function.

        Returns
        -------
        str
            VGSL activation code.

        Raises
        ------
        ValueError
            If an unsupported activation function is provided.
        """
        ACTIVATION_MAP = {
            'softmax': 's', 'tanh': 't', 'relu': 'r',
            'linear': 'l', 'sigmoid': 'm', 'identity': 'l'
        }
        act_code = ACTIVATION_MAP.get(activation.lower(), None)
        if act_code is None:
            raise ValueError(f"Unsupported activation '{activation}'.")
        return act_code
