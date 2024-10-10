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

    @abstractmethod
    def parse_model(self, model) -> str:
        """Parse the model into a VGSL spec string."""
        pass

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
        configs : List[Union[Conv2DConfig, Pooling2DConfig, DenseConfig, RNNConfig, DropoutConfig, ReshapeConfig, InputConfig]]
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

    # VGSL Generation Methods
    def _vgsl_input(self, config: InputConfig) -> str:
        return ",".join(map(str, filter(lambda x: x != -1, [
            config.batch_size,
            config.depth,
            config.height,
            config.width,
            config.channels
        ])))

    def _vgsl_conv2d(self, config: Conv2DConfig) -> str:
        act = self._get_activation_code(config.activation)
        stride_spec = ",".join(map(str, config.strides)) if config.strides != (1, 1) else ""
        stride_str = f",{stride_spec}" if stride_spec else ""
        return f"C{act}{config.kernel_size[0]},{config.kernel_size[1]}{stride_str},{config.filters}"

    def _vgsl_pooling2d(self, config: Pooling2DConfig) -> str:
        pool_type_code = 'Mp' if config.pool_type.lower() == 'max' else 'Ap'
        pool_size_str = ",".join(map(str, config.pool_size))
        strides_str = ",".join(map(str, config.strides)) if config.strides != config.pool_size else ""
        return f"{pool_type_code}{pool_size_str}{',' + strides_str if strides_str else ''}"

    def _vgsl_dense(self, config: DenseConfig) -> str:
        act = self._get_activation_code(config.activation)
        return f"F{act}{config.units}"

    def _vgsl_rnn(self, config: RNNConfig) -> str:
        direction = 'r' if config.go_backwards else 'f'
        return_sequences = 's' if config.return_sequences else ''
        dropout_spec = ""
        if config.dropout > 0:
            dropout_spec += f",D{int(config.dropout * 100)}"
        if config.recurrent_dropout > 0:
            dropout_spec += f",Rd{int(config.recurrent_dropout * 100)}"
        rnn_type_code = 'L' if config.rnn_type.lower() == 'lstm' else 'G' if config.rnn_type.lower() == 'gru' else 'R'
        return f"{rnn_type_code}{direction}{return_sequences}{config.units}{dropout_spec}"

    def _vgsl_dropout(self, config: DropoutConfig) -> str:
        return f"D{int(config.rate * 100)}"

    def _vgsl_reshape(self, config: ReshapeConfig) -> str:
        reshape_dims = ",".join(map(str, config.target_shape))
        return f"R{reshape_dims}"

    def _vgsl_activation(self, config: ActivationConfig) -> str:
        act = self._get_activation_code(config.activation)
        return f"A{act}"

    def _get_activation_code(self, activation: str) -> str:
        ACTIVATION_MAP = {
            'softmax': 's', 'tanh': 't', 'relu': 'r',
            'linear': 'l', 'sigmoid': 'm'
        }
        act_code = ACTIVATION_MAP.get(activation.lower(), None)
        if act_code is None:
            raise ValueError(f"Unsupported activation '{activation}'.")
        return act_code
