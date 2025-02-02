import importlib.util

__all__ = []

# Check if TensorFlow is available before importing
if importlib.util.find_spec("tensorflow") is not None:
    from .tensorflow import TensorFlowModelParser

    __all__.append("TensorFlowModelParser")

# Check if Torch is available before importing
if importlib.util.find_spec("torch") is not None:
    from .torch import TorchModelParser

    __all__.append("TorchModelParser")
