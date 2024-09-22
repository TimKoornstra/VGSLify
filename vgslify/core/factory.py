from abc import ABC, abstractmethod


class LayerFactory(ABC):
    """
    Abstract base class for creating neural network layers from VGSL
    specifications. This class defines the interface that must be implemented
    by concrete factories for different frameworks (e.g., TensorFlow, PyTorch).

    It also provides common methods for output shape calculations to be used by
    subclasses.
    """

    def __init__(self):
        self.layers = []
        self.shape = None  # Shape excluding batch size

    @abstractmethod
    def conv2d(self, spec: str):
        pass

    @abstractmethod
    def maxpooling2d(self, spec: str):
        pass

    @abstractmethod
    def avgpool2d(self, spec: str):
        pass

    @abstractmethod
    def dense(self, spec: str):
        pass

    @abstractmethod
    def lstm(self, spec: str):
        pass

    @abstractmethod
    def gru(self, spec: str):
        pass

    @abstractmethod
    def bidirectional(self, spec: str):
        pass

    @abstractmethod
    def dropout(self, spec: str):
        pass

    @abstractmethod
    def batchnorm(self, spec: str):
        pass

    @abstractmethod
    def activation(self, spec: str):
        pass

    @abstractmethod
    def reshape(self, spec: str):
        pass

    @abstractmethod
    def input(self, spec: str):
        pass

    @abstractmethod
    def flatten(self, spec: str):
        pass

    @abstractmethod
    def build_final_model(self, name):
        pass

    def _compute_conv_output_shape(self, input_shape, config):
        """
        Computes the output shape of a convolutional layer.

        Parameters
        ----------
        input_shape : tuple
            The input shape, format depends on the backend (e.g., (C, H, W) for
            PyTorch or (H, W, C) for TensorFlow).
        config : Any
            The configuration object returned by parse_conv2d_spec.

        Returns
        -------
        tuple
            The output shape after the convolution.
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses.")

    def _compute_pool_output_shape(self, input_shape, config):
        """
        Computes the output shape of a pooling layer.

        Parameters
        ----------
        input_shape : tuple
            The input shape.
        config : Any
            The configuration object returned by parse_pooling2d_spec.

        Returns
        -------
        tuple
            The output shape after pooling.
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses.")
