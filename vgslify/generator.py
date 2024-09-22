# Imports

# > Standard Libraries
from typing import Any, Dict, List

# > Internal Libraries
from vgslify.core.parser import parse_spec


class VGSLModelGenerator:
    """
    VGSLModelGenerator constructs a neural network model based on a VGSL (Variable-size Graph
    Specification Language) specification string. This class supports dynamic model generation
    for different backends, with current support for TensorFlow and PyTorch.

    The generator takes a VGSL specification string that defines the architecture of the neural
    network, including the input layer, convolutional layers, pooling layers, RNN layers, dense
    layers, and more. The class parses this string, constructs the layers in sequence, and builds
    the final model.
    """

    def __init__(self, backend: str = "auto") -> None:
        """
        Initialize the VGSLModelGenerator with the backend.

        Parameters
        ----------
        backend : str, optional
            The backend to use for building the model. Can be "tensorflow", "torch", or "auto".
            Default is "auto", which will attempt to automatically detect the available backend.
        """
        self.backend = self._detect_backend(backend)
        self.layer_factory_class, self.layer_constructors = self._initialize_backend_and_factory(
            self.backend)

    def generate_model(self, model_spec: str) -> Any:
        """
        Build the model based on the VGSL spec string.

        This method parses the VGSL specification string, creates each layer
        using the layer factory, and constructs the model sequentially.

        Parameters
        ----------
        model_spec : str
            The VGSL specification string defining the model architecture.

        Returns
        -------
        Any
            The built model using the specified backend.
        """
        # Create a new instance of the layer factory for this model
        layer_factory = self.layer_factory_class()

        # Parse the specification string
        specs = parse_spec(model_spec)

        # Initialize the first layer (input layer)
        layer_factory.input(specs[0])

        # Build the model by iterating through each layer specification
        for spec in specs[1:]:
            self._construct_layer(spec, layer_factory)

        # Build and return the final model
        return layer_factory.build_final_model()

    def generate_history(self, model_spec: str) -> List[Any]:
        """
        Generate the history of layer specifications without building the full model.

        This method parses the VGSL specification string, constructs each layer using
        the layer factory, and stores them in a list, but does not chain them or connect
        input/output layers.

        Parameters
        ----------
        model_spec : str
            The VGSL specification string defining the model architecture.

        Returns
        -------
        list
            A list of layers constructed from the specification string.
        """
        # Create a new instance of the layer factory
        layer_factory = self.layer_factory_class()

        # Parse the specification string
        specs = parse_spec(model_spec)

        history = []

        # Build each layer and store in history
        for spec in specs:
            layer = self._construct_layer(spec, layer_factory)
            history.append(layer)

        return history

    def construct_layer(self, spec: str) -> Any:
        """
        Constructs a single layer using the layer factory based on the spec string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for a layer.

        Returns
        -------
        Any
            The constructed layer.

        Raises
        ------
        ValueError
            If the layer specification is unknown.
        """
        # Create a new instance of the layer factory
        layer_factory = self.layer_factory_class()
        return self._construct_layer(spec, layer_factory)

    ### Private Helper Methods ###

    def _detect_backend(self, backend: str) -> str:
        """
        Detect the backend automatically by checking available libraries.
        If both TensorFlow and PyTorch are available, TensorFlow is selected by default.

        Parameters
        ----------
        backend : str
            The backend to use for building the model. Can be "tensorflow", "torch", or "auto".

        Returns
        -------
        str
            The detected or provided backend ("tensorflow" or "torch").
        """
        if backend != "auto":
            return backend

        try:
            import tensorflow as tf
            return "tensorflow"
        except ImportError:
            pass

        try:
            import torch
            return "torch"
        except ImportError:
            pass

        raise ImportError(
            "Neither TensorFlow nor PyTorch is installed. Please install one of them.")

    def _initialize_backend_and_factory(self, backend: str) -> tuple:
        """
        Initialize the backend and return the layer factory class and constructor map.

        Parameters
        ----------
        backend : str
            The backend to use for building the model.

        Returns
        -------
        tuple
            A tuple containing the layer factory class and layer constructors dictionary.
        """
        try:
            if backend == "tensorflow":
                from vgslify.tensorflow.layers import TensorFlowLayerFactory as LayerFactory
            elif backend == "torch":
                from vgslify.torch.layers import TorchLayerFactory as LayerFactory
            else:
                raise ValueError(
                    f"Unsupported backend: {backend}. Choose 'tensorflow' or 'torch'.")
        except ImportError:
            raise ImportError(
                f"Backend '{backend}' is not available. Please install the required library.")

        layer_constructors: Dict[str, Any] = {
            'C': LayerFactory.conv2d,
            'Mp': LayerFactory.maxpooling2d,
            'Ap': LayerFactory.avgpool2d,
            'L': LayerFactory.lstm,
            'G': LayerFactory.gru,
            'B': LayerFactory.bidirectional,
            'Flt': LayerFactory.flatten,
            'F': LayerFactory.dense,
            'D': LayerFactory.dropout,
            'Bn': LayerFactory.batchnorm,
            'A': LayerFactory.activation,
            'R': LayerFactory.reshape,
            'Rc': LayerFactory.reshape,
        }

        return LayerFactory, layer_constructors

    def _construct_layer(self, spec: str, layer_factory) -> Any:
        """
        Constructs a layer using the layer factory based on the specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for a layer.
        layer_factory : Any
            The layer factory instance to use for constructing the layer.

        Returns
        -------
        Any
            The constructed layer.

        Raises
        ------
        ValueError
            If the layer specification is unknown.
        """
        for prefix in sorted(self.layer_constructors.keys(), key=len, reverse=True):
            if spec.startswith(prefix):
                layer_constructor = getattr(
                    layer_factory, self.layer_constructors[prefix].__name__)
                return layer_constructor(spec)

        raise ValueError(f"Unknown layer specification: {spec}")
