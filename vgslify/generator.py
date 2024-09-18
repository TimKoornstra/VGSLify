# Imports

# > Standard Libraries
from typing import Any, Dict, List

# > Internal Libraries
from vgslify.core.parser import parse_spec


class VGSLModelGenerator:
    """
    VGSLModelGenerator constructs a neural network model based on a VGSL (Variable-size Graph
    Specification Language) specification string. This class supports dynamic model generation
    for different backends, with current support for TensorFlow.

    The generator takes a VGSL specification string that defines the architecture of the neural
    network, including the input layer, convolutional layers, pooling layers, RNN layers, dense
    layers, and more. The class parses this string, constructs the layers in sequence, and builds
    the final model.
    """

    def __init__(self, backend: str = "auto") -> None:
        """
        Initialize the VGSLModelGenerator with the backend and layer factory.

        Parameters
        ----------
        backend : str, optional
            The backend to use for building the model. Can be "tensorflow", "torch", or "auto".
            Default is "auto", which will attempt to automatically detect the available backend.
        """
        self.backend = self._detect_backend(backend)
        self.layer_factory, self.layer_constructors = self._initialize_backend_and_factory(
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
        # Parse the specification string
        specs = parse_spec(model_spec)

        # Initialize the first layer (input layer)
        inputs = self.layer_factory.input(specs[0])
        outputs = inputs

        # Build the model by iterating through each layer specification
        prev_layer = inputs
        for spec in specs[1:]:
            outputs = self._construct_and_chain_layer(
                spec, outputs, prev_layer)
            prev_layer = outputs  # Keep track of the previous layer for `Rc`

        # Build and return the final model
        return self.layer_factory.build_final_model(inputs, outputs)

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
        # Parse the specification string
        specs = parse_spec(model_spec)
        history = []

        # Build each layer and store in history
        prev_layer = None
        for spec in specs:
            layer = self._construct_layer(spec, prev_layer)
            history.append(layer)
            prev_layer = layer

        return history

    def construct_layer(self, spec: str, prev_layer: Any = None) -> Any:
        """
        Constructs a single layer using the layer factory based on the spec string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for a layer.
        prev_layer : Any, optional
            The previous layer, required for spatial collapsing ('Rc').

        Returns
        -------
        Any
            The constructed layer.

        Raises
        ------
        ValueError
            If the layer specification is unknown or if 'Rc' requires the previous layer's shape.
        """
        return self._construct_layer(spec, prev_layer)

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
        Initialize the backend and return the layer factory and constructor map.

        Parameters
        ----------
        backend : str
            The backend to use for building the model.

        Returns
        -------
        tuple
            A tuple containing the layer factory and layer constructors dictionary.
        """
        if backend == "tensorflow":
            from vgslify.tensorflow.layers import TensorFlowLayerFactory as LayerFactory
        elif backend == "torch":
            raise NotImplementedError(
                "The 'torch' backend is not implemented yet.")
        else:
            raise ValueError(
                f"Unsupported backend: {backend}. Choose 'tensorflow' or 'torch'.")

        layer_factory = LayerFactory()

        layer_constructors: Dict[str, Any] = {
            'C': layer_factory.conv2d,
            'Mp': layer_factory.maxpooling2d,
            'Ap': layer_factory.avgpool2d,
            'L': layer_factory.lstm,
            'G': layer_factory.gru,
            'B': layer_factory.bidirectional,
            'Flt': layer_factory.flatten,
            'F': layer_factory.dense,
            'D': layer_factory.dropout,
            'Bn': layer_factory.batchnorm,
            'A': layer_factory.activation,
            'R': layer_factory.reshape,
            'O': layer_factory.output,
            'Rc': layer_factory.reshape,
        }

        return layer_factory, layer_constructors

    def _construct_layer(self, spec: str, prev_layer: Any = None) -> Any:
        """
        Constructs a layer using the layer factory based on the specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for a layer.
        prev_layer : Any, optional
            The previous layer, required for spatial collapsing ('Rc').

        Returns
        -------
        Any
            The constructed layer.

        Raises
        ------
        ValueError
            If the layer specification is unknown or if 'Rc' requires the previous layer's shape.
        """
        for prefix in sorted(self.layer_constructors.keys(), key=len, reverse=True):
            if spec.startswith(prefix):
                if prefix == 'Rc':
                    return self.layer_constructors[prefix](spec, prev_layer)
                return self.layer_constructors[prefix](spec)

        raise ValueError(f"Unknown layer specification: {spec}")

    def _construct_and_chain_layer(self, spec: str, outputs: Any, prev_layer: Any = None) -> Any:
        """
        Constructs and chains a layer to the given outputs.

        Parameters
        ----------
        spec : str
            The layer specification string.
        outputs : Any
            The current output of the previous layer to which the new layer will be connected.
        prev_layer : Any, optional
            The previous layer, required for spatial collapsing ('Rc').

        Returns
        -------
        Any
            The updated outputs after chaining the new layer.
        """
        layer = self._construct_layer(spec, prev_layer)
        return layer(outputs)
