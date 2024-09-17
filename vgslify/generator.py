from vgslify.core.parser import parse_spec
from typing import List, Any, Optional, Dict


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

    def __init__(self, model_spec: str, backend: str = "auto") -> None:
        """
        Initialize the VGSLModelGenerator with a given model specification string.

        Parameters
        ----------
        model_spec : str
            The VGSL specification string defining the model architecture.
        backend : str, optional
            The backend to use for building the model. Can be "tensorflow", "torch", or "auto".
            Default is "auto", which will attempt to automatically detect the available backend.
        """
        self.model_spec: str = model_spec
        self.history: List[Any] = []
        self.inputs: Optional[Any] = None
        self.outputs: Optional[Any] = None

        # Automatically detect backend if set to "auto"
        if backend == "auto":
            backend = self._detect_backend()

        # Dynamically import and set the layer factory based on the backend
        if backend == "tensorflow":
            from vgslify.tensorflow.layers import TensorFlowLayerFactory as LayerFactory
        elif backend == "torch":
            raise NotImplementedError(
                "The 'torch' backend is not implemented yet."
            )
        else:
            raise ValueError(
                f"Unsupported backend: {backend}. Choose 'tensorflow' or 'torch'."
            )

        self.layer_factory: Any = LayerFactory()

        # Create a dictionary that maps prefixes to layer creation methods
        self.layer_constructors: Dict[str, Any] = {
            'C': self.layer_factory.conv2d,
            'Mp': self.layer_factory.maxpooling2d,
            'Ap': self.layer_factory.avgpool2d,
            'L': self.layer_factory.lstm,
            'G': self.layer_factory.gru,
            'B': self.layer_factory.bidirectional,
            'Flt': self.layer_factory.flatten,
            'F': self.layer_factory.dense,
            'D': self.layer_factory.dropout,
            'Bn': self.layer_factory.batchnorm,
            'A': self.layer_factory.activation,
            'R': self.layer_factory.reshape,
            'O': self.layer_factory.output,
        }

    def _detect_backend(self) -> str:
        """
        Detect the backend automatically by checking available libraries.
        If both TensorFlow and PyTorch are available, TensorFlow is selected by default.

        Returns
        -------
        str
            The detected backend ("tensorflow" or "torch").

        Raises
        ------
        ImportError
            If neither TensorFlow nor PyTorch is available.
        """
        try:
            import tensorflow as tf
            tf_available = True
        except ImportError:
            tf_available = False

        try:
            import torch
            torch_available = True
        except ImportError:
            torch_available = False

        if tf_available and not torch_available:
            return "tensorflow"
        if torch_available and not tf_available:
            return "torch"
        if tf_available and torch_available:
            print("Both TensorFlow and PyTorch are available. Defaulting to TensorFlow.")
            return "tensorflow"
        raise ImportError(
            "Neither TensorFlow nor PyTorch is installed. Please install one of them."
        )

    def generate_history(self) -> List[Any]:
        """
        Generate the history of layer specifications without building the full model.

        This method parses the VGSL specification string, constructs each layer using
        the layer factory, and stores them in history, but does not chain them or
        connect input/output layers.

        Returns
        -------
        list
            A list of layers constructed from the specification string.
        """
        specs = self._parse_specifications()
        history = []

        for spec in specs:
            layer = self.construct_layer(spec)
            history.append(layer)

        # Update the instance's history
        self.history = history
        return history

    def construct_layer(self, spec: str) -> Any:
        """
        Constructs a layer using the layer factory based on the spec string.

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
        # Find the longest prefix match in the layer_constructors dictionary
        for prefix in sorted(self.layer_constructors.keys(), key=len, reverse=True):
            if spec.startswith(prefix):
                # Return the layer without chaining it to any inputs/outputs
                return self.layer_constructors[prefix](spec)

        raise ValueError(f"Unknown layer specification: {spec}")

    def build_layers(self) -> List[Any]:
        """
        Build the layers as specified, without connecting them to input/output layers.

        Returns
        -------
        list
            A list of constructed layers.
        """
        # Parse the specification string to get the list of layer specs
        specs = self._parse_specifications()

        # Build each layer without chaining them
        layers = [self.construct_layer(spec) for spec in specs]
        return layers

    def build_model(self) -> Any:
        """
        Build the model based on the VGSL spec string.

        This method parses the VGSL specification string, creates each layer
        using the layer factory, and constructs the model sequentially.

        Returns
        -------
        Any
            The built model using the specified backend.
        """
        # Parse the specification string to get the list of layer specs
        specs = self._parse_specifications()

        # Initialize the model with the first layer (input layer)
        self._initialize_first_layer(specs[0])

        # Process each subsequent layer in the spec and keep track of the latest layer (output)
        for spec in specs[1:]:
            self._process_layer_spec(spec)

        # Finalize and build the model using the last layer as output
        model = self._finalize_model()
        return model

    def _parse_specifications(self) -> List[str]:
        """
        Parse the VGSL specification string into individual layer specifications.

        Returns
        -------
        list of str
            A list of parsed specifications.
        """
        return parse_spec(self.model_spec)

    def _initialize_first_layer(self, first_spec: str) -> None:
        """
        Initialize the model by creating the input layer from the first specification.

        Parameters
        ----------
        first_spec : str
            The specification for the input layer.
        """
        self.inputs = self.layer_factory.input(first_spec)
        self.outputs = self.inputs  # The first output is the input layer

        # Store the input layer in history for future layers
        self.history.append(self.inputs)

    def _process_layer_spec(self, spec: str) -> None:
        """
        Process a single layer specification and append it to the model.

        Parameters
        ----------
        spec : str
            The layer specification string.
        """
        if spec.startswith("Rc"):  # Reshape layer with spatial collapse
            layer = self.layer_factory.reshape(spec, self.history[-1])
        else:
            layer = self.construct_layer(spec)

        # Connect the layer to the previous output
        self.outputs = layer(self.outputs)
        self.history.append(layer)  # Append each layer to history

    def _finalize_model(self) -> Any:
        """
        Finalize the model by connecting the layers and returning the built model.

        Returns
        -------
        Any
            The final built model.
        """
        # Build and return the final model using inputs and outputs
        return self.layer_factory.build_final_model(self.inputs, self.outputs)
