# Imports

# > Standard Libraries
import importlib
from typing import Any, Callable, Dict, List

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
        model_name : str, optional
            The name of the model, by default "VGSL_Model"
        """
        self.backend = self._detect_backend(backend)
        self.layer_factory_class, self.layer_constructors = (
            self._initialize_backend_and_factory(self.backend)
        )
        self.layer_factory = self.layer_factory_class()

    def generate_model(self, model_spec: str, model_name: str = "VGSL_Model") -> Any:
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
        return self._process_layers(
            model_spec, return_history=False, model_name=model_name
        )

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
        return self._process_layers(model_spec, return_history=True)

    def _process_layers(
        self,
        model_spec: str,
        return_history: bool = False,
        model_name: str = "VGSL_Model",
    ) -> Any:
        """
        Process the VGSL specification string to build the model or generate a history of layers.

        Parameters
        ----------
        model_spec : str
            The VGSL specification string defining the model architecture.
        return_history : bool, optional
            If True, returns a list of constructed layers (history) instead of the final model.
        model_name : str, optional
            The name of the model, by default "VGSL_Model"

        Returns
        -------
        Any
            The built model using the specified backend if `return_history` is False.
            Otherwise, a list of constructed layers.
        """
        # Create a new instance of the layer factory for this model
        self.layer_factory = self.layer_factory_class()

        # Parse the specification string
        specs = parse_spec(model_spec)

        # Initialize the first layer (input layer)
        input_layer = self.layer_factory.input(specs[0])

        # Initialize history if required
        history = [input_layer] if return_history else None

        # Process each layer specification
        for spec in specs[1:]:
            layer = self._construct_layer(spec, self.layer_factory)
            if return_history:
                history.append(layer)

        if return_history:
            return history

        # Build and return the final model
        return self.layer_factory.build(name=model_name)

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

        if importlib.util.find_spec("tensorflow") is not None:
            return "tensorflow"

        if importlib.util.find_spec("torch") is not None:
            return "torch"

        raise ImportError(
            "Neither TensorFlow nor PyTorch is installed. Please install one of them."
        )

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
                from vgslify.tensorflow.layers import (
                    TensorFlowLayerFactory as FactoryClass,
                )
            elif backend == "torch":
                from vgslify.torch.layers import TorchLayerFactory as FactoryClass
            else:
                raise ValueError(
                    f"Unsupported backend: {backend}. Choose 'tensorflow' or 'torch'."
                )
        except ImportError:
            raise ImportError(
                f"Backend '{backend}' is not available. Please install the required library."
            )

        layer_constructors: Dict[str, Callable[[Any, str], Any]] = {
            "C": lambda f, s: f.conv2d(s),
            "Mp": lambda f, s: f.pooling2d(s),
            "Ap": lambda f, s: f.pooling2d(s),
            "L": lambda f, s: f.rnn(s),
            "G": lambda f, s: f.rnn(s),
            "B": lambda f, s: f.rnn(s),
            "Flt": lambda f, s: f.flatten(s),
            "F": lambda f, s: f.dense(s),
            "D": lambda f, s: f.dropout(s),
            "Bn": lambda f, s: f.batchnorm(s),
            "A": lambda f, s: f.activation(s),
            "R": lambda f, s: f.reshape(s),
            "Rc": lambda f, s: f.reshape(s),
        }

        # Fetch any user-registered (prefix -> function) for this factory
        custom_registry = FactoryClass.get_custom_layer_registry()

        # Wrap each custom builder so returned layers automatically get appended
        for prefix, builder_fn in custom_registry.items():

            def wrapped_builder(factory, spec, _builder_fn=builder_fn):
                layer = _builder_fn(factory, spec)
                # Only append if not already appended. (Usually it won't be.)
                if layer not in factory.layers:
                    factory.layers.append(layer)
                return layer

            layer_constructors[prefix] = wrapped_builder

        return FactoryClass, layer_constructors

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
                layer_fn = self.layer_constructors[prefix]
                return layer_fn(layer_factory, spec)

        raise ValueError(f"Unknown layer specification: {spec}")
