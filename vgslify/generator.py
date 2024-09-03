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

    Parameters
    ----------
    model_spec : str
        The VGSL specification string defining the model architecture.
        Example format: "None,64,64,3 C3,3,32 Mp2,2,2,2 Flt Bgl256 D50 O1s10"
    backend : str, optional
        The backend to use for building the model. Default is "tensorflow".
        Supported options: "tensorflow". Other backends like "torch" can be implemented in the
        future.

    Methods
    -------
    build_model():
        Parses the VGSL spec string, constructs the layers, and builds the model.

    construct_layer(spec: str):
        Constructs a layer based on the provided VGSL specification string.
    """

    def __init__(self, model_spec: str, backend: str = "tensorflow"):
        """
        Initialize the VGSLModelGenerator with a given model specification string.

        Parameters
        ----------
        model_spec : str
            The VGSL specification string defining the model architecture.
        backend : str, optional
            The backend to use for building the model. Default is "tensorflow".
            Options include "tensorflow" (currently implemented) and "torch" (NotImplemented).
        """
        self.model_spec = model_spec
        self.history = []
        self.inputs = None

        # Dynamically import and set the layer factory based on the backend
        if backend == "tensorflow":
            from vgslify.tensorflow.layers import TensorFlowLayerFactory as LayerFactory
        elif backend == "torch":
            raise NotImplementedError(
                "The 'torch' backend is not implemented yet.")
        else:
            raise ValueError(
                f"Unsupported backend: {backend}. Choose 'tensorflow'.")

        # Initialize the layer factory
        self.layer_factory = LayerFactory()

        # Create a dictionary that maps prefixes to layer creation methods
        self.layer_constructors = {
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

    def build_model(self):
        """
        Build the model based on the VGSL spec string.

        Returns
        -------
        model
            The built model using the specified backend.
        """
        specs = parse_spec(self.model_spec)  # Parse the spec string
        for index, spec in enumerate(specs):
            if index == 0:  # First layer should be the input layer
                self.inputs = self.layer_factory.input(spec)
                self.history.append(self.inputs)
            else:
                layer = self.construct_layer(spec)
                self.history.append(layer)

        # Example of how to build the model (TensorFlow example)
        x = self.inputs
        for layer in self.history[1:]:
            x = layer(x)

        # Finalize the model creation
        model = self.layer_factory.build_final_model(self.inputs, x)
        return model

    def construct_layer(self, spec: str):
        """
        Constructs a layer using the layer factory based on the spec string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for a layer.

        Returns
        -------
        Layer
            The constructed layer.
        """
        # Find the longest prefix match in the layer_constructors dictionary
        for prefix in sorted(self.layer_constructors.keys(), key=len, reverse=True):
            if spec.startswith(prefix):
                return self.layer_constructors[prefix](spec)

        raise ValueError(f"Unknown layer specification: {spec}")
