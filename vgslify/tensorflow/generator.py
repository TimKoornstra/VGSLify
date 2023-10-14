# Imports

# > Standard library
import re
import logging

# > Third party dependencies
import tensorflow as tf

logger = logging.getLogger(__name__)


class VGSLModelGenerator:
    """
    Generates a VGSL (Variable-size Graph Specification Language) model based
    on a given specification string.

    VGSL is a domain-specific language that allows the rapid specification of
    neural network architectures. This class provides a way to generate and
    initialize a model using a custom VGSL specification string. It supports
    various layers like Conv2D, LSTM, GRU, Dense, etc.

    Parameters
    ----------
    model : str
        VGSL spec string defining the model architecture.
    name : str, optional
        Name of the model. Defaults to the given `model` string or
        "custom_model" if it's a VGSL spec string.
    channels : int, optional
        Number of input channels. Overrides the channels specified in the
        VGSL spec string if provided.
    output_classes : int, optional
        Number of output classes. Overrides the number of classes specified in
        the VGSL spec string if provided.

    Attributes
    ----------
    model_name : str
        Name of the model.
    history : list
        A list that keeps track of the order of layers added to the model.
    selected_model_vgsl_spec : list
        List of individual layers/components from the VGSL spec string.
    inputs : tf.layers.Input
        Input layer of the model.

    Raises
    ------
    ValueError
        If there's an issue with the VGSL spec string format or unsupported
        operations.

    Examples
    --------
    >>> vgsl_gn = VGSLModelGenerator("None,64,None,1 Cr3,3,32 Mp2,2,2,2 O1s92")
    >>> model = vgsl_gn.build()
    >>> model.summary()
    """

    def __init__(self,
                 model: str,
                 name: str = None,
                 channels: int = None,
                 output_classes: int = None,
                 initializer: tf.keras.initializers.Initializer
                 = tf.keras.initializers.GlorotNormal):
        """
        Initialize the VGSLModelGenerator instance.

        Parameters
        ----------
        model : str
            VGSL spec string.
        name : str, optional
            Custom name for the model. If not provided, uses the model name
            or "custom_model" for VGSL specs.
        channels : int, optional
            Number of input channels. If provided, overrides the channels
            from the VGSL spec.
        output_classes : int, optional
            Number of output classes. If provided, overrides the number from
            the VGSL spec.
        initializer : tf.keras.initializers.Initializer, optional
            Initializer for the kernel weights. Defaults to GlorotNormal.

        Raises
        ------
        ValueError:
            If there's an issue with the VGSL spec string format or unsupported
            operations.
        """

        self._initializer = initializer
        self._channel_axis = -1

        if model is None:
            raise ValueError(
                "No model provided. Please provide a VGSL-spec string.")

        try:
            logger.info("Found VGSL-Spec String, testing validity...")
            self.init_model_from_string(model,
                                        channels,
                                        output_classes)
            self.model_name = name if name else "custom_model"

        except (TypeError, AttributeError) as e:
            raise ("Something is wrong with the input string, "
                   "please check the VGSL-spec formatting "
                   "with the documentation.") from e

    def init_model_from_string(self,
                               vgsl_spec_string: str,
                               channels: int = None,
                               output_classes: int = None) -> None:
        """
        Initialize the model based on the given VGSL spec string. This method
        parses the string and creates the model layer by layer.

        Parameters
        ----------
        vgsl_spec_string : str
            VGSL spec string defining the model architecture.
        channels : int, optional
            Number of input channels. Overrides the channels specified in the
            VGSL spec string if provided.
        output_classes : int, optional
            Number of output classes. Overrides the number of classes specified
            in the VGSL spec string if provided.

        Raises
        ------
        ValueError:
            If there's an issue with the VGSL spec string format or unsupported
            operations.
        """

        logger.info("Initializing model")
        self.history = []
        self.selected_model_vgsl_spec = vgsl_spec_string.split()

        # Check if the first layer is an input layer
        pattern = r'^(None|\d+),(None|\d+),(None|\d+),(None|\d+)$'
        if re.match(pattern, self.selected_model_vgsl_spec[0]):
            self.inputs = self.make_input_layer(
                self.selected_model_vgsl_spec[0], channels)
            starting_index = 1
        else:
            self.inputs = None
            starting_index = 0

        for index, layer in \
                enumerate(self.selected_model_vgsl_spec[starting_index:]):
            logger.debug(layer)
            if layer.startswith('C'):
                setattr(self, f"conv2d_{index}", self.Conv2D(
                    layer, self._initializer))
                self.history.append(f"conv2d_{index}")
            elif layer.startswith('Bn'):
                setattr(self, f"batchnorm_{index}",
                        tf.keras.layers.BatchNormalization(
                            axis=self._channel_axis))
                self.history.append(f"batchnorm_{index}")
            elif layer.startswith('L'):
                setattr(self, f"lstm_{index}", self.lstm(
                    layer, self._initializer))
                self.history.append(f"lstm_{index}")
            elif layer.startswith('F'):
                setattr(self, f"dense{index}", self.Dense(
                    layer, self._initializer))
                self.history.append(f"dense{index}")
            elif layer.startswith('B'):
                setattr(self, f"bidirectional_{index}",
                        self.bidirectional(layer, self._initializer))
                self.history.append(f"bidirectional_{index}")
            elif layer.startswith('G'):
                setattr(self, f"gru_{index}", self.gru(
                    layer, self._initializer))
                self.history.append(f"gru_{index}")
            elif layer.startswith('Mp'):
                setattr(self, f"maxpool_{index}",
                        self.MaxPooling2D(layer))
                self.history.append(f"maxpool_{index}")
            elif layer.startswith('Ap'):
                setattr(self, f"avgpool_{index}",
                        self.AvgPool2D(layer))
                self.history.append(f"avgpool_{index}")
            elif layer.startswith('D'):
                setattr(self, f"dropout_{index}",
                        self.dropout(layer))
                self.history.append(f"dropout_{index}")
            elif layer.startswith('R'):
                self.history.append(f"reshape_{index}_{layer}")
            elif layer.startswith('O'):
                setattr(self, f"output_{index}",
                        self.get_output_layer(layer, output_classes))
                self.history.append(f"output_{index}")
            else:
                raise ValueError(f"The current layer: {layer} is not "
                                 "recognised, please check for correct "
                                 "formatting in the VGSL-Spec")

    def build(self) -> tf.keras.models.Model:
        """
        Build the model based on the VGSL spec string.

        Returns
        -------
        tf.keras.models.Model
            The built model.
        """

        logger.info("Building model for: %s", self.selected_model_vgsl_spec)
        if self.inputs is None:
            raise ValueError("No input layer found. Please check the "
                             "VGSL-spec string.")

        x = self.inputs
        for index, layer in enumerate(self.history):
            if layer.startswith("reshape"):
                x = self.Reshape(layer.split("_")[2], x)(x)
            else:
                x = getattr(self, layer)(x)
        output = tf.keras.layers.Activation('linear', dtype=tf.float32)(x)

        logger.info("Model has been built\n")

        return tf.keras.models.Model(inputs=self.inputs,
                                     outputs=output,
                                     name=self.model_name)

    ########################
    #   Helper functions   #
    ########################

    @ staticmethod
    def get_units_or_outputs(layer: str) -> int:
        """
        Retrieve the number of units or outputs from a layer string

        Parameters
        ----------
        layer : str
            Layer string from the VGSL spec.

        Returns
        -------
        int
            Number of units or outputs.
        """

        match = re.findall(r'\d+', layer)
        if not match:
            raise ValueError(
                f"No units or outputs found in layer string {layer}.")
        return int(match[-1])

    @ staticmethod
    def get_activation_function(nonlinearity: str) -> str:
        """
        Retrieve the activation function from the layer string

        Parameters
        ----------
        nonlinearity : str
            Non-linearity string from the layer string.

        Returns
        -------
        str
            Activation function.
        """

        mapping = {'s': 'softmax', 't': 'tanh', 'r': 'relu',
                   'e': 'elu', 'l': 'linear', 'm': 'sigmoid'}

        if nonlinearity not in mapping:
            raise ValueError(
                f"Unsupported nonlinearity '{nonlinearity}' provided.")

        return mapping[nonlinearity]

    def make_input_layer(self,
                         inputs: str,
                         channels: int = None) -> tf.keras.layers.Input:
        """
        Create the input layer based on the input string

        Parameters
        ----------
        inputs : str
            Input string from the VGSL spec.
        channels : int, optional
            Number of input channels.

        Returns
        -------
        tf.keras.layers.Input
            Input layer.
        """

        try:
            batch, height, width, depth = map(
                lambda x: None if x == "None" else int(x), inputs.split(","))
        except ValueError:
            raise ValueError(
                f"Invalid input string format {inputs}. Expected format: "
                "batch,height,width,depth.")

        if channels and depth != channels:
            logger.warning("Overwriting channels from input string. "
                           "Was: %s, now: %s", depth, channels)
            depth = channels
            self.selected_model_vgsl_spec[0] = (f"{batch},{height},"
                                                f"{width},{depth}")

        logger.info("Creating input layer with shape: (%s, %s, %s, %s)",
                    batch, height, width, depth)
        return tf.keras.Input(shape=(width, height, depth))

    #######################
    #   Layer functions   #
    #######################

    @ staticmethod
    def conv2d(layer: str,
               initializer: tf.keras.initializers.Initializer
               = tf.keras.initializers.GlorotNormal) \
            -> tf.keras.layers.Conv2D:
        """
        Generate a 2D convolutional layer based on a VGSL specification string.

        The method creates a Conv2D layer based on the provided VGSL spec
        string. The string can optionally include strides, and if not provided,
        default stride values are used.

        Parameters
        ----------
        layer : str
            VGSL specification for the convolutional layer. Expected format:
            `C(s|t|r|l|m)<x>,<y>,[<s_x>,<s_y>,]<d>`
            - (s|t|r|l|m): Activation type.
            - <x>,<y>: Kernel size.
            - <s_x>,<s_y>: Optional strides (defaults to (1, 1) if not
              provided).
            - <d>: Number of filters (depth).
        initializer : tf.keras.initializers.Initializer, optional
            Initializer for the kernel weights. Defaults to GlorotNormal.

        Returns
        -------
        tf.keras.layers.Conv2D
            A Conv2D layer with the specified parameters.

        Raises
        ------
        ValueError:
            If the provided VGSL spec string does not match the expected
            format.

        Examples
        --------
        >>> from vgslify.tensorflow.generator import VGSLModelGenerator
        >>> conv_layer = VGSLModelGenerator.conv2d("Ct3,3,64")
        >>> type(conv_layer)
        <class 'keras.src.layers.convolutional.conv2d.Conv2D'>
        """

        # Extract convolutional parameters
        conv_filter_params = [int(match)
                              for match in re.findall(r'\d+', layer)]

        # Check if the layer format is as expected
        if len(conv_filter_params) < 3:
            raise ValueError(f"Conv layer {layer} has too few parameters. "
                             "Expected format: C<x>,<y>,<d> or C<x>,<y>,<s_x>"
                             ",<s_y>,<d>")
        elif len(conv_filter_params) > 5:
            raise ValueError(f"Conv layer {layer} has too many parameters. "
                             "Expected format: C<x>,<y>,<d> or C<x>,<y>,<s_x>,"
                             "<s_y>,<d>")

        # Get activation function
        try:
            activation = VGSLModelGenerator.get_activation_function(layer[1])
        except ValueError:
            raise ValueError(
                f"Invalid activation function specified in {layer}")

        # Check parameter length and generate corresponding Conv2D layer
        if len(conv_filter_params) == 3:
            x, y, d = conv_filter_params
            logger.warning(
                "No stride provided, setting default stride of (1,1)")
            return tf.keras.layers.Conv2D(d,
                                          kernel_size=(y, x),
                                          strides=(1, 1),
                                          padding='same',
                                          activation=activation,
                                          kernel_initializer=initializer)
        elif len(conv_filter_params) == 5:
            x, y, s_x, s_y, d = conv_filter_params
            return tf.keras.layers.Conv2D(d,
                                          kernel_size=(y, x),
                                          strides=(s_x, s_y),
                                          padding='same',
                                          activation=activation,
                                          kernel_initializer=initializer)

    @ staticmethod
    def maxpooling2d(layer: str) -> tf.keras.layers.MaxPooling2D:
        """
        Generate a MaxPooling2D layer based on a VGSL specification string.

        Parameters
        ----------
        layer : str
            VGSL specification for the max pooling layer. Expected format:
            `Mp<x>,<y>,<s_x>,<s_y>`
            - <x>,<y>: Pool size.
            - <s_x>,<s_y>: Strides.

        Returns
        -------
        tf.keras.layers.MaxPooling2D
            A MaxPooling2D layer with the specified parameters.

        Raises
        ------
        ValueError:
            If the provided VGSL spec string does not match the expected
            format.

        Examples
        --------
        >>> from vgslify.tensorflow.generator import VGSLModelGenerator
        >>> maxpool_layer = VGSLModelGenerator.maxpooling2d("Mp2,2,2,2")
        >>> type(maxpool_layer)
        <class 'keras.src.layers.pooling.max_pooling2d.MaxPooling2D'>
        """

        # Extract pooling and stride parameters
        pool_stride_params = [int(match)
                              for match in re.findall(r'-?\d+', layer)]

        # Check if the parameters are as expected
        if len(pool_stride_params) != 4:
            raise ValueError(f"MaxPooling layer {layer} does not have the "
                             "expected number of parameters. Expected format: "
                             "Mp<pool_x>,<pool_y>,<stride_x>,<stride_y>")

        pool_x, pool_y, stride_x, stride_y = pool_stride_params

        # Check if pool and stride values are valid
        if pool_x <= 0 or pool_y <= 0 or stride_x <= 0 or stride_y <= 0:
            raise ValueError(f"Invalid values for pooling or stride in "
                             f"{layer}. All values should be positive "
                             "integers.")

        return tf.keras.layers.MaxPooling2D(pool_size=(pool_x, pool_y),
                                            strides=(stride_x, stride_y),
                                            padding='same')

    @ staticmethod
    def avgpool2d(layer: str) -> tf.keras.layers.AvgPool2D:
        """
        Generate an AvgPool2D layer based on a VGSL specification string.

        Parameters
        ----------
        layer : str
            VGSL specification for the average pooling layer. Expected format:
            `Ap<x>,<y>,<s_x>,<s_y>`
            - <x>,<y>: Pool size.
            - <s_x>,<s_y>: Strides.

        Returns
        -------
        tf.keras.layers.AvgPool2D
            An AvgPool2D layer with the specified parameters.

        Raises
        ------
        ValueError:
            If the provided VGSL spec string does not match the expected

        Examples
        --------
        >>> from vgslify.tensorflow.generator import VGSLModelGenerator
        >>> avgpool_layer = VGSLModelGenerator.avgpool2d("Ap2,2,2,2")
        >>> type(avgpool_layer)
        <class 'keras.src.layers.pooling.average_pooling2d.AveragePooling2D'>
        """

        # Extract pooling and stride parameters
        pool_stride_params = [int(match)
                              for match in re.findall(r'-?\d+', layer)]

        # Check if the parameters are as expected
        if len(pool_stride_params) != 4:
            raise ValueError(f"AvgPool layer {layer} does not have the "
                             "expected number of parameters. Expected format: "
                             "Ap<pool_x>,<pool_y>,<stride_x>,<stride_y>")

        pool_x, pool_y, stride_x, stride_y = pool_stride_params

        # Check if pool and stride values are valid
        if pool_x <= 0 or pool_y <= 0 or stride_x <= 0 or stride_y <= 0:
            raise ValueError(f"Invalid values for pooling or stride in "
                             f"{layer}. All values should be positive "
                             "integers.")

        return tf.keras.layers.AvgPool2D(pool_size=(pool_x, pool_y),
                                         strides=(stride_x, stride_y),
                                         padding='same')

    @ staticmethod
    def reshape(layer: str,
                prev_layer: tf.keras.layers.Layer) \
            -> tf.keras.layers.Reshape:
        """
        Generate a reshape layer based on a VGSL specification string.

        The method reshapes the output of the previous layer based on the
        provided VGSL spec string.
        Currently, it supports collapsing the spatial dimensions into a single
        dimension.

        Parameters
        ----------
        layer : str
            VGSL specification for the reshape operation. Expected formats:
            - `Rc`: Collapse the spatial dimensions.
        prev_layer : tf.keras.layers.Layer
            The preceding layer that will be reshaped.

        Returns
        -------
        tf.keras.layers.Reshape
            A Reshape layer with the specified parameters if the operation is
            known, otherwise a string indicating the operation is not known.

        Raises
        ------
        ValueError:
            If the VGSL spec string does not match the expected format or if
            the reshape operation is unknown.

        Examples
        --------
        >>> from vgslify.tensorflow.generator import VGSLModelGenerator
        >>> vgsl_gn = VGSLModelGenerator("Some VGSL string")
        >>> prev_layer = vgsl_gn.make_input_layer("None,64,None,1")
        >>> reshape_layer = vgsl_gn.reshape("Rc", prev_layer)
        >>> type(reshape_layer)
        <class 'keras.src.layers.reshaping.reshape.Reshape'>
        """

        # Check if the layer format is as expected
        if len(layer) < 2:
            raise ValueError(f"Reshape layer {layer} is of unexpected format. "
                             "Expected format: Rc.")

        if layer[1] == 'c':
            prev_layer_y, prev_layer_x = prev_layer.shape[-2:]
            return tf.keras.layers.Reshape((-1, prev_layer_y * prev_layer_x))
        else:
            raise ValueError(f"Reshape operation {layer} is not supported.")

    @ staticmethod
    def dense(layer: str,
              initializer: tf.keras.initializers.Initializer
              = tf.keras.initializers.GlorotNormal) \
            -> tf.keras.layers.Dense:
        """
        Generate a fully connected (dense) layer based on a VGSL specification
        string.

        Parameters
        ----------
        layer : str
            VGSL specification for the fully connected layer. Expected format:
            `F(s|t|r|l|m)<d>`
            - `(s|t|r|l|m)`: Non-linearity type. One of sigmoid, tanh, relu,
            linear, or softmax.
            - `<d>`: Number of outputs.
        initializer : tf.keras.initializers.Initializer, optional
            Initializer for the kernel weights. Defaults to GlorotNormal.

        Returns
        -------
        tf.keras.layers.Dense
            A Dense layer with the specified parameters.

        Raises
        ------
        ValueError:
            If the provided VGSL spec string does not match the expected
            format.

        Examples
        --------
        >>> from vgslify.tensorflow.generator import VGSLModelGenerator
        >>> dense_layer = VGSLModelGenerator.dense("Fr64")
        >>> type(dense_layer)
        <class 'keras.src.layers.core.dense.Dense'>
        >>> dense_layer.activation
        <function relu at 0x7f8b1c0b1d30>

        Notes
        -----
        This method produces a fully connected layer that reduces the height
        and width of the input to 1, producing a single vector as output. The
        input height and width must be constant. For sliding-window operations
        that leave the input image size unchanged, use a 1x1 convolution
        (e.g., Cr1,1,64) instead of this method.
        """

        # Ensure the layer string format is as expected
        if not re.match(r'^F[a-z]-?\d+$', layer):
            raise ValueError(
                f"Dense layer {layer} is of unexpected format. Expected "
                "format: F(s|t|r|l|m)<d>."
            )

        # Check if the activation function is valid
        # or any other supported activations
        try:
            activation = VGSLModelGenerator.get_activation_function(layer[1])
        except ValueError:
            raise ValueError(
                f"Invalid activation '{layer[1]}' for Dense layer "
                f"{layer}.")

        # Extract the number of neurons
        n = int(layer[2:])
        if n <= 0:
            raise ValueError(
                f"Invalid number of neurons {n} for Dense layer {layer}."
            )

        return tf.keras.layers.Dense(n,
                                     activation=activation,
                                     kernel_initializer=initializer)

    @ staticmethod
    def lstm(layer: str,
             initializer: tf.keras.initializers.Initializer
             = tf.keras.initializers.GlorotNormal) \
            -> tf.keras.layers.LSTM:
        """
        Generate an LSTM layer based on a VGSL specification string.

        Parameters
        ----------
        layer : str
            VGSL specification for the LSTM layer. Expected format:
            `L(f|r)[s]<n>[,D<rate>,Rd<rate>]`
            - `(f|r)`: Direction of LSTM. 'f' for forward, 'r' for reversed.
            - `[s]`: (Optional) Summarizes the output, outputting only the
            final step.
            - `<n>`: Number of outputs.
            - `D<rate>` : Dropout rate. Should be between 0 and 100.
            - `Rd<rate>`: Recurrent dropout rate. Should be between 0 and 100.
        initializer : tf.keras.initializers.Initializer, optional
            Initializer for the kernel weights. Defaults to GlorotNormal.

        Returns
        -------
        tf.keras.layers.LSTM
            An LSTM layer with the specified parameters.

        Raises
        ------
        ValueError
            If the provided VGSL spec string does not match the expected
            format.

        Examples
        --------
        >>> from vgslify.tensorflow.generator import VGSLModelGenerator
        >>> lstm_layer = VGSLModelGenerator.lstm("Lf64")
        >>> type(lstm_layer)
        <class 'keras.src.layers.rnn.lstm.LSTM'>
        """

        # Extract direction, summarization, and units
        match = re.match(r'L([fr])(s?)(-?\d+),?(D\d+)?,?(Rd\d+)?$', layer)
        if not match:
            raise ValueError(
                f"LSTM layer {layer} is of unexpected format. Expected "
                "format: L(f|r)[s]<n>[,D<rate>,Rd<rate>].")

        direction, summarize, n, dropout, recurrent_dropout = match.groups()
        dropout = 0 if dropout is None else int(dropout.replace('D', ""))
        recurrent_dropout = 0 if recurrent_dropout is None else int(
            recurrent_dropout.replace("Rd", ""))

        # Check if the dropout is valid
        if dropout < 0 or dropout > 100:
            raise ValueError("Dropout rate must be in the range [0, 100].")

        # Check if the recurrent dropout is valid
        if recurrent_dropout < 0 or recurrent_dropout > 100:
            raise ValueError(
                "Recurrent dropout rate must be in the range [0, 100].")

        n = int(n)

        # Check if the number of units is valid
        if n <= 0:
            raise ValueError(
                f"Invalid number of units {n} for LSTM layer {layer}.")

        lstm_params = {
            "units": n,
            "return_sequences": 's' in layer,
            "go_backwards": direction == 'r',
            "kernel_initializer": initializer,
            "dropout": dropout/100 if dropout > 0 else 0,
            "recurrent_dropout": recurrent_dropout / 100
            if recurrent_dropout > 0 else 0,
        }

        return tf.keras.layers.LSTM(**lstm_params)

    @ staticmethod
    def gru(layer: str,
            initializer: tf.keras.initializers.Initializer
            = tf.keras.initializers.GlorotNormal) \
            -> tf.keras.layers.GRU:
        """
        Generate a GRU layer based on a VGSL specification string.

        Parameters
        ----------
        layer : str
            VGSL specification for the GRU layer. Expected format:
            `G(f|r)[s]<n>[,D<rate>,Rd<rate>]`
            - `(f|r)`: Direction of GRU. 'f' for forward, 'r' for reversed
            - `[s]`: (Optional) Summarizes the output, outputting only the
            final step.
            - `<n>`: Number of outputs.
            - `D<rate>` : Dropout rate. Should be between 0 and 100.
            - `Rd<rate>`: Recurrent dropout rate. Should be between 0 and 100.
        initializer : tf.keras.initializers.Initializer, optional
            Initializer for the kernel weights. Defaults to GlorotNormal.

        Returns
        -------
        tf.keras.layers.GRU
            A GRU layer with the specified parameters.

        Raises
        ------
        ValueError
            If the provided VGSL spec string does not match the expected
            format.

        Examples
        --------
        >>> from vgslify.tensorflow.generator import VGSLModelGenerator
        >>> gru_layer = VGSLModelGeneratorl.gru("Gf64")
        >>> type(gru_layer)
        <class 'keras.src.layers.rnn.gru.GRU'>
        """

        # Extract direction, summarization, and units
        match = re.match(r'G([fr])(s?)(-?\d+),?(D-?\d+)?,?(Rd-?\d+)?$', layer)
        if not match:
            raise ValueError(
                f"GRU layer {layer} is of unexpected format. Expected "
                "format: G(f|r)[s]<n>[,D<rate>,Rd<rate>].")

        direction, summarize, n, dropout, recurrent_dropout = match.groups()
        dropout = 0 if dropout is None else int(dropout.replace('D', ""))
        recurrent_dropout = 0 if recurrent_dropout is None else int(
            recurrent_dropout.replace("Rd", ""))

        # Check if the dropout is valid
        if dropout < 0 or dropout > 100:
            raise ValueError("Dropout rate must be in the range [0, 100].")

        # Check if the recurrent dropout is valid
        if recurrent_dropout < 0 or recurrent_dropout > 100:
            raise ValueError(
                "Recurrent dropout rate must be in the range [0, 100].")

        # Convert n to integer
        n = int(n)

        # Check if the number of units is valid
        if n <= 0:
            raise ValueError(
                f"Invalid number of units {n} for GRU layer {layer}.")

        gru_params = {
            "units": n,
            "return_sequences": bool(summarize),
            "go_backwards": direction == 'r',
            "kernel_initializer": initializer,
            "dropout": dropout/100 if dropout > 0 else 0,
            "recurrent_dropout": recurrent_dropout/100
            if recurrent_dropout > 0 else 0
        }

        return tf.keras.layers.GRU(**gru_params)

    @ staticmethod
    def bidirectional(layer: str,
                      initializer: tf.keras.initializers.Initializer
                      = tf.keras.initializers.GlorotNormal) \
            -> tf.keras.layers.Bidirectional:
        """
        Generate a Bidirectional RNN layer based on a VGSL specification
        string.
        The method supports both LSTM and GRU layers for the bidirectional RNN.

        Parameters
        ----------
        layer : str
            VGSL specification for the Bidirectional layer. Expected format:
            `B(g|l)<n>[,D<rate>,Rd<rate>]`
            - `(g|l)`: Type of RNN layer. 'g' for GRU and 'l' for LSTM.
            - `<n>`: Number of units in the RNN layer.
            - `D<rate>` : Dropout rate. Should be between 0 and 100.
            - `Rd<rate>`: Recurrent dropout rate. Should be between 0 and 100.
        initializer : tf.keras.initializers.Initializer, optional
            Initializer for the kernel weights. Defaults to GlorotNormal.

        Returns
        -------
        tf.keras.layers.Bidirectional
            A Bidirectional RNN layer with the specified parameters.

        Raises
        ------
        ValueError
            If the provided VGSL spec string does not match the expected
            format.

        Examples
        --------
        >>> from vgslify.tensorflow.generator import VGSLModelGenerator
        >>> bidirectional_layer = VGSLModelGenerator.bidirectional("Bl256")
        >>> type(bidirectional_layer)
        <class ''keras.src.layers.rnn.bidirectional.Bidirectional>
        >>> type(bidirectional_layer.layer)
        <class 'keras.src.layers.rnn.lstm.LSTM'>

        Notes
        -----
        The Bidirectional layer wraps an RNN layer (either LSTM or GRU) and
        runs it in both forward and backward directions.
        """

        # Extract layer type and units
        match = re.match(r'B([gl])(-?\d+),?(D-?\d+)?,?(Rd-?\d+)?$', layer)
        if not match:
            raise ValueError(f"Layer {layer} is of unexpected format. "
                             "Expected format: B(g|l)<n>[,D<rate>,Rd<rate>] "
                             "where 'g' stands for GRU, 'l' stands for LSTM, "
                             "'n' is the number of units, 'rate' is the "
                             "(recurrent) dropout rate.")

        layer_type, units, dropout, recurrent_dropout = match.groups()
        dropout = 0 if dropout is None else int(dropout.replace('D', ""))
        recurrent_dropout = 0 if recurrent_dropout is None else int(
            recurrent_dropout.replace("Rd", ""))

        # Check if the dropout is valid
        if dropout < 0 or dropout > 100:
            raise ValueError("Dropout rate must be in the range [0, 100].")

        # Check if the recurrent dropout is valid
        if recurrent_dropout < 0 or recurrent_dropout > 100:
            raise ValueError(
                "Recurrent dropout rate must be in the range [0, 100].")

        units = int(units)

        # Check if the number of units is valid
        if units <= 0:
            raise ValueError(
                f"Invalid number of units {units} for layer {layer}.")

        # Determine the RNN layer type
        rnn_layer = tf.keras.layers.LSTM if layer_type == 'l' else \
            tf.keras.layers.GRU

        rnn_params = {
            "units": units,
            "return_sequences": True,
            "kernel_initializer": initializer,
            "dropout": dropout/100 if dropout > 0 else 0,
            "recurrent_dropout": recurrent_dropout/100
            if recurrent_dropout > 0 else 0
        }

        return tf.keras.layers.Bidirectional(rnn_layer(**rnn_params),
                                             merge_mode='concat')

    @ staticmethod
    def dropout(layer: str) -> tf.keras.layers.Dropout:
        """
        Generate a Dropout layer based on a VGSL specification string.

        Parameters
        ----------
        layer : str
            VGSL specification for the Dropout layer. Expected format:
            `D<rate>`
            - `<rate>`: Dropout percentage (0-100).

        Returns
        -------
        tf.keras.layers.Dropout
            A Dropout layer with the specified dropout rate.

        Raises
        ------
        ValueError
            If the layer format is unexpected or if the specified dropout rate
            is not in range [0, 100].

        Examples
        --------
        >>> from vgslify.tensorflow.generator import VGSLModelGenerator
        >>> dropout_layer = VGSLModelGenerator.dropout("Do50")
        >>> type(dropout_layer)
        <class 'keras.src.layers.regularization.dropout.Dropout'>
        """

        # Validate layer format and extract dropout rate
        match = re.match(r'D(-?\d+)$', layer)
        if not match:
            raise ValueError(
                f"Layer {layer} is of unexpected format. Expected format: "
                "D<rate> where rate is between 0 and 100.")

        dropout_rate = int(match.group(1))

        # Validate dropout rate
        if dropout_rate < 0 or dropout_rate > 100:
            raise ValueError("Dropout rate must be in the range [0, 100].")

        return tf.keras.layers.Dropout(dropout_rate / 100)

    def get_output_layer(self,
                         layer: str,
                         output_classes: int = None) -> tf.keras.layers.Dense:
        """
        Generate an output layer based on a VGSL specification string.

        Parameters
        ----------
        layer : str
            VGSL specification for the output layer. Expected format:
            `O(2|1|0)(l|s)<n>`
            - `(2|1|0)`: Dimensionality of the output.
            - `(l|s)`: Non-linearity type.
            - `<n>`: Number of output classes.
        output_classes : int
            Number of output classes to overwrite the classes defined in the
            VGSL string.

        Returns
        -------
        tf.keras.layers.Dense
            An output layer with the specified parameters.

        Raises
        ------
        ValueError
            If the output layer type specified is not supported or if an
            unsupported linearity is specified.

        Examples
        --------
        >>> from vgslify.tensorflow.generator import VGSLModelGenerator
        >>> vgsl_gn = VGSLModelGenerator("Some VGSL string")
        >>> output_layer = vgsl_gn.get_output_layer("O1s10", 10)
        >>> type(output_layer)
        <class 'keras.src.layers.core.dense.Dense'>
        """

        # Validate layer format
        match = re.match(r'O([210])([a-z])(\d+)$', layer)
        if not match:
            raise ValueError(
                f"Layer {layer} is of unexpected format. Expected format: "
                "O[210](l|s)<n>.")

        dimensionality, linearity, classes = match.groups()
        classes = int(classes)

        # Handle potential mismatch in specified classes and provided
        # output_classes
        if output_classes and classes != output_classes:
            logger.warning(
                "Overwriting output classes from input string. Was: %s, now: "
                "%s", classes, output_classes)
            classes = output_classes
            self.selected_model_vgsl_spec[-1] = (f"O{dimensionality}"
                                                 f"{linearity}{classes}")

        if linearity == "s":
            return tf.keras.layers.Dense(classes,
                                         activation='softmax',
                                         kernel_initializer=self._initializer)
        elif linearity == "l":
            return tf.keras.layers.Dense(classes,
                                         activation='linear',
                                         kernel_initializer=self._initializer)
        else:
            raise ValueError(
                f"Output layer linearity {linearity} is not supported.")
