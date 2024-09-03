from abc import ABC, abstractmethod


class LayerFactory(ABC):
    """
    Abstract base class for creating neural network layers from VGSL specifications.
    This class defines the interface that must be implemented by concrete factories
    for different frameworks (e.g., TensorFlow, PyTorch).

    All methods are static to allow direct layer creation without instantiating the factory.
    """

    @staticmethod
    @abstractmethod
    def conv2d(spec: str):
        """
        Create a Conv2D layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Conv2D layer.

        Returns
        -------
        Layer
            The created Conv2D layer.
        """
        pass

    @staticmethod
    @abstractmethod
    def maxpooling2d(spec: str):
        """
        Create a MaxPooling2D layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the MaxPooling2D layer.

        Returns
        -------
        Layer
            The created MaxPooling2D layer.
        """
        pass

    @staticmethod
    @abstractmethod
    def avgpool2d(spec: str):
        """
        Create an AvgPooling2D layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the AvgPooling2D layer.

        Returns
        -------
        Layer
            The created AvgPooling2D layer.
        """
        pass

    @staticmethod
    @abstractmethod
    def dense(spec: str):
        """
        Create a Dense (fully connected) layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Dense layer.

        Returns
        -------
        Layer
            The created Dense layer.
        """
        pass

    @staticmethod
    @abstractmethod
    def lstm(spec: str):
        """
        Create an LSTM layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the LSTM layer.

        Returns
        -------
        Layer
            The created LSTM layer.
        """
        pass

    @staticmethod
    @abstractmethod
    def gru(spec: str):
        """
        Create a GRU layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the GRU layer.

        Returns
        -------
        Layer
            The created GRU layer.
        """
        pass

    @staticmethod
    @abstractmethod
    def bidirectional(spec: str):
        """
        Create a Bidirectional RNN layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Bidirectional RNN layer.

        Returns
        -------
        Layer
            The created Bidirectional RNN layer.
        """
        pass

    @staticmethod
    @abstractmethod
    def dropout(spec: str):
        """
        Create a Dropout layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Dropout layer.

        Returns
        -------
        Layer
            The created Dropout layer.
        """
        pass

    @staticmethod
    @abstractmethod
    def output(spec: str):
        """
        Create the output layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the output layer.
        output_classes : int, optional
            The number of output classes to override the VGSL specification.

        Returns
        -------
        Layer
            The created output layer.
        """
        pass

    @staticmethod
    @abstractmethod
    def batchnorm(spec: str):
        """
        Create a BatchNormalization layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the BatchNormalization layer.

        Returns
        -------
        Layer
            The created BatchNormalization layer.
        """
        pass

    @staticmethod
    @abstractmethod
    def activation(spec: str):
        """
        Create an Activation layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Activation layer.

        Returns
        -------
        Layer
            The created Activation layer.
        """
        pass

    @staticmethod
    @abstractmethod
    def reshape(spec: str):
        """
        Create a Reshape layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Reshape layer.

        Returns
        -------
        Layer
            The created Reshape layer.
        """
        pass

    @staticmethod
    @abstractmethod
    def input(spec: str):
        """
        Create the input layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the input layer.

        Returns
        -------
        Layer
            The created input layer.
        """
        pass

    @staticmethod
    @abstractmethod
    def flatten(spec: str):
        """
        Create a Flatten layer based on the VGSL specification string.

        Parameters
        ----------
        spec : str
            The VGSL specification string for the Flatten layer.

        Returns
        -------
        Layer
            The created Flatten layer.
        """
        pass

    @staticmethod
    @abstractmethod
    def build_final_model(inputs, outputs, name):
        """
        Build the final model using the specified backend.

        Parameters
        ----------
        inputs : Layer
            The input layer of the model.
        outputs : Layer
            The output layer of the model.
        name : str
            The name of the model.

        Returns
        -------
        model
            The built model using the specified backend.
        """
        pass
