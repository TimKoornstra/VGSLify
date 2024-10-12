# Imports

# > Third-Party Dependencies
import torch.nn as nn


class Reshape(nn.Module):
    """
    Custom PyTorch Reshape layer. To be used in the VGSL spec.
    """

    def __init__(self, *args):
        """
        Initialize the Reshape layer.

        Parameters
        ----------
        *args : int
            Dimensions of the target shape excluding the batch size.
        """
        super().__init__()
        self.target_shape = args

    def forward(self, x):
        """
        Forward pass for reshaping the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to reshape.

        Returns
        -------
        torch.Tensor
            Reshaped tensor.
        """
        return x.view(x.size(0), *self.target_shape)
