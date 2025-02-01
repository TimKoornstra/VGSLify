import pytest

from vgslify.core.config import (
    Conv2DConfig,
    DenseConfig,
    DropoutConfig,
    InputConfig,
    OutputLayerConfig,
    Pooling2DConfig,
    ReshapeConfig,
    RNNConfig,
)
from vgslify.core.spec_parser import (
    parse_activation_spec,
    parse_conv2d_spec,
    parse_dense_spec,
    parse_dropout_spec,
    parse_input_spec,
    parse_output_layer_spec,
    parse_pooling2d_spec,
    parse_reshape_spec,
    parse_rnn_spec,
    parse_spec,
)


# Test parse_spec
def test_parse_spec():
    spec = "None,64,64,3 Cr3,3,64 Mp2,2,2,2"
    result = parse_spec(spec)
    assert result == ["None,64,64,3", "Cr3,3,64", "Mp2,2,2,2"]


# Test parse_conv2d_spec
def test_parse_conv2d_spec():
    config = parse_conv2d_spec("Cr3,3,64")
    assert isinstance(config, Conv2DConfig)
    assert config.activation == "relu"
    assert config.kernel_size == (3, 3)
    assert config.strides == (1, 1)
    assert config.filters == 64


def test_parse_conv2d_spec_with_strides():
    config = parse_conv2d_spec("Cr3,3,2,2,64")
    assert config.strides == (2, 2)


def test_invalid_conv2d_spec_too_few_params():
    with pytest.raises(ValueError):
        parse_conv2d_spec("Cr3,64")


def test_invalid_conv2d_spec_too_many_params():
    with pytest.raises(ValueError):
        parse_conv2d_spec("Cr3,3,1,1,64,128")


# Test parse_pooling2d_spec
def test_parse_pooling2d_spec():
    config = parse_pooling2d_spec("Mp2,2,2,2")
    assert isinstance(config, Pooling2DConfig)
    assert config.pool_size == (2, 2)
    assert config.strides == (2, 2)


def test_invalid_pooling2d_spec():
    with pytest.raises(ValueError):
        parse_pooling2d_spec("Mp2,2")


# Test parse_dense_spec
def test_parse_dense_spec():
    config = parse_dense_spec("Fr64")
    assert isinstance(config, DenseConfig)
    assert config.activation == "relu"
    assert config.units == 64


def test_invalid_dense_spec():
    with pytest.raises(ValueError):
        parse_dense_spec("Fx64")


# Test parse_rnn_spec
def test_parse_rnn_spec():
    config = parse_rnn_spec("Lf64,D50,Rd25")
    assert isinstance(config, RNNConfig)
    assert config.units == 64
    assert config.dropout == 0.5
    assert config.recurrent_dropout == 0.25


def test_invalid_rnn_spec():
    with pytest.raises(ValueError):
        parse_rnn_spec("Lx64,D50")


# Test parse_dropout_spec
def test_parse_dropout_spec():
    config = parse_dropout_spec("D50")
    assert isinstance(config, DropoutConfig)
    assert config.rate == 0.5


def test_invalid_dropout_spec():
    with pytest.raises(ValueError):
        parse_dropout_spec("D-10")


# Test parse_output_layer_spec
def test_parse_output_layer_spec():
    config = parse_output_layer_spec("O1s10")
    assert isinstance(config, OutputLayerConfig)
    assert config.dimensionality == 1
    assert config.activation == "softmax"
    assert config.units == 10


def test_invalid_output_layer_spec():
    with pytest.raises(ValueError):
        parse_output_layer_spec("O3s10")


# Test parse_activation_spec
def test_parse_activation_spec():
    activation = parse_activation_spec("Ar")
    assert activation == "relu"


def test_invalid_activation_spec():
    with pytest.raises(ValueError):
        parse_activation_spec("Ax")


# Test parse_reshape_spec
def test_parse_reshape_spec():
    config = parse_reshape_spec("R64,64,3")
    assert isinstance(config, ReshapeConfig)
    assert config.target_shape == (64, 64, 3)


def test_invalid_reshape_spec():
    with pytest.raises(ValueError):
        parse_reshape_spec("R64,64")


# Test parse_input_spec
def test_parse_input_spec():
    config = parse_input_spec("None,64,64,3")
    assert isinstance(config, InputConfig)
    assert config.height == 64
    assert config.width == 64
    assert config.channels == 3


def test_invalid_input_spec():
    with pytest.raises(ValueError):
        parse_input_spec("None")
