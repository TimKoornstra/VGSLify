import pytest
import tensorflow as tf

from vgslify.tensorflow.layers import TensorFlowLayerFactory


# Test conv2d
def test_conv2d():
    conv_layer = TensorFlowLayerFactory.conv2d("Cr3,3,64")
    assert isinstance(conv_layer, tf.keras.layers.Conv2D)
    assert conv_layer.filters == 64
    assert conv_layer.kernel_size == (3, 3)
    assert conv_layer.strides == (1, 1)


def test_invalid_conv2d():
    with pytest.raises(ValueError):
        TensorFlowLayerFactory.conv2d("Cr3,64")


# Test maxpooling2d
def test_maxpooling2d():
    pool_layer = TensorFlowLayerFactory.maxpooling2d("Mp2,2,2,2")
    assert isinstance(pool_layer, tf.keras.layers.MaxPooling2D)
    assert pool_layer.pool_size == (2, 2)
    assert pool_layer.strides == (2, 2)


def test_invalid_maxpooling2d():
    with pytest.raises(ValueError):
        TensorFlowLayerFactory.maxpooling2d("Mp2,2")


# Test dense
def test_dense():
    dense_layer = TensorFlowLayerFactory.dense("Fr64")
    assert isinstance(dense_layer, tf.keras.layers.Dense)
    assert dense_layer.units == 64
    assert dense_layer.activation.__name__ == "relu"


def test_invalid_dense():
    with pytest.raises(ValueError):
        TensorFlowLayerFactory.dense("Fx64")


# Test lstm
def test_lstm():
    lstm_layer = TensorFlowLayerFactory.lstm("Lr64,D20,Rd10")
    assert isinstance(lstm_layer, tf.keras.layers.LSTM)
    assert lstm_layer.units == 64
    assert lstm_layer.dropout == 0.2
    assert lstm_layer.recurrent_dropout == 0.1
    assert lstm_layer.go_backwards is True


def test_invalid_lstm():
    with pytest.raises(ValueError):
        TensorFlowLayerFactory.lstm("Lx64")


# Test bidirectional
def test_bidirectional():
    bidir_layer = TensorFlowLayerFactory.bidirectional("Bl64,D20,Rd10")
    assert isinstance(bidir_layer, tf.keras.layers.Bidirectional)
    assert isinstance(bidir_layer.forward_layer, tf.keras.layers.LSTM)
    assert bidir_layer.forward_layer.units == 64


def test_invalid_bidirectional():
    with pytest.raises(ValueError):
        TensorFlowLayerFactory.bidirectional("Bx64")


# Test dropout
def test_dropout():
    dropout_layer = TensorFlowLayerFactory.dropout("D50")
    assert isinstance(dropout_layer, tf.keras.layers.Dropout)
    assert dropout_layer.rate == 0.5


def test_invalid_dropout():
    with pytest.raises(ValueError):
        TensorFlowLayerFactory.dropout("D200")


# Test output layer
def test_output():
    output_layer = TensorFlowLayerFactory.output("O1s10")
    assert isinstance(output_layer, tf.keras.layers.Dense)
    assert output_layer.units == 10
    assert output_layer.activation.__name__ == "softmax"


def test_invalid_output():
    with pytest.raises(ValueError):
        TensorFlowLayerFactory.output("O3s10")


# Test input
def test_input():
    input_layer = TensorFlowLayerFactory.input("None,64,64,3")
    # Special case for input layer
    assert isinstance(input_layer, tf.keras.KerasTensor)
    assert input_layer.shape == (None, 64, 64, 3)


def test_invalid_input():
    with pytest.raises(ValueError):
        TensorFlowLayerFactory.input("None")


# Test flatten
def test_flatten():
    flatten_layer = TensorFlowLayerFactory.flatten("Flt")
    assert isinstance(flatten_layer, tf.keras.layers.Flatten)


def test_invalid_flatten():
    with pytest.raises(ValueError):
        TensorFlowLayerFactory.flatten("InvalidSpec")


# Test batchnorm
def test_batchnorm():
    batchnorm_layer = TensorFlowLayerFactory.batchnorm("Bn")
    assert isinstance(batchnorm_layer, tf.keras.layers.BatchNormalization)


def test_invalid_batchnorm():
    with pytest.raises(ValueError):
        TensorFlowLayerFactory.batchnorm("InvalidBn")


# Test build_final_model
def test_build_final_model():
    inputs = TensorFlowLayerFactory.input("None,32,32,3")
    outputs = TensorFlowLayerFactory.conv2d("Cr3,3,64")(inputs)
    model = TensorFlowLayerFactory.build(inputs, outputs)
    assert isinstance(model, tf.keras.models.Model)
    assert model.input_shape == (None, 32, 32, 3)
    assert model.output_shape == (None, 32, 32, 64)
