import pytest
import tensorflow as tf

from vgslify.model_parsers.tensorflow import tf_to_spec


def test_input_layer():
    # Create a simple model with only an input layer
    inputs = tf.keras.Input(shape=(32, 32, 3))
    model = tf.keras.Model(inputs=inputs, outputs=inputs)

    vgsl_spec = tf_to_spec(model)

    assert vgsl_spec == "None,32,32,3"


def test_conv2d_layer():
    # Create a simple model with Conv2D layer
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", strides=(2, 2))(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=x)

    vgsl_spec = tf_to_spec(model)

    assert vgsl_spec == "None,32,32,3 Cr3,3,64,2,2"


def test_dense_layer():
    # Create a simple model with a dense layer
    inputs = tf.keras.Input(shape=(32,))
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=x)

    vgsl_spec = tf_to_spec(model)

    assert vgsl_spec == "None,32 O1r64"


def test_maxpooling2d_layer():
    # Create a simple model with MaxPooling2D layer
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=x)

    vgsl_spec = tf_to_spec(model)

    assert vgsl_spec == "None,32,32,3 Mp2,2,2,2"


def test_lstm_layer():
    # Create a simple model with an LSTM layer
    inputs = tf.keras.Input(shape=(32, 10))
    x = tf.keras.layers.LSTM(
        64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1
    )(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=x)

    vgsl_spec = tf_to_spec(model)

    assert vgsl_spec == "None,32,10 Lfs64,D20,Rd10"


def test_bidirectional_layer():
    # Create a model with a Bidirectional LSTM layer
    inputs = tf.keras.Input(shape=(32, 10))
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.1)
    )(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=x)

    vgsl_spec = tf_to_spec(model)

    assert vgsl_spec == "None,32,10 Bl64,D20,Rd10"


def test_dropout_layer():
    # Create a simple model with Dropout layer
    inputs = tf.keras.Input(shape=(32,))
    x = tf.keras.layers.Dropout(0.5)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=x)

    vgsl_spec = tf_to_spec(model)

    assert vgsl_spec == "None,32 D50"


def test_reshape_layer():
    # Create a simple model with Reshape layer
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Reshape((1024, 3))(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=x)

    vgsl_spec = tf_to_spec(model)

    assert vgsl_spec == "None,32,32,3 Rc"


def test_batchnorm_layer():
    # Create a simple model with BatchNormalization layer
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.BatchNormalization()(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=x)

    vgsl_spec = tf_to_spec(model)

    assert vgsl_spec == "None,32,32,3 Bn"


def test_unsupported_layer():
    # Create a model with an unsupported layer type
    inputs = tf.keras.Input(shape=(32, 32, 32, 3))
    x = tf.keras.layers.Conv3D(64, (3, 3, 3))(inputs)  # Conv3D is not supported
    model = tf.keras.Model(inputs=inputs, outputs=x)

    with pytest.raises(ValueError):
        tf_to_spec(model)
