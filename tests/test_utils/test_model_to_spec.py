from unittest import mock

import pytest

from vgslify.utils import model_to_spec

# Mocking TensorFlow imports to create a test case
try:
    import tensorflow as tf
except ImportError:
    tf = None


def test_model_to_spec_tensorflow():
    if not tf:
        pytest.skip("TensorFlow is not available")

    # Create a simple TensorFlow model
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    # Call the model_to_spec function
    vgsl_spec = model_to_spec(model)

    # Expected VGSL spec string
    expected_spec = "None,32,32,3 Cr3,3,64 Mp2,2,2,2 O1s10"

    # Verify the spec matches the expected output
    assert vgsl_spec == expected_spec


def test_model_to_spec_unsupported_model():
    # Create a dummy model that is neither TensorFlow nor PyTorch
    class UnsupportedModel:
        pass

    unsupported_model = UnsupportedModel()

    # Call the function and check that a ValueError is raised
    with pytest.raises(ValueError, match="Unsupported model type"):
        model_to_spec(unsupported_model)


def test_model_to_spec_no_tensorflow_installed():
    # Mock TensorFlow as not available and ensure the function doesn't fail
    with mock.patch("vgslify.utils.model_to_spec.tf", None):
        with pytest.raises(ValueError, match="Unsupported model type"):
            model_to_spec("Not a model")  # This should raise a ValueError
