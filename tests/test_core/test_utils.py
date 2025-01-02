import pytest

from vgslify.core.utils import get_activation_function


def test_get_activation_function_valid_inputs():
    # Test valid inputs
    assert get_activation_function("s") == "softmax"
    assert get_activation_function("t") == "tanh"
    assert get_activation_function("r") == "relu"
    assert get_activation_function("l") == "linear"
    assert get_activation_function("m") == "sigmoid"


def test_get_activation_function_invalid_input():
    # Test invalid input
    with pytest.raises(ValueError, match="Invalid activation character 'x'"):
        get_activation_function("x")


def test_get_activation_function_edge_cases():
    # Test edge cases such as empty string or invalid types
    with pytest.raises(ValueError, match="Invalid activation character ''"):
        get_activation_function("")

    with pytest.raises(ValueError, match="Invalid activation character '1'"):
        get_activation_function("1")
