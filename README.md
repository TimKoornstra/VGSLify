# VGSLify: Variable-size Graph Specification for TensorFlow & PyTorch

![PyPI](https://img.shields.io/pypi/v/vgslify)
![Downloads](https://pepy.tech/badge/vgslify)
![License](https://img.shields.io/pypi/l/vgslify)

VGSLify simplifies defining, training, and interpreting deep learning models using the Variable-size Graph Specification Language (VGSL). Inspired by [Tesseract's VGSL specs](https://tesseract-ocr.github.io/tessdoc/tess4/VGSLSpecs.html), VGSLify enhances and streamlines the process for both TensorFlow and PyTorch.

## Table of Contents

- [Installation](#installation)
- [How VGSL Works](#how-vgsl-works)
- [Quick Start](#quick-start)
    - [Generating a Model with VGSLify](#generating-a-model-with-vgslify)
    - [Creating Individual Layers with VGSLify](#creating-individual-layers-with-vgslify)
    - [Converting Models to VGSL](#converting-models-to-vgsl)
- [Additional Documentation](#additional-documentation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)


## Installation

### Basic Installation

To install VGSLify without any deep learning backend, run:

```bash
pip install vgslify
```

This installs only the core functionalities of VGSLify without `torch` or `tensorflow`.

### Installing with a Specific Backend

VGSLify supports both TensorFlow and PyTorch. You can install it with the required backend:

```bash
# For TensorFlow (latest compatible version)
pip install vgslify[tensorflow]

# For PyTorch (latest compatible version)
pip install vgslify[torch]
```

By default, this will install:

- `tensorflow` (latest stable version)
- `torch` (latest stable version)

### Controlling Backend Versions

If you need a specific version of `torch` or `tensorflow`, install VGSLify first and then manually install the backend:

```bash
pip install vgslify
pip install torch==2.1.0      # Example for PyTorch
pip install tensorflow==2.14  # Example for TensorFlow
```

Alternatively, you can specify the version during installation:

```bash
pip install vgslify[torch] torch==2.1.0
pip install vgslify[tensorflow] tensorflow==2.14
```

⚠ **Note**: If a different version of torch or tensorflow is already installed, pip may not downgrade it automatically. Use `--force-reinstall` or `--upgrade` if necessary:

```bash
pip install --upgrade --force-reinstall torch==2.1.0
```

### Verify installation

To check that VGSLify is installed correctly, run:

```bash
python -c "import vgslify; print(vgslify.__version__)"
```

## How VGSL Works

VGSL uses concise strings to define model architectures. For example:

```vgsl
None,None,64,1 Cr3,3,32 Mp2,2 Cr3,3,64 Mp2,2 Rc3 Fr64 D20 Lrs128 D20 Lrs64 D20 Fs92
```

Each part represents a layer: input, convolution, pooling, reshaping, fully connected, LSTM, and output. VGSL allows specifying activation functions for customization.


## Quick Start

### Generating a Model with VGSLify

```python
from vgslify import VGSLModelGenerator

# Define the VGSL specification
vgsl_spec = "None,None,64,1 Cr3,3,32 Mp2,2 Fs92"

# Choose backend: "tensorflow", "torch", or "auto" (defaults to whichever is available)
vgsl_gn = VGSLModelGenerator(backend="tensorflow") 
model = vgsl_gn.generate_model(vgsl_spec, model_name="MyModel")
model.summary()


vgsl_gn = VGSLModelGenerator(backend="torch") # Switch to PyTorch
model = vgsl_gn.generate_model(vgsl_spec, model_name="MyTorchModel")
print(model)


```

### Creating Individual Layers with VGSLify

```python
from vgslify import VGSLModelGenerator

vgsl_gn = VGSLModelGenerator(backend="tensorflow")
conv2d_layer = vgsl_gn.construct_layer("Cr3,3,64")

# Integrate into an existing model:
# model = tf.keras.Sequential()
# model.add(conv2d_layer) # ...

# Example with generate_history:
history = vgsl_gn.generate_history("None,None,64,1 Cr3,3,32 Mp2,2 Fs92")
for layer in history:
    print(layer)
```


### Converting Models to VGSL

```python
from vgslify import model_to_spec
import tensorflow as tf
# Or import torch.nn as nn

# TensorFlow example:
model = tf.keras.models.load_model("path_to_your_model.keras") # If loading from file

# PyTorch example:
# model = MyPyTorchModel() # Assuming MyPyTorchModel is defined elsewhere


vgsl_spec_string = model_to_spec(model)
print(vgsl_spec_string)
```

**Note:** Flatten/Reshape layers might require manual input shape adjustment in the generated VGSL.


## Additional Documentation

See the [VGSL Documentation](https://timkoornstra.github.io/VGSLify) for more details on supported layers and their specifications.


## Contributing

Contributions are welcome!  Fork the repository, set up your environment, make changes, and submit a pull request. Create issues for bugs or suggestions.


## License

MIT License. See [LICENSE](LICENSE) file.


## Acknowledgements

Thanks to the creators and contributors of the original VGSL specification.
