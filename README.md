# VGSLify: Variable-size Graph Specification for TensorFlow & PyTorch

VGSLify is a powerful toolkit designed to simplify the process of defining, training, and interpreting deep learning models using the Variable-size Graph Specification Language (VGSL). Inspired by [Tesseract's VGSL specs](https://tesseract-ocr.github.io/tessdoc/tess4/VGSLSpecs.html), VGSLify introduces a set of enhancements and provides a streamlined interface for both TensorFlow and, in future releases, PyTorch.

## Table of Contents

- [Installation](#installation)
- [How VGSL Works](#how-vgsl-works)
- [Quick Start](#quick-start)
- [Supported Layers and Their Specifications](#supported-layers-and-their-specifications)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation

Installing VGSLify is simple and straightforward. Follow the steps below to get started:

### 1. Prerequisites

Before installing VGSLify, ensure you have Python 3.8 or newer installed. You can check your Python version with the following command:

```bash
python --version
```

If you don't have Python installed, download it from [python.org](https://www.python.org/downloads/) or use your system's package manager.

### 2. Using pip

The easiest way to install VGSLify is via `pip`. Run the following command to install the latest version of VGSLify:

```bash 
pip install vgslify
```

### 3. From Source

If you want to install the development version or a modified version of VGSLify, you can do so from source. First, clone the repository or download the source code. Then navigate to the directory and run:

```bash 
pip install .
```

### 4. Installing a backend

VGSLify allows you to choose your preferred deep learning framework. You must install the backend separately according to your needs:

- **TensorFlow**: To use VGSLify with TensorFlow, install TensorFlow using the following command:

  ```bash
  pip install tensorflow
  ```

- **PyTorch**: For PyTorch support, install PyTorch using the following command:

  ```bash
  pip install torch
  ```

You can install both backends if you want to switch between TensorFlow and PyTorch.

### 5. Verifying Installation

After installation, you can verify that VGSLify was installed correctly by importing it in Python:

```bash
python -c "import vgslify; print(vgslify.__version__)"
```

This should print the installed version of VGSLify without any errors.

Remember to periodically update VGSLify to benefit from new features and bug fixes. You can update it using `pip`:

```bash
pip install --upgrade vgslify
```

## How VGSL works

VGSL operates through short definition strings. For instance:

`None,None,64,1 Cr3,3,32 Mp2,2,2,2 Cr3,3,64 Mp2,2,2,2 Rc Fc64 D20 Lrs128 D20 Lrs64 D20 Fs192`

In this example, the string defines a neural network with input layers, convolutional layers, pooling, reshaping, fully connected layers, LSTM and output layers. Each segment of the string corresponds to a specific layer or operation in the neural network. Moreover, VGSL provides the flexibility to specify the type of activation function for certain layers, enhancing customization.

## Quick Start

Using VGSLify, you can rapidly prototype TensorFlow models using the Variable-size Graph Specification Language (VGSL). Here's a simple example to get you started:

### Generating a Model with VGSLify

1. **Import the Necessary Module**:
   
   First, import the `VGSLModelGenerator` class from VGSLify's core module.

   ```python
   from vgslify.generator import VGSLModelGenerator
   ```

2. **Initialize the Model Generator**:

    VGSLify is a "BYOB" (Bring Your Own Backend) package, meaning you can choose the backend that suits your needs. Options include:

    - "tensorflow": Use TensorFlow for model generation.
    - "torch": Use PyTorch as the backend (currently under development).
    - "auto": VGSLify will automatically detect the available backend and use TensorFlow by default if both are available.

    For example, to explicitly use TensorFlow as the backend:

    ```python
    vgsl_gn = VGSLModelGenerator(backend="tensorflow")
    ```

    Alternatively, let VGSLify automatically detect the backend:

    ```python
    vgsl_gn = VGSLModelGenerator(backend="auto")
    ```

    or

    ```python
    vgsl_gn = VGSLModelGenerator()
    ```

3. **Build and View the Model**:

   After initializing the model generator, you can pass the VGSL specification string to the `generate_model()` method to build the model. Here's an example where we create a TensorFlow model with a convolutional layer, max-pooling layer, and an output softmax layer:

   ```python
   vgsl_spec = "None,64,None,1 Cr3,3,32 Mp2,2,2,2 O1s92"
   model = vgsl_gn.generate_model(vgsl_spec)
   model.summary()
   ```

This example demonstrates the flexibility and simplicity of creating a TensorFlow model using VGSLify. The VGSL spec string allows you to define complex architectures with ease, and the `VGSLModelGenerator` handles the construction of the model. You can use the same generator instance to create multiple models by passing different VGSL spec strings.

For more details on the VGSL spec syntax and supported layers, refer to the "[Supported Layers and Their Specifications](#supported-layers-and-their-specifications)" section.

### Creating Individual Layers with VGSLify

In addition to creating complete models, VGSLify allows you to generate individual layers from a VGSL specification string. This is particularly useful when you want to integrate a VGSL-defined layer into an existing model or when you wish to experiment with individual components without having to construct a full model.

1. **Import the Necessary Module**:

   Just like before, you'll need to import the `VGSLModelGenerator` class.

   ```python
   from vgslify.generator import VGSLModelGenerator
   ```

2. **Initialize the Model Generator**:

    Like when building models, you first initialize the model generator and select your backend. In this example, we'll use TensorFlow:
    
    ```python
    vgsl_gn = VGSLModelGenerator(backend="tensorflow")
    ```

3. **Generate an Individual Layer**:

    You can now use the `construct_layer()` method of VGSLModelGenerator to create individual layers directly from a VGSL spec string.

    Here’s an example where we create a convolutional layer using the VGSL spec:

    ```python
    vgsl_spec_for_conv2d = "Cr3,3,64"
    conv2d_layer = vgsl_gn.construct_layer(vgsl_spec_for_conv2d)
    ```

    These layers can be added to an existing TensorFlow model just like any other `tf.keras.layers.Layer`.

4. **Generate the History of Layers**:

    If you want to inspect the sequence of layers from a VGSL specification without constructing a full model, you can use the `generate_history()` method. This method parses the VGSL spec and generates the layers in sequence, allowing you to experiment with or debug the architecture.

    Here’s how you can use it:

    ```python
    vgsl_spec = "None,64,None,1 Cr3,3,32 Mp2,2,2,2 O1s92"
    history = vgsl_gn.generate_history(vgsl_spec)

    for layer in history:
        print(layer)
    ```

    This will print each layer generated by the VGSL spec, giving you insight into the layer sequence without building or chaining them together.

### Converting TensorFlow Models to VGSL [WIP]

Once you have trained a TensorFlow model, you might want to convert it back into a VGSL spec string for various purposes. VGSLify provides an easy way to do this:

1. **Load Your TensorFlow Model**:

   If you've saved your trained Keras model to a file, first load it using TensorFlow's `load_model` method:

   ```python
   import tensorflow as tf
   model = tf.keras.models.load_model("path_to_your_model.keras")
   ``` 
   
2. **Convert to VGSL Spec String**:

   With your model loaded, use the `model_to_spec` function from VGSLify to convert the model into a VGSL spec string:

   ```python
   from vgslify.utils import model_to_spec
   vgsl_spec_string = model_to_spec(model)
   print(vgsl_spec_string)
   ```

This provides a concise representation of your model's architecture in VGSL format. Please note that while VGSLify aims to support a wide variety of architectures, there might be specific TensorFlow layers or configurations that are not supported. In such cases, a `ValueError` will be raised.

## Supported Layers and Their Specifications

VGSLify supports a wide range of layers, including convolutional, pooling, LSTM, GRU, and more. For a detailed list and specifications of all supported layers, please refer to the [Supported Layers Documentation](https://timkoornstra.github.io/VGSLify/supported_layers.html).

## Future Work

### PyTorch Model Support

While the current version of VGSLify supports TensorFlow models, I recognize the growing popularity and capabilities of PyTorch in the deep learning community. In an upcoming release, I aim to expand VGSLify's capabilities to generate and parse PyTorch models using VGSL spec, providing a unified experience across both major deep learning frameworks.

### Custom Layer Support

To make VGSLify even more versatile, I'm working on a feature that will allow users to define and integrate custom layers into the VGSL specification. This enhancement will empower users to seamlessly integrate specialized layers or proprietary architectures into their VGSL-defined models, further bridging the gap between rapid prototyping and production-ready models.

### Spec to Code

The Spec to Code feature will generate Python code from a VGSL spec string, allowing users to easily view, customize, and integrate their model architectures into existing codebases. This enhances transparency and flexibility, giving users full control over the generated model, allowing them to optimize and tune it as needed.

### Model to Code

The Model to Code feature will convert trained models back into maintainable Python code, allowing easy integration, modification, and deployment. It will support both TensorFlow and PyTorch, generating framework-specific code based on the model’s architecture.


## Contributing

I warmly welcome contributions to VGSLify! Whether you're fixing bugs, adding new features, or improving the documentation, your efforts will make VGSLify better for everyone.

### How to Contribute:

1. **Fork the Repository**: Start by forking the VGSLify repository.
2. **Set Up Your Development Environment**: Clone your fork to your local machine and set up the development environment.
3. **Make Your Changes**: Implement your changes, improvements, or fixes.
4. **Submit a Pull Request**: Once you're done with your changes, push them to your fork and submit a pull request. Please provide a clear description of the changes and their purpose.
5. **Create Issues**: If you find bugs or want to suggest improvements, please create an issue in the repository.

Please ensure that your contributions adhere to our coding standards and practices. Your efforts will help VGSLify grow and thrive!

## License

VGSLify is open-source software and is licensed under the MIT License. This means you can freely use, modify, and distribute it, provided you include the original copyright notice. For more details, you can refer to the [LICENSE](LICENSE) file in the repository.

## Acknowledgements

A special thank you to:

- The creators and contributors of the original VGSL specification, which inspired and laid the groundwork for VGSLify.

