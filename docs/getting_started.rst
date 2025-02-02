Getting Started
===============

Overview
--------

VGSLify makes it incredibly simple to define, build, and train deep learning models using the Variable-size Graph Specification Language (VGSL). VGSL strings serve as compact representations of neural network architectures, allowing you to build models in a single line. 

VGSLify abstracts the complexity of backend-specific syntax, enabling seamless switching between TensorFlow and PyTorch. This flexibility allows you to focus on model architecture and training without worrying about framework-specific implementations.

What is a VGSL Specification?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A VGSL specification string concisely defines a neural network's architecture. The string encodes all layers, including input, convolutional layers, pooling, fully connected layers, and more. Each part of the string corresponds to a different component of the model.

For example, the following VGSL string defines a simple convolutional neural network:

``None,28,28,1 Cr3,3,32 Mp2,2 Rc2 Fr64 D20 Fs10``

This string represents a model with an input layer, a convolutional layer, a max pooling layer, a reshape layer, a dense (fully connected) layer, dropout, and an output layer. The model's structure is encoded entirely within this single line.

Key functionality of VGSLify includes:

- **Building models with a single line**: You can define complex architectures with a VGSL string, reducing the need for verbose code.
- **Switching between TensorFlow and PyTorch**: VGSLify supports both TensorFlow and PyTorch, allowing you to easily switch between backends.

Simple Example: Building a Model
--------------------------------

Let’s walk through building a simple deep learning model using VGSLify. 

1. **Import the VGSLModelGenerator**:

   The `VGSLModelGenerator` class is the core component for building models from VGSL strings. Begin by importing it:

   .. code-block:: python

      from vgslify import VGSLModelGenerator

2. **Define the VGSL Specification String**:

   The VGSL spec string encodes the structure of the model. In this example, we will define a simple convolutional neural network suitable for handling MNIST digit images (28x28 grayscale):

   .. code-block:: python

      vgsl_spec = "None,28,28,1 Cr3,3,32 Mp2,2 Rc2 Fr64 D20 Fs10"

3. **Build and View the Model**:

   Initialize the `VGSLModelGenerator` and use it to build the model based on the VGSL spec string:

   .. code-block:: python

      vgsl_gn = VGSLModelGenerator(backend="tensorflow")  # Set backend to TensorFlow
      model = vgsl_gn.generate_model(vgsl_spec)
      model.summary()  # View the model architecture

   This will generate the model and display a summary of its architecture, including all layers defined by the VGSL spec string.

Explanation of Layers
---------------------

Let’s break down the layers defined by the VGSL specification string in our example:

- **Input Layer**: ``None,28,28,1`` 
  - This defines the input shape of the model, which corresponds to grayscale images of size 28x28 pixels. The first dimension (`None`) allows for a variable batch size.
  
- **Convolutional Layer**: ``Cr3,3,32`` 
  - This adds a 2D convolutional layer with a 3x3 kernel and 32 output filters, using ReLU activation (`r` for ReLU).

- **MaxPooling Layer**: ``Mp2,2`` 
  - This reduces the spatial dimensions by applying 2x2 max pooling with a stride of 2x2, which downsamples the input by taking the maximum value over each 2x2 window.

- **Reshape Layer**: ``Rc2`` 
  - Reshapes the output from the previous layer, collapsing the spatial dimensions into a single vector suitable for fully connected layers.

- **Fully Connected Layer**: ``Fr64`` 
  - Adds a fully connected layer (dense layer) with 64 units.

- **Dropout Layer**: ``D20`` 
  - Applies dropout with a 20% rate to prevent overfitting by randomly setting a portion of the inputs to zero during training.

- **Output Layer**: ``Fs10`` 
  - Represents the output layer with 10 units (for 10 classes, such as the digits in MNIST) using softmax activation.

This VGSL string provides a concise, human-readable format for specifying complex model architectures. VGSLify automatically translates this specification into a deep learning model that can be trained using TensorFlow or PyTorch.

Quick Start with Custom Layers (Advanced)
-----------------------------------------
You can also extend VGSLify by registering custom layer builders. For example, if you want to add a custom “Swish” activation layer in TensorFlow:

.. code-block:: python

   from vgslify.tensorflow import register_custom_layer
   import tensorflow as tf

   @register_custom_layer("Xsw")
   def build_swish_layer(factory, spec):
       # spec example: "Xsw" triggers this custom layer
       return tf.keras.layers.Lambda(lambda x: x * tf.keras.activations.sigmoid(x))

   # Use the custom prefix in your VGSL spec
   vgsl_spec = "None,28,28,1 Cr3,3,32 Xsw Mp2,2 Rc2 Fr64 D20 Fs10"
   vgsl_gn = VGSLModelGenerator(backend="tensorflow")
   model = vgsl_gn.generate_model(vgsl_spec)
   model.summary()

Now you have a model with a custom activation layer integrated seamlessly!

Next Steps
----------

Once you’ve built and explored a basic model, you can dive deeper into VGSLify's capabilities. Follow the `tutorials <tutorials.html>`_ to explore more advanced use cases such as:

- Using different VGSL spec strings to define custom architectures.
- Switching between TensorFlow and PyTorch backends.
- Integrating VGSLify models into larger deep learning workflows.

Check out the `API reference <source/vgslify.html>`_ for detailed information on all available classes, methods, and utilities in VGSLify.

Additional Topics
-----------------
For more examples and advanced workflows, continue reading the `Tutorials <tutorials.html>`_ and `Advanced Usage <advanced_usage.html>`_ sections.
