Getting Started
===============

Overview
--------

- Introduction to VGSL specifications.
  - VGSL strings represent deep learning models.
  - Example of a VGSL string for a basic model.
  
- Key functionality of VGSLify:
  - Building models with a single line.
  - Switching between TensorFlow and PyTorch backends.

Simple Example: Building a Model
--------------------------------

1. Import VGSLModelGenerator:

   .. code-block:: python

      from vgslify.generator import VGSLModelGenerator

2. Define the VGSL spec string:

   .. code-block:: python

      vgsl_spec = "None,28,28,1 Cr3,3,32 Mp2,2,2,2 Rc Fc64 D20 O1s10"

3. Build the model:

   .. code-block:: python

      vgsl_gn = VGSLModelGenerator(backend="tensorflow")
      model = vgsl_gn.generate_model(vgsl_spec)
      model.summary()

Explanation of Layers
---------------------

- **Input Layer**: ``None,28,28,1`` defines the input shape (e.g., for MNIST images).
- **Convolutional Layer**: ``Cr3,3,32`` adds a 2D convolution layer.
- **MaxPooling Layer**: ``Mp2,2,2,2`` reduces the spatial dimensions.
- **Reshape Layer**: ``Rc`` reshapes the data to feed into a dense layer.
- **Fully Connected Layer**: ``Fc64`` adds a dense layer with 64 units.
- **Dropout Layer**: ``D20`` applies a 20% dropout to prevent overfitting.
- **Output Layer**: ``O1s10`` represents the output layer for 10 classes with softmax activation.

Next Steps
----------

- Link to tutorials for more advanced examples.

