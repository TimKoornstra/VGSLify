Supported Layers
================

VGSLify supports a range of layers that can be specified using the VGSL format. Each layer type has its own configuration format, allowing you to define models concisely and flexibly. This section provides an overview of the supported layers and their VGSL specifications.

Layer Specifications
--------------------

**Input Layer**
^^^^^^^^^^^^^^^

- **VGSL Spec**: `<batch_size>,<height>,<width>[,<depth>,<channels>]`
- **Description**: Defines the input shape for the model, where the first value is the batch size (set to `None` for variable), followed by the height, width, and optionally the depth and channels.
- **Example**: `None,28,28,1`

  - Defines an input layer with variable batch size, height and width of 28, and 1 channel (e.g., for grayscale images).

**Conv2D Layer**
^^^^^^^^^^^^^^^^

- **VGSL Spec**: `C(s|t|r|l|m),<x>,<y>,[<s_x>,<s_y>,]<d>`
- **Description**: Defines a 2D convolutional layer with a kernel size of `<x>` by `<y>`, optional strides `<s_x>,<s_y>`, and `<d>` filters. Activation functions are specified as follows:

  - `s`: Sigmoid
  - `t`: Tanh
  - `r`: ReLU
  - `l`: Linear
  - `m`: Softmax

- **Example**: `Cr3,3,32`

  - Adds a convolutional layer with ReLU activation, a 3x3 kernel, default strides (1,1), and 32 filters.

**Pooling2D Layer**
^^^^^^^^^^^^^^^^^^^

- **VGSL Spec**: `<p>(<x>,<y>[,<s_x>,<s_y>])`

  - `Mp` for max-pooling, `Ap` for average pooling.

- **Description**: Specifies a pooling operation, which reduces the spatial dimensions by applying a window of `<x>` by `<y>` and strides of `<s_x>,<s_y>`. If strides are not specified, they default to the pool size.
- **Example**: `Mp2,2,1,1`

  - Defines a max-pooling layer with a pool size of 2x2 and strides of 1x1.

**Dense (Fully Connected) Layer**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **VGSL Spec**: `F(s|t|r|l|m)<d>`
- **Description**: Defines a fully connected (dense) layer with `<d>` units. The non-linearity can be:

  - `s`: Sigmoid
  - `t`: Tanh
  - `r`: ReLU
  - `l`: Linear
  - `m`: Softmax

- **Example**: `Fr64`

  - Adds a dense layer with 64 units and ReLU activation.

**RNN Layer (LSTM/GRU/Bidirectional)**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **VGSL Spec**: `L(f|r)[s]<n>[,D<rate>,Rd<rate>]` for LSTM/GRU, `B(g|l)<n>[,D<rate>,Rd<rate>]` for Bidirectional RNN
- **Description**: Specifies an RNN layer with `n` units. The optional dropout `D` and recurrent dropout `Rd` rates can be included.

  - `L`: LSTM
  - `G`: GRU
  - `B`: Bidirectional
  - `f`: Forward direction, `r`: Reverse direction, `g`: GRU, `l`: LSTM

- **Example**: `Lf64,D50,Rd25`

  - Defines an LSTM layer with 64 units, 50% dropout, and 25% recurrent dropout.

**Dropout Layer**
^^^^^^^^^^^^^^^^^

- **VGSL Spec**: `D<rate>`
- **Description**: Specifies a dropout layer, where `<rate>` is the dropout percentage (0–100).
- **Example**: `D50`

  - Adds a dropout layer with a 50% dropout rate.

**Reshape Layer**
^^^^^^^^^^^^^^^^^

- **VGSL Spec**: `Rc2`, `Rc3`, or `R<x>,<y>,<z>`
- **Description**: The Reshape layer reshapes the output tensor from the previous layer. It has two primary functions:

  - **Rc2**: Collapses the spatial dimensions (height, width, and channels) into a 2D tensor. This is typically used when transitioning to a fully connected (dense) layer. 

    - Example: Reshaping from `(batch_size, height, width, channels)` to `(batch_size, height * width * channels)`.

  - **Rc3**: Collapses the spatial dimensions into a 3D tensor suitable for RNN layers. This creates a 3D tensor in the form of `(batch_size, time_steps, features)`.

    - Example: Reshaping from `(batch_size, height, width, channels)` to `(batch_size, height * width, channels)` for input to LSTM or GRU layers.

  - **R<x>,<y>,<z>**: Directly reshapes to the specified target shape.

- **Example**:

  - `Rc2` collapses the output from `(None, 8, 8, 64)` to `(None, 4096)` for a fully connected layer.
  - `Rc3` collapses the output from `(None, 8, 8, 64)` to `(None, 64, 64)` for input to an RNN layer.
  - `R64,64,3` reshapes the output to `(None, 64, 64, 3)`.

Custom Layers
-------------
Custom layers can be defined using a custom prefix (e.g., "Xsw" or "Xcustom"). Once registered, these layers are invoked automatically when the spec string begins with the custom prefix.

For example, a custom activation or a compound layer can be added via the methods described in the [Advanced Usage](advanced_usage.html) section.

More Examples
-------------

Explore additional examples and advanced configurations in the `tutorials <tutorials.html>`_.

