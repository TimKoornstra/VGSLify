VGSL Specification Guide
=========================

Overview of VGSL
----------------

- Introduction to the VGSL (Variable-size Graph Specification Language) format.
- How VGSL specifies neural network architectures using short strings.
- Explanation of key components and syntax.

VGSL Spec String Format
-----------------------

- **Input Layer**: ``batch_size, height, width, channels``.
- **Convolutional Layer**: ``C<activation>,<filter_x>,<filter_y>,<num_filters>``.
- **Pooling Layer**: ``Mp<filter_x>,<filter_y>,<stride_x>,<stride_y>``.
- **Fully Connected Layer**: ``Fc<units>``.
- **Dropout Layer**: ``D<dropout_rate>``.
- **Output Layer**: ``O<dims><activation><num_classes>``.

Layer Examples
--------------

- Input Layer:

  .. code-block:: rst

     None,28,28,1

- Convolutional Layer:

  .. code-block:: rst

     Cr3,3,64

- MaxPooling Layer:

  .. code-block:: rst

     Mp2,2,2,2

- Fully Connected Layer:

  .. code-block:: rst

     Fc128

- Output Layer:

  .. code-block:: rst

     O1s10

Common Patterns
---------------

- How to combine convolutional and pooling layers.
- How to use reshaping layers to connect CNNs to fully connected layers.
- How to apply regularization (e.g., dropout) in VGSL specs.

Advanced Usage
--------------

- Using VGSL with RNNs (e.g., LSTM, GRU).
- Bidirectional and stacked RNNs in VGSL specs.

