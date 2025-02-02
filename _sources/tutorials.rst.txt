Tutorials
=========

This section provides hands-on tutorials for using VGSLify to build and train deep learning models. Follow these step-by-step guides to get familiar with how VGSLify simplifies model creation through VGSL specifications.

Tutorial 1: Building a CNN for Image Classification
---------------------------------------------------

Overview
~~~~~~~~

In this tutorial, you will build a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. We will define the model using a VGSL spec string, which allows us to specify the architecture in a concise, human-readable format. By the end of this tutorial, you will have a fully trained CNN model for image classification.

Step-by-Step Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Import required libraries**:
   
   Begin by importing the necessary libraries for TensorFlow and VGSLify.

   .. code-block:: python

      import tensorflow as tf
      from vgslify import VGSLModelGenerator

2. **Load and preprocess the dataset**:

   CIFAR-10 is a dataset of 60,000 32x32 color images in 10 classes, with 6,000 images per class. You can load and preprocess the dataset as follows:

   .. code-block:: python

      (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

      # Normalize the images to the range [0, 1]
      x_train, x_test = x_train / 255.0, x_test / 255.0

      # Convert labels to one-hot encoding
      y_train = tf.keras.utils.to_categorical(y_train, 10)
      y_test = tf.keras.utils.to_categorical(y_test, 10)

3. **Define the VGSL spec string for the CNN**:

   The VGSL spec string defines the layers of the CNN. Here's a simple CNN architecture:

   .. code-block:: python

      vgsl_spec = "None,32,32,3 Cr3,3,32 Mp2,2 Rc2 Cr3,3,64 Mp2,2 Rc2 Fr128 D25 Fs10"

   Explanation:

   - `None,32,32,3`: Input layer for images of size 32x32 with 3 color channels (RGB).
   - `Cr3,3,32`: Convolutional layer with a 3x3 filter, ReLU activation, and 32 filters.
   - `Mp2,2`: MaxPooling layer with a 2x2 pool size (and default strides).
   - `Cr3,3,64`: Second convolutional layer with 64 filters.
   - `Rc2`: Reshape layer to flatten the output for the fully connected layer.
   - `Fr128`: Fully connected (dense) layer with 128 units and ReLU activation.
   - `D25`: Dropout layer with a 25% dropout rate.
   - `Fs10`: Output layer with 10 units and softmax activation for classification into 10 classes.

4. **Build and compile the model**:

   Use VGSLify to build and compile the model. This step generates the CNN architecture based on the VGSL string and compiles it for training.

   .. code-block:: python

      vgsl_gn = VGSLModelGenerator(backend="tensorflow")
      model = vgsl_gn.generate_model(vgsl_spec)

      model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

5. **Train the model**:

   Now, train the CNN on the CIFAR-10 training set. You can adjust the batch size and number of epochs as needed.

   .. code-block:: python

      history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

6. **Evaluate the model performance**:

   After training, evaluate the model on the test set to see how well it performs.

   .. code-block:: python

      test_loss, test_acc = model.evaluate(x_test, y_test)
      print(f'Test accuracy: {test_acc}')

   You can also plot the training history to visualize how the accuracy and loss evolve over time.

Tutorial 2: Creating an LSTM for Sequence Prediction
----------------------------------------------------

Overview
~~~~~~~~

In this tutorial, you will build an LSTM (Long Short-Term Memory) model using VGSLify to predict the next value in a sequence. This is commonly used in time-series forecasting. We will generate synthetic data, define an LSTM model using a VGSL string, and train the model to predict future values in the sequence.

Step-by-Step Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Import necessary libraries**:

   .. code-block:: python

      import numpy as np
      from vgslify import VGSLModelGenerator

2. **Generate synthetic sequence data**:

   For this example, let's generate a sine wave as our synthetic sequence data. The LSTM will learn to predict the next value in this sequence.

   .. code-block:: python

      def generate_sine_wave(seq_length=1000):
          x = np.arange(seq_length)
          y = np.sin(x / 20.0)
          return y

      sine_wave = generate_sine_wave()

      # Prepare the data for LSTM input
      def create_sequences(data, seq_length):
          x = []
          y = []
          for i in range(len(data) - seq_length):
              x.append(data[i:i+seq_length])
              y.append(data[i+seq_length])
          return np.array(x), np.array(y)

      seq_length = 50
      x_train, y_train = create_sequences(sine_wave, seq_length)

      x_train = np.expand_dims(x_train, axis=-1)  # LSTM expects input shape (batch, time steps, features)
      y_train = np.expand_dims(y_train, axis=-1)

3. **Define the VGSL spec string for the LSTM model**:

   Here's the VGSL string to define an LSTM with 50 units, followed by dropout and an output layer:

   .. code-block:: python

      vgsl_spec = f"None,50,1 Lf50 D20 Fl1"

   Explanation:

   - `None,seq_length,x_train.shape[1]`: Input shape with 50 sequence length and 50 features.
   - `Lf50`: LSTM with 50 units, without returning sequences.
   - `D20`: Dropout layer with 20% dropout rate.
   - `Fl1`: Output layer with 1 unit and linear activation for sequence prediction.

4. **Build and compile the model**:

   .. code-block:: python

      vgsl_gn = VGSLModelGenerator(backend="tensorflow")
      model = vgsl_gn.generate_model(vgsl_spec)

      model.compile(optimizer='adam',
                    loss='mean_squared_error')

5. **Train the model**:

   Train the model to predict the next value in the sine wave sequence.

   .. code-block:: python

      history = model.fit(x_train, y_train, epochs=20, batch_size=64)

6. **Evaluate the model**:

   Once training is complete, evaluate the model by plotting the true vs predicted values in the sine wave sequence.

   .. code-block:: python

      y_pred = model.predict(x_train)

      import matplotlib.pyplot as plt
      plt.plot(y_train, label='True')
      plt.plot(y_pred, label='Predicted')
      plt.legend()
      plt.show()

Tutorial 3: Converting Models Back to VGSL Specification
--------------------------------------------------------
VGSLify not only builds models from VGSL specs—it can also convert an existing model back to its VGSL string.

1. **Load an existing model**:

   .. code-block:: python

      from tensorflow.keras.models import load_model
      model = load_model("path_to_your_model.keras")

2. **Convert the model to a VGSL spec string**:

   .. code-block:: python

      from vgslify import model_to_spec
      vgsl_spec = model_to_spec(model)
      print("VGSL spec:", vgsl_spec)

3. **Modify and reuse the spec**:

   The generated VGSL string can be saved, shared, or modified. Rebuild the model later using:

   .. code-block:: python

      vgsl_gn = VGSLModelGenerator(backend="tensorflow")
      model = vgsl_gn.generate_model(vgsl_spec)

Additional Topics
-----------------
For more examples and advanced workflows, continue reading the `Advanced Usage <advanced_usage.html>`_ or `API Reference <source/vgslify.html>`_ sections.
