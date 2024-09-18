Tutorials
=========

Tutorial 1: Building a CNN for Image Classification
---------------------------------------------------

Overview
~~~~~~~~

- Create a simple CNN model using VGSLify for image classification (e.g., on CIFAR-10).
- Step-by-step explanation of building, training, and evaluating the model.

Step-by-Step Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

1. Import required libraries.
   
   .. code-block:: python

      import tensorflow as tf
      from vgslify.generator import VGSLModelGenerator

2. Load and preprocess the dataset (e.g., CIFAR-10).

3. Define the VGSL spec string for a CNN.

   .. code-block:: python

      vgsl_spec = "None,32,32,3 Cr3,3,32 Mp2,2,2,2 Cr3,3,64 Mp2,2,2,2 Rc Fc128 D25 O1s10"

4. Build and compile the model using VGSLify.

   .. code-block:: python

      vgsl_gn = VGSLModelGenerator(backend="tensorflow")
      model = vgsl_gn.generate_model(vgsl_spec)
      model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

5. Train the model with the CIFAR-10 dataset.

6. Evaluate the model performance and visualize the results.

   .. code-block:: python

      test_loss, test_acc = model.evaluate(x_test, y_test)
      print(f'Test accuracy: {test_acc}')

Tutorial 2: Creating an LSTM for Sequence Prediction
----------------------------------------------------

Overview
~~~~~~~~

- Build an LSTM model for sequence prediction using VGSLify.
- Example: Predict the next value in a time-series dataset.

Step-by-Step Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

1. Import necessary libraries.
   
   .. code-block:: python

      import numpy as np
      from vgslify.generator import VGSLModelGenerator

2. Generate synthetic sequence data (e.g., sine wave).

3. Define the VGSL spec string for an LSTM model.

   .. code-block:: python

      vgsl_spec = "None,None,1 Lf50s D20 O1s1"

4. Build and compile the model.

5. Train and evaluate the model.

Next Steps
----------

- Link to advanced tutorials.

