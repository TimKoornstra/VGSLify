Advanced Usage and Customization
================================

VGSLify is designed to be highly extensible. This section covers advanced topics such as:

- Registering **custom layers** to extend VGSLify’s capabilities.
- Creating **custom parsers** to enable VGSL spec conversion for non-standard layers.
- Debugging VGSL strings and best practices for troubleshooting.

Custom Layer Registration
-------------------------
VGSLify allows you to extend its functionality by registering **custom layer builder functions**. This is especially useful if you need to add experimental or framework-specific layers.

How Custom Layers Work
~~~~~~~~~~~~~~~~~~~~~~

1. **Decorators and Prefixes**: Register a custom layer using `@register_custom_layer("<prefix>")`.  

   - The prefix is a short identifier (e.g., `"Xcustom"`) used in the VGSL spec.  

2. **Function Signature**: The builder function must accept:

   - `factory`: A `TensorFlowLayerFactory` or `TorchLayerFactory` instance.
   - `spec`: The entire VGSL token that starts with your custom prefix.

3. **Return Value**: Your function must return a valid layer:

   - **TensorFlow**: A `tf.keras.layers.Layer` or `tf.keras.Sequential` model.
   - **PyTorch**: An `nn.Module` or `nn.Sequential` module.

Once registered, if VGSLify encounters a VGSL spec token with your prefix (e.g., `"Xcustom"`), it will call your function to construct the layer.

TensorFlow Example
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from vgslify.tensorflow import register_custom_layer
   import tensorflow as tf

   @register_custom_layer("Xcustom")
   def build_custom_layer(factory, spec):
       """
       Builds a custom TensorFlow layer or layer block.

       Parameters
       ----------
       factory : TensorFlowLayerFactory
           The factory object that manages layer creation in TensorFlow.
       spec : str
           The VGSL spec token starting with 'Xcustom'. Example: "Xcustom,128".

       Returns
       -------
       tf.keras.layers.Layer
           A valid TensorFlow layer or Sequential model.
       """
       dense = tf.keras.layers.Dense(128, activation=None)
       bn = tf.keras.layers.BatchNormalization()
       return tf.keras.Sequential([
           dense,
           bn,
           tf.keras.layers.Activation('relu')
       ])

PyTorch Example
~~~~~~~~~~~~~~~

.. code-block:: python

   from vgslify.torch import register_custom_layer
   import torch.nn as nn

   @register_custom_layer("Xcustom")
   def build_custom_layer(factory, spec):
       """
       Builds a custom PyTorch layer or module.

       Parameters
       ----------
       factory : TorchLayerFactory
           The factory object that manages layer creation in PyTorch.
       spec : str
           The VGSL spec token starting with 'Xcustom'. Example: "Xcustom,128".

       Returns
       -------
       nn.Module
           A valid PyTorch module or Sequential model.
       """
       in_features = factory.shape[-1]  # The current output shape from the previous layer
       linear = nn.Linear(in_features, 128)
       dropout = nn.Dropout(p=0.3)
       activation = nn.ReLU()
       return nn.Sequential(linear, dropout, activation)

Using Custom Layers in a VGSL Spec
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once registered, you can use the custom prefix in a VGSL spec:

.. code-block:: text

   None,28,28,1 Cr3,3,32 Xcustom Mp2,2 Rc2 Fr64 Fs10

When `VGSLModelGenerator` processes `"Xcustom"`, it will call the corresponding `build_custom_layer()` function.

Custom Parser Registration
--------------------------
VGSLify also supports **custom model parsers**, which enable conversion from deep learning models back to VGSL spec strings.

Why Register a Custom Parser?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **For non-standard layers**: If you have proprietary or experimental layers, the built-in parser may not recognize them.
- **For accurate model conversion**: You define how the layer should be represented in VGSL format.

How Custom Parsers Work
~~~~~~~~~~~~~~~~~~~~~~~

1. **Decorator-Based Registration**: Use `@register_custom_parser(SomeLayerClass)`.
2. **Function Signature**:

   - `layer`: The actual layer instance from a `tf.keras.Model` or `nn.Module`.
   - **Return**: A valid VGSL spec token (e.g., `"XmyLayer,128"`).

TensorFlow Model Parser Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from vgslify.model_parsers.tensorflow import register_custom_parser
   import tensorflow as tf

   class MyCustomKerasLayer(tf.keras.layers.Layer):
       # Custom layer logic here
       pass

   @register_custom_parser(MyCustomKerasLayer)
   def parse_my_custom_layer(layer):
       """
       Converts MyCustomKerasLayer instance to a VGSL spec string.

       Parameters
       ----------
       layer : MyCustomKerasLayer
           The actual layer instance from a tf.keras.Model.

       Returns
       -------
       str
           A VGSL spec token such as "XmyLayer".
       """
       return "XmyLayer"

PyTorch Model Parser Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from vgslify.model_parsers.torch import register_custom_parser
   import torch.nn as nn

   class MyCustomTorchLayer(nn.Module):
       def __init__(self, features):
           super().__init__()
           self.linear = nn.Linear(features, 64)

   @register_custom_parser(MyCustomTorchLayer)
   def parse_my_custom_torch_layer(layer):
       """
       Converts MyCustomTorchLayer instance to a VGSL spec token.

       Parameters
       ----------
       layer : MyCustomTorchLayer
           The actual PyTorch layer instance from an nn.Module.

       Returns
       -------
       str
           A VGSL spec token like "XmyTorchLayer,64".
       """
       out_features = layer.linear.out_features
       return f"XmyTorchLayer,{out_features}"

Full Round-Trip Support
~~~~~~~~~~~~~~~~~~~~~~~

If both a **custom layer builder** and a **custom parser** are registered:

- You can **build** models from VGSL specs that contain `"XmyLayer"`.
- You can **convert** models back to VGSL spec, ensuring the same `"XmyLayer"` appears in the output.

Extending Model Conversion
--------------------------
VGSLify provides the ``model_to_spec()`` utility to convert existing models back into VGSL specifications. Advanced users can modify the output spec string to tweak or optimize the architecture. This is particularly useful for:
 
- **Reproducing models**: Share a compact spec string instead of verbose code.
- **Architecture search**: Programmatically adjust VGSL strings and rebuild models for hyperparameter tuning.
- **Debugging**: Compare the generated spec with your intended design.

Debugging and Best Practices
-----------------------------
When working with VGSL strings:

- **Validate Your Spec**: Use simple, incremental specs. Start with an input layer and add one layer at a time, verifying the output shape.
- **Use the Generator History**: The ``generate_history()`` method of ``VGSLModelGenerator`` returns a list of intermediate layers. This can help you inspect and debug layer shapes before building the final model.
  
  .. code-block:: python

      vgsl_gn = VGSLModelGenerator(backend="tensorflow")
      history = vgsl_gn.generate_history("None,28,28,1 Cr3,3,32 Mp2,2 Rc2 Fr64 D20 Fs10")
      for idx, layer in enumerate(history):
          print(f"Layer {idx}: {layer}")

- **Custom Parsers**: Advanced users may want to write custom parsers for new types of layers. Both the TensorFlow and PyTorch model parsers support decorators for custom parser registration. Check the source code and examples in the corresponding modules.

Best Practices Summary:

  - Start simple and test each layer’s output shape.
  - Use custom registrations to keep your VGSL strings clean.
  - Convert models back to VGSL specs to verify consistency.
  - Leverage the history method to debug complex architectures.

Integrating VGSLify in Larger Workflows
----------------------------------------
VGSLify’s design encourages easy integration with existing deep learning pipelines. For instance, after generating a model, you can directly plug it into your training framework or integrate it with experiment tracking tools like TensorBoard or Weights & Biases.

Example integration with TensorBoard:

.. code-block:: python

   import tensorflow as tf
   from vgslify import VGSLModelGenerator

   vgsl_spec = "None,28,28,1 Cr3,3,32 Mp2,2 Rc2 Fr64 D20 Fs10"
   vgsl_gn = VGSLModelGenerator(backend="tensorflow")
   model = vgsl_gn.generate_model(vgsl_spec)
   model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
   tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
   model.fit(x_train, y_train, epochs=10, callbacks=[tensorboard_callback])

This flexibility makes VGSLify suitable not only for prototyping but also for production-level model management.
