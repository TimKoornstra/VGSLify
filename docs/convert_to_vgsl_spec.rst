Converting Models Back to VGSL Spec
===================================

VGSLify now includes the ability to convert a trained or existing model back into a VGSL specification string. This functionality is useful for:

- Sharing model architectures in a concise format.
- Reproducing models from the VGSL spec string.
- Analyzing and understanding complex models via their VGSL representation.

How It Works
------------

After you build or load a model using TensorFlow or PyTorch, you can convert it back into its VGSL specification string using the `model_to_spec()` function provided by VGSLify.

Example: Convert a Model to VGSL Spec
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here’s how you can convert an existing model to its VGSL spec:

.. code-block:: python

    from vgslify import model_to_spec
    from tensorflow.keras.models import load_model

    # Load an existing TensorFlow model (previously saved)
    model = load_model("path_to_your_model.keras")

    # Convert the model to VGSL spec
    vgsl_spec = model_to_spec(model)
    print(vgsl_spec)

The above example will output the VGSL spec string corresponding to the architecture of the loaded model.

Saving and Reusing VGSL Spec
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you've converted the model to a VGSL spec, you can easily save or share the spec string. This can be reused to rebuild the same model using VGSLify.

1. **Save the VGSL Spec**:

   - Save the generated VGSL spec string to a file or store it in your project for later use.

.. code-block:: python

    with open("model_spec.txt", "w") as f:
        f.write(vgsl_spec)

2. **Rebuild the Model from the Spec**:

   - You can use the saved VGSL spec to rebuild the exact same model at any time with either TensorFlow or PyTorch backend.

.. code-block:: python

    from vgslify import VGSLModelGenerator

    # Load the VGSL spec from file
    with open("model_spec.txt", "r") as f:
        vgsl_spec = f.read()

    # Rebuild the model from the spec
    vgsl_gn = VGSLModelGenerator(backend="auto")
    model = vgsl_gn.generate_model(vgsl_spec)

By using this functionality, you can quickly share, reproduce, and analyze deep learning models in a concise format.


