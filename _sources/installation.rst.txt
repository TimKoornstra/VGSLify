Installation
============

Prerequisites
-------------

Before installing VGSLify, make sure your system meets the following requirements:

- **Python Version**: VGSLify requires Python 3.8 or newer. Ensure that you have the correct version installed by running the following command:

  .. code-block:: bash

     python --version

- **Required Packages**:

  - `pip`: Python's package manager is required to install VGSLify and its dependencies.
  - **TensorFlow**: If you are using VGSLify with TensorFlow, you will need to install TensorFlow as a backend.
  - **PyTorch**: If you are using VGSLify with PyTorch, you will need to install PyTorch as a backend.

- **VGSLify is BYOB (Bring Your Own Backend)**: VGSLify itself does not include a deep learning framework. Users must install their preferred backend—TensorFlow or PyTorch. This approach gives you flexibility in choosing your backend.

Installing VGSLify
------------------

You can install VGSLify in several ways, depending on whether you want the stable release or a development version.

1. **Install the latest version via pip**:

   The easiest way to get VGSLify is by using `pip`. Run the following command in your terminal:

   .. code-block:: bash

      pip install vgslify

2. **Install TensorFlow or PyTorch Backend**:

   VGSLify is a BYOB package, which means you will need to install a backend separately. If you want to use TensorFlow as the backend, you can install it with the following command:

   .. code-block:: bash

      pip install tensorflow

   Or if you want to use PyTorch as the backend, you can install it with the following command:

   .. code-block:: bash

      pip install torch

3. **Install the Development Version from Source**:

   If you want to work with the development version or modify VGSLify, you can install it directly from the source repository. Follow these steps:

   .. code-block:: bash

      git clone https://github.com/TimKoornstra/vgslify.git
      cd vgslify
      pip install .

   This will install VGSLify and all of its dependencies in your environment.

Verifying Installation
----------------------

After installation, you can verify that VGSLify has been successfully installed and is functioning correctly by running the following command:

.. code-block:: bash

   python -c "import vgslify; print(vgslify.__version__)"

This should print the installed version of VGSLify without any errors. If the version is displayed correctly, the installation is successful.

