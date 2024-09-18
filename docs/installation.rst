Installation
============

Prerequisites
-------------

- Python version: Ensure Python 3.8 or newer is installed.
- Verify Python version using:

  .. code-block:: bash

     python --version

- Required packages:
  - `pip`
  - TensorFlow (for TensorFlow users)
  - PyTorch (planned for future releases)

Installing VGSLify
------------------

1. Install VGSLify via pip:

   .. code-block:: bash

      pip install vgslify

2. Install TensorFlow backend:

   .. code-block:: bash

      pip install tensorflow

3. (Optional) Install development version from source:

   .. code-block:: bash

      git clone https://github.com/your-repo/vgslify.git
      cd vgslify
      pip install .

Verifying Installation
----------------------

- Test that VGSLify was installed correctly:

  .. code-block:: bash

     python -c "import vgslify; print(vgslify.__version__)"

- Ensure no errors occur.

