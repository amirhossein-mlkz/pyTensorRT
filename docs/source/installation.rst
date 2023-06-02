Installation
===============

Requirement
------------
To use pyTensorRT, first install requirement:

.. note::
   versions compatibility is so important, please be sure all modules you installed and all files you download are compatible

.. note::
   at the end of this section, we introduce some tested compatibles versions of Modules

.. code-block:: console

   (.venv) $ pip install tf2onnx
   (.venv) $ pip install pycuda


Install CUDA Toolkit
^^^^^^^^^^^^^^^^^^^^^^
Follow this `link <https://developer.nvidia.com/cuda-toolkit-archive>`_ to download and install compatible version of CUDA Toolkit and install it.


Install CUDNN
^^^^^^^^^^^^^^^^^^^^^^
* go to `cudnn-archive <https://developer.nvidia.com/cuda-toolkit-archive>`_
* Create a user profile if needed and log in
* Download a cudnn that its version is compatible with your cuda-toolkit version that is installed

.. tabs::

   .. tab:: Windows
      Extract the contents of the zip file (i.e. the folder named cuda) inside ``<INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v11.2\``, where ``<INSTALL_PATH>`` points to the installation directory specified during the installation of the CUDA Toolkit. By default ``<INSTALL_PATH> = C:\Program Files``.
   
   .. tab:: Linux
      Follow the instructions under Section 2.3.1 of the `CuDNN Installation Guide <https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-linux>`_ to install CuDNN.


install TensorRT
^^^^^^^^^^^^^^^^^^^^

.. tabs::

   .. tab:: Windows
      
      - download a compatible version of TensorRt from `here <https://docs.nvidia.com/deeplearning/tensorrt/archives/index.html>`_
      - Extract the zip file
      - add ``TensorRT-x.x.x.x\lib`` folder to environment path
      - go to ``tensorrt_path/python`` path , open cmd and install compatible tensorrt .whl file base on your python version 

      ``python.exe -m pip install FILE_NAME.whl``

   .. tab:: Linux
      
compatible versions

======================

.. list-table:: madules compatibility
   :header-rows: 1

   * - CUDA Toolkit
     - Cudnn
     - tensorRT
     - pycuda
     - tf2onnx
     - tensorflow
   * - 11.2
     - 8.1.0
     - 8.2.1.8
     - 2022.2.2
     - 1.13.0
     - 2.10 

