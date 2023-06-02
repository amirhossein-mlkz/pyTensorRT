Getting start
===================

step 1: convert model to onnx
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the first step, you have to convert your model to the standard and common ``.onnx`` format.
This section may be different for TensorFlow and PyTorge modules. In this section, we create a model with tensorflow and convert it to ``onnx`` format
to convert a tensorflow model into an onnx model, we use ``modelConverter`` modules and its ``Converter`` class



.. code-block:: python

   from tensorflow import keras
   import numpy as np
   from modelConverter import Converter

now lets build a test model Using keras and tensorflow

.. code-block:: python

   base_model = keras.applications.resnet.ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=(300,300,3), pooling=None, classes=1000)
   y = keras.layers.Flatten()( base_model.layers[-1].output)
   y = keras.layers.Dense(200, activation='relu')(y)
   y = keras.layers.Dense(200, activation='relu')(y)
   y = keras.layers.Dense(10, activation='softmax')(y)
   x = base_model.layers[0].input
   model = keras.models.Model(x,y)
   model.compile(loss='categorical_crossentropy')

   model.load_weights('model.h5')

to convert a TensorFlow model into a .onnx model, we should save the model in .pb format.
``Converter.pb.kerasmodel_to_pb`` method get two arguments. first one is the model and the second is your desired path to save the model as .pb format 



.. code-block:: python
   
   #save model as pb 
   Converter.pb.kerasmodel_to_pb(model, 'model/')

After executing this part of the code, a folder named ``model`` will be created in the project path, which will hold the files related to the ``.pb`` format.

As the last step, just run the following code snippet to create an onnx moel. 
``Converter.onnx.pb_to_onnx`` method get two arguments. first one is the path of ``.pb`` folder  and the second is your desired path to save the model as ``.onnx`` format

.. code-block:: python
   
   #convert .pb model into onnx format
   Converter.onnx.pb_to_onnx('model/', 'model.onnx')




step 2: convert onnx model to tensorRT engine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Regardless of how you converted your model to ONNX format, the steps from this point forward will be the same. After creating the model in **ONNX** format, we can now easily create the **tensor RT engine**
for doing this, we using ``Converter.tensorRT.onnx_to_tensorrt`` method. this method got these arguments.

* ``onnx_path`` : path of your .onnx file that you built previously.
* ``res_path`` : your desired path to save the tensor RT model. This file should be **.plan**
* ``batch_size`` : batch size of your inputs. you can't change this after building TensorRT engine
* ``precision`` : the precision of the tensorRT model. In this version only ``fp16`` and ``fp32`` are supported
* ``cuda_idx`` : index of the GPU device that you want to build your model on it. default is 0 for the first GPU device. use this argument when you have multi GPU
* ``max_memory_size`` : The maximum amount of space that the model can occupy on the graphics card. . This entry is in MB and must be a power of 2


.. code-block:: python

   from modelConverter import Converter

   #convert onnx model into tensort rt
   Converter.tensorRT.onnx_to_tensorrt('model.onnx', 'model.plan', batch_size=8)

After executing this part of the code, a file named ``model.plan`` will be created in the project path, which is tensorRT engine


step 3: inference TensorRt
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After building the TensorRt engine, now it's time to use it. for using tensorRT engine, we use ``engineRT`` class of ``engineRT`` module. for more details see :doc:`EngineRT_src`

for first step we should load engineRT

.. note::
   set ``cuda_idx`` argument 0 if you are using one GPU device


.. code-block:: python

   import numpy as np
   from EngineRT import EngineRT

   #load tensorRT engine
   engine = EngineRT("model.plan", cuda_idx=0)


Now we give an arbitrary input to the model with the ``inference`` method. This method returns the output of the model for the given input.

.. note::
   Note that both input and output are lists of model's inputs and outputs. If your model has only one input and one output, the input should be a ``list`` of length 1 including the batch inputs ``np.array`` and the output will be a ``list`` of length 1 including the batch outputs ``np.array``


.. code-block:: python

   # Generate random input. Its shape must match the input shape of the model
   test_inputs = np.random.rand(8,300,300,3)

   #inference inputs into model
   result = engine.inference([test_inputs])

   print(f" this model has {len(result)} output")
   print(f" output shape: {result[0].shape}")

.. code-block::

   >> this model has 1 output
   >> output shape: (8, 10)
